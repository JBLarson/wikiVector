#!/usr/bin/env python3
"""
WIKIPEDIA EMBEDDINGS GENERATION SYSTEM
Production-grade pipeline for generating semantic embeddings of all English Wikipedia articles.

Architecture:
- Multi-process XML parsing (Producer-Consumer)
- GPU-optimized batched encoding with FP16
- Incremental FAISS index building
- Robust checkpointing every 100K articles
- Comprehensive validation and quality checks
"""

import os
import sys
import bz2
import json
import time
import sqlite3
import hashlib
import logging
import re
import shutil
from glob import glob
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterator, List, Tuple, Optional
from multiprocessing import Pool, Manager, Process, Queue
from queue import Empty
from lxml import etree as ET

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - PRODUCTION
# ============================================================================

@dataclass
class Config:
    """System configuration parameters"""
    
    # --- PRODUCTION PATHS ---
    xml_chunk_dir: str = "/mnt/data/wikipedia/raw/xml_chunks" 
    output_dir: str = "/mnt/data/wikipedia/embeddings/"
    checkpoint_dir: str = "/mnt/data/wikipedia/checkpoints/"
    
    # Model configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_seq_length: int = 512
    
    # --- PRODUCTION TUNING ---
    # Use (CPUs - 1) for parsing
    num_workers: int = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    batch_size: int = 512
    checkpoint_interval: int = 100_000 # Original 100k
    use_fp16: bool = True # Enable for NVIDIA GPU
    
    # Validation
    quick_validation_size: int = 10_000
    validation_queries: List[Tuple[str, List[str]]] = None
    
    # Filtering
    min_article_length: int = 100 # Your setting
    # max_article_length is REMOVED. We truncate instead.
    
    def __post_init__(self):
        """Initialize validation queries"""
        if self.validation_queries is None:
            self.validation_queries = [
                ("programming languages", ["Python_(programming_language)", "JavaScript", "Java_(programming_language)"]),
                ("machine learning", ["Machine_learning", "Deep_learning", "Neural_network"]),
                ("European countries", ["France", "Germany", "United_Kingdom"]),
                ("physics concepts", ["Quantum_mechanics", "General_relativity", "Thermodynamics"]),
                ("Renaissance artists", ["Leonardo_da_Vinci", "Michelangelo", "Raphael"]),
            ]

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config):
    """Configure logging with both file and console output"""
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"embeddings_{timestamp}.log"
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def get_worker_logger(worker_id: int):
    """Creates a dedicated logger for a worker process"""
    logger = logging.getLogger(f"Worker-{worker_id}")
    
    if logger.hasHandlers():
        return logger

    root_logger = logging.getLogger()
    file_handler = None
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler):
            file_handler = h
            break
            
    if file_handler:
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Could not find root FileHandler for Worker-{worker_id}. Logging to console.")

    logger.propagate = False
    return logger

# ============================================================================
# XML PARSING (WORKER)
# ============================================================================

@dataclass
class Article:
    """Represents a parsed Wikipedia article"""
    page_id: int
    title: str
    namespace: int
    text: str

def _clean_wikitext(text: str) -> str:
    """Static version of the cleaner for multiprocessing."""
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*\/>', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[http[^\]]*\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def xml_parser_worker(
    worker_id: int,
    xml_file_path: str, 
    queue: Queue,
    config: Config
):
    """Producer function - parses XML chunks"""
    logger = get_worker_logger(worker_id)
    
    logger.info(f"Worker has started. Processing file: {xml_file_path}")
    
    try:
        namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
        page_count = 0
        skipped_count = 0
        put_count = 0
        
        open_func = bz2.open if xml_file_path.endswith('.bz2') else open
        
        with open_func(xml_file_path, 'rb') as f:
            logger.info(f"File opened with {open_func.__name__}. Starting iterparse...")
            
            context = ET.iterparse(f, events=('end',), tag=f"{namespace}page")
            
            for event, elem in context:
                page_count += 1
                if page_count % 20000 == 0:
                    logger.info(f"Parsed {page_count} pages so far... (Skipped: {skipped_count}, Queued: {put_count})")
                    
                try:
                    title_elem = elem.find(f"{namespace}title")
                    if title_elem is None or not title_elem.text:
                        skipped_count += 1
                        continue
                    title = title_elem.text.strip().replace(" ", "_")

                    ns_elem = elem.find(f"{namespace}ns")
                    ns_value = int(ns_elem.text) if ns_elem is not None else 0
                    
                    id_elem = elem.find(f"{namespace}id")
                    if id_elem is None:
                        skipped_count += 1
                        continue
                    page_id = int(id_elem.text)
                        
                    if elem.find(f"{namespace}redirect") is not None:
                        skipped_count += 1
                        continue
                        
                    revision = elem.find(f"{namespace}revision")
                    if revision is None:
                        skipped_count += 1
                        continue
                        
                    text_elem = revision.find(f"{namespace}text")
                    if text_elem is None or not text_elem.text:
                        skipped_count += 1
                        continue
                        
                    if ns_value != 0:
                        skipped_count += 1
                        continue
                    if title.startswith("List_of_"):
                        skipped_count += 1
                        continue
                    if "(disambiguation)" in title.lower():
                        skipped_count += 1
                        continue

                    text = _clean_wikitext(text_elem.text)
                    text_len = len(text)
                    
                    # --- THIS IS YOUR NEW LOGIC ---
                    # 1. Check MINIMUM length only
                    if text_len < config.min_article_length:
                        if page_count % 20000 == 0: # Log skips less often
                             logger.warning(f"Skipping {title} (length: {text_len})")
                        skipped_count += 1
                        continue
                        
                    # 2. Truncate text for storage (e.g., first 20k chars)
                    #    This is just to keep the metadata DB manageable.
                    truncated_text = text[:20000] 
                    
                    article = Article(
                        page_id=page_id,
                        title=title,
                        namespace=ns_value,
                        text=truncated_text # Save truncated text
                    )
                    
                    # 3. Model input is *already* truncated to first 2k chars
                    model_input_text = f"{title.replace('_', ' ')}. {text[:2000]}"
                        
                    queue.put((article, model_input_text))
                    put_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to parse a page in {xml_file_path}: {e}")
                
                finally:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

    except Exception as e:
        logger.error(f"WORKER FAILED FATALLY on {xml_file_path}: {e}", exc_info=True)
    finally:
        logger.info(f"Worker finished file. Total pages: {page_count}. Skipped: {skipped_count}. Queued: {put_count}.")

# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

class EmbeddingGenerator:
    """GPU-optimized embedding generation with batching"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = self._setup_device()
        self.model = self._load_model()
        
    def _setup_device(self) -> str:
        # --- PRODUCTION DEVICE SETUP ---
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            device = "cpu"
            self.logger.warning("No GPU detected, using CPU (will be much slower)")
        return device
    
    def _load_model(self) -> SentenceTransformer:
        self.logger.info(f"Loading model: {self.config.model_name}")
        model = SentenceTransformer(self.config.model_name, device=self.device)
        model.max_seq_length = self.config.max_seq_length
        # --- PRODUCTION FP16 ---
        if self.config.use_fp16 and self.device == "cuda":
            model = model.half()
            self.logger.info("Enabled FP16 inference for 2x speedup")
        _ = model.encode(["warmup"], show_progress_bar=False)
        return model
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                self.logger.error("NaN or Inf detected in embeddings!")
                return None
            return embeddings.astype(np.float32)
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.error("GPU OOM! Reduce batch size")
                torch.cuda.empty_cache()
            raise

# ============================================================================
# FAISS INDEX BUILDER
# ============================================================================

class FAISSIndexBuilder:
    """Incremental FAISS index construction with optimization"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.index = None
        self.db_path = Path(self.config.output_dir) / "metadata.db"
        self.metadata_db = self._init_metadata_db()
        
    def _init_index(self):
        """Initialize FAISS index - only call if NOT loading from checkpoint"""
        self.index = faiss.IndexFlatIP(self.config.embedding_dim)
        self.index = faiss.IndexIDMap(self.index) 
        self.logger.info(f"Initialized new FAISS index (dim={self.config.embedding_dim})")
    
    def _init_metadata_db(self) -> sqlite3.Connection:
        """Initialize SQLite database for metadata"""
        db_path = self.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_db = sqlite3.connect(str(db_path))
        cursor = metadata_db.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                idx INTEGER PRIMARY KEY,
                article_id INTEGER UNIQUE,
                title TEXT,
                namespace INTEGER,
                char_count INTEGER,
                embedding_timestamp INTEGER
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON articles(title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_id ON articles(article_id)")
        
        metadata_db.commit()
        self.logger.info(f"Initialized metadata database at {db_path}")
        return metadata_db
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        articles: List[Article]
    ):
        """Add batch of embeddings to index"""
        
        ids = np.array([a.page_id for a in articles], dtype=np.int64)
        
        self.index.add_with_ids(embeddings, ids)
        
        timestamp = int(time.time())
        cursor = self.metadata_db.cursor()
        
        cursor.executemany(
            """INSERT OR IGNORE INTO articles 
               (idx, article_id, title, namespace, char_count, embedding_timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (None, a.page_id, a.title, a.namespace, len(a.text), timestamp)
                for a in articles
            ]
        )
        
        self.metadata_db.commit()
    
    # ======================================================
    # --- PATCHED OPTIMIZE FUNCTION ---
    # ======================================================
    def optimize_index(self):
        """Convert to optimized IVF+PQ index for production"""
        self.logger.info("Optimizing index with IVF+PQ...")
        
        n_vectors = self.index.ntotal
        
        if n_vectors < 1000: # Need enough vectors to train
            self.logger.warning(f"Too few vectors ({n_vectors}) for optimization, keeping flat index")
            return

        self.logger.info(f"Reconstructing {n_vectors} vectors from flat index...")
        
        # --- START FIX ---
        # The original code tried to reconstruct from self.index (the IDMap),
        # which is not supported.
        # The correct way is to reconstruct from self.index.index (the FlatIndex)
        # using the *internal* 0-based IDs.

        # 1. Reconstruct all vectors from the underlying flat index
        vectors = np.zeros((n_vectors, self.config.embedding_dim), dtype=np.float32)
        for i in tqdm(range(n_vectors), desc="Reconstructing vectors"):
            try:
                vectors[i] = self.index.index.reconstruct(i)
            except Exception as e:
                self.logger.error(f"Failed to reconstruct internal index {i}: {e}")
                raise
        
        # 2. Get the corresponding *external* article_ids
        #    self.index.id_map contains the external_id at the internal_id's position
        self.logger.info("Fetching all article IDs from index map...")
        ids = faiss.vector_to_array(self.index.id_map)
        
        # Safety check
        if len(ids) != n_vectors:
            self.logger.error(f"FATAL: Vector count ({n_vectors}) and ID count ({len(ids)}) mismatch!")
            return
        
        self.logger.info("All vectors and IDs reconstructed.")
        # --- END FIX ---
        
        self.logger.info("All vectors reconstructed. Training new index...")

        n_clusters = int(np.sqrt(n_vectors))
        # Ensure n_clusters is valid for IVF
        n_clusters = max(1, min(n_clusters, n_vectors // 39))
        if n_clusters < 1: n_clusters = 1
        
        self.logger.info(f"Training IVF with {n_clusters} clusters...")
        
        quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
        index_ivf = faiss.IndexIVFPQ(
            quantizer,
            self.config.embedding_dim,
            n_clusters,
            64,  # 64 sub-quantizers
            8    # 8 bits per sub-quantizer
        )
        
        index_ivf.train(vectors)
        self.logger.info("Training complete.")

        self.logger.info("Creating new optimized IndexIDMap2...")
        # IndexIDMap2 is for mapping external IDs to IVF indexes
        new_index = faiss.IndexIDMap2(index_ivf)
        
        new_index.add_with_ids(vectors, ids)
        
        # Set nprobe for search (how many clusters to check)
        ivf_index_cast = faiss.downcast_index(new_index.index)
        ivf_index_cast.nprobe = min(32, n_clusters)
        
        self.logger.info(f"Index optimized: {new_index.ntotal} vectors, {n_clusters} clusters")
        
        self.index = new_index
    
    def save_final(self):
        """Save final optimized index"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        index_path = output_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        self.metadata_db.close()
        
        self.logger.info(f"Final index saved: {index_path}")
        self.logger.info(f"Metadata DB: {self.db_path}")

# ============================================================================
# VALIDATION
# ============================================================================

class ValidationSuite:
    """Comprehensive validation and quality checks"""
    
    def __init__(
        self,
        config: Config,
        index_builder: FAISSIndexBuilder,
        logger: logging.Logger
    ):
        self.config = config
        self.index = index_builder.index
        self.metadata_db = index_builder.metadata_db
        self.logger = logger
    
    def run_semantic_validation(self, model: SentenceTransformer):
        """Validate semantic search quality"""
        self.logger.info("Running semantic validation...")
        
        results = []
        
        try:
            ivf_index = faiss.downcast_index(self.index)
            ivf_index.nprobe = 32
        except:
             try:
                ivf_index = faiss.downcast_index(self.index.index)
                ivf_index.nprobe = min(32, ivf_index.nlist)
             except:
                self.logger.info("Index is Flat, nprobe not needed for validation.")
        
        for query_text, expected_titles in self.config.validation_queries:
            query_emb = model.encode([query_text], normalize_embeddings=True)
            
            query_emb = query_emb.astype(np.float32)
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            
            k = 10
            distances, article_ids = self.index.search(query_emb, k)
            
            cursor = self.metadata_db.cursor()
            result_titles = []
            
            for article_id in article_ids[0]:
                if int(article_id) == -1: continue
                cursor.execute("SELECT title FROM articles WHERE article_id = ?", (int(article_id),))
                row = cursor.fetchone()
                if row:
                    result_titles.append(row[0])
            
            hits = sum(1 for title in expected_titles if title in result_titles)
            recall = hits / len(expected_titles)
            
            self.logger.info(f"Query '{query_text}': {hits}/{len(expected_titles)} recall={recall:.2f}")
            self.logger.info(f"  > Found: {result_titles}")
            results.append(recall)
        
        avg_recall = np.mean(results)
        self.logger.info(f"Average recall@10: {avg_recall:.3f}")
        return avg_recall
    
    def run_full_validation(self) -> dict:
        """Comprehensive post-processing validation"""
        self.logger.info("Running full validation suite...")
        
        report = {
            "total_articles": self.index.ntotal,
            "embedding_dimension": self.config.embedding_dim,
            "index_type": type(self.index).__name__,
            "validation_timestamp": int(time.time())
        }
        
        cursor = self.metadata_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles")
        db_count = cursor.fetchone()[0]
        
        if db_count != self.index.ntotal:
            self.logger.error(f"Mismatch: index has {self.index.ntotal}, DB has {db_count}")
            report["coverage_error"] = True
        else:
            report["coverage_error"] = False
        
        n_searches = 100
        query_vec = np.random.randn(1, self.config.embedding_dim).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        latencies = []
        for _ in range(n_searches):
            start = time.time()
            self.index.search(query_vec, k=10)
            latencies.append((time.time() - start) * 1000)
        
        report["search_latency_ms"] = {
            "mean": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
        
        self.logger.info(f"Search latency: {report['search_latency_ms']['mean']:.2f}ms (mean)")
        
        return report

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class WikipediaEmbeddingsPipeline:
    """Main orchestration pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.logger.info("=" * 80)
        self.logger.info("WIKIPEDIA EMBEDDINGS PIPELINE (PRODUCTION)")
        self.logger.info("=" * 80)
        
        self.generator = EmbeddingGenerator(config, self.logger)
        self.index_builder = FAISSIndexBuilder(config, self.logger)
        self.validator = None 
        
        self.stats = {
            "start_time": time.time(),
            "articles_processed": 0,
            "articles_skipped": 0,
            "batches_processed": 0,
            "checkpoints_saved": 0
        }
        self.processed_ids = set() 

    def load_latest_checkpoint(self):
        """Load the most recent checkpoint if it exists"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            self.logger.info("No checkpoint directory found, starting from scratch")
            return False
        
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
        
        if not checkpoints:
            self.logger.info("No checkpoints found, starting from scratch")
            return False
        
        latest_checkpoint = checkpoints[-1]
        self.logger.info(f"Found latest checkpoint: {latest_checkpoint.name}")
        
        try:
            index_file = latest_checkpoint / "index.faiss"
            if not index_file.exists():
                self.logger.warning(f"Index file not found in {latest_checkpoint}, skipping")
                return False
            
            self.logger.info(f"Loading FAISS index from {index_file}")
            self.index_builder.index = faiss.read_index(str(index_file))
            
            checkpoint_db = latest_checkpoint / "metadata.db"
            if not checkpoint_db.exists():
                self.logger.warning(f"Metadata DB not found in {latest_checkpoint}, skipping")
                return False
            
            working_db = self.index_builder.db_path 
            self.logger.info(f"Copying metadata DB from checkpoint to {working_db}")
            
            import shutil
            shutil.copy2(checkpoint_db, working_db)
            
            self.index_builder.metadata_db.close()
            self.index_builder.metadata_db = sqlite3.connect(str(working_db))
            
            stats_file = latest_checkpoint / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
                    self.stats["start_time"] = time.time() # Reset timer
            else:
                self.stats["articles_processed"] = int(latest_checkpoint.name.split('_')[1])
            
            self.logger.info(f"✔️ Checkpoint loaded successfully")
            self.logger.info(f"  Articles processed: {self.stats['articles_processed']:,}")
            self.logger.info(f"  FAISS index size: {self.index_builder.index.ntotal:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return False

    def run_full_processing_phase(self):
        self.logger.info("=" * 80)
        self.logger.info("PHASE 2: FULL PROCESSING (Producer-Consumer)")
        self.logger.info("=" * 80)
        
        xml_files = sorted(glob(f"{self.config.xml_chunk_dir}/*.xml*"))
        if not xml_files:
            self.logger.error(f"No XML chunks found in {self.config.xml_chunk_dir}")
            raise FileNotFoundError(f"No XML chunks found in {self.config.xml_chunk_dir}")
        
        articles_already_processed = self.stats["articles_processed"]
        
        if articles_already_processed > 0:
            self.logger.info("Building set of already-processed article IDs from database...")
            cursor = self.index_builder.metadata_db.cursor()
            cursor.execute("SELECT article_id FROM articles")
            processed_ids = set(row[0] for row in cursor.fetchall())
            self.logger.info(f"Loaded {len(processed_ids):,} article IDs from database")
            self.processed_ids = processed_ids
            self.logger.info(f"RESUMING: Will filter {articles_already_processed:,} already-processed articles")
        else:
            self.processed_ids = set()
        
        self.logger.info(f"Processing {len(xml_files)} XML chunks")

        manager = Manager()
        article_queue = manager.Queue(maxsize=1024)

        producer_pool = Pool(self.config.num_workers)
        
        producer_args = [(i, f, article_queue, self.config) for i, f in enumerate(xml_files)]
        
        producer_pool.starmap_async(xml_parser_worker, producer_args)
        producer_pool.close()
        
        self.logger.info(f"Started {self.config.num_workers} producers...")

        pbar = tqdm(
            desc="Processing articles",
            unit="article",
            initial=articles_already_processed,
            dynamic_ncols=True
        )
        
        article_buffer = []
        texts_buffer = []
        last_checkpoint = articles_already_processed
        
        try:
            while True:
                try:
                    article, model_input_text = article_queue.get(timeout=30)
                    
                    article_buffer.append(article)
                    texts_buffer.append(model_input_text)
                    
                    if len(article_buffer) >= self.config.batch_size:
                        self._process_batch(article_buffer, texts_buffer, pbar)
                        article_buffer, texts_buffer = [], []
                    
                    if (self.stats["articles_processed"] - last_checkpoint >= 
                        self.config.checkpoint_interval):
                        self._save_checkpoint()
                        last_checkpoint = self.stats["articles_processed"]

                except Empty:
                    if not producer_pool._cache:
                        self.logger.info("Queue is empty and producers are finished.")
                        break
                    else:
                        self.logger.info("Queue empty, waiting for producers...")
                        time.sleep(5)
            
            if article_buffer:
                self.logger.info(f"Processing final batch of {len(article_buffer)} articles.")
                self._process_batch(article_buffer, texts_buffer, pbar)
            
            pbar.close()
            producer_pool.join()
            self.logger.info("All producers joined.")

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user!")
            producer_pool.terminate()
            pbar.close()
            self._save_checkpoint() # Save on interrupt
            raise
        except Exception as e:
            self.logger.error(f"Fatal error in consumer: {e}", exc_info=True)
            producer_pool.terminate()
            pbar.close()
            self._save_checkpoint() # Save on error
            raise
        
        self.logger.info(f"✔️ Processed {self.stats['articles_processed']:,} articles")

    def _process_batch(self, articles: List[Article], texts: List[str], pbar: tqdm):
        """Process a batch of articles (Consumer side)"""
        
        filtered_articles = []
        filtered_texts = []
        skipped_count = 0
        
        for article, text in zip(articles, texts):
            if article.page_id not in self.processed_ids:
                filtered_articles.append(article)
                filtered_texts.append(text)
                self.processed_ids.add(article.page_id)
            else:
                skipped_count += 1
        
        if not filtered_articles:
            self.stats["articles_skipped"] += len(articles)
            return
        
        if skipped_count > 0 and skipped_count < len(articles):
            self.logger.info(f"Batch: {len(filtered_articles)} new, {skipped_count} duplicates")
        
        embeddings = self.generator.encode_batch(filtered_texts)
        
        if embeddings is not None:
            self.index_builder.add_batch(embeddings, filtered_articles)
            self.stats["articles_processed"] += len(filtered_articles)
            self.stats["batches_processed"] += 1
            pbar.update(len(filtered_articles))

    def _save_checkpoint(self):
            """Save progress checkpoint and prune old ones"""
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_name = f"checkpoint_{self.stats['articles_processed']:09d}"
            checkpoint_path = checkpoint_dir / checkpoint_name
            
            self.logger.info(f"Saving checkpoint: {checkpoint_name}...")
            
            try:
                # 1. Create directory
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # 2. Save FAISS index
                index_file = checkpoint_path / "index.faiss"
                faiss.write_index(self.index_builder.index, str(index_file))
                
                # 3. Save stats
                stats_file = checkpoint_path / "stats.json"
                with open(stats_file, 'w') as f:
                    json.dump(self.stats, f, indent=2)
                    
                # 4. Save metadata DB (by copying the live file)
                # Ensure DB changes are committed before copy
                self.index_builder.metadata_db.commit() 
                db_file = self.index_builder.db_path
                shutil.copy2(db_file, checkpoint_path / "metadata.db")
                
                self.logger.info(f"Checkpoint {checkpoint_name} saved successfully.")
                
                # 5. Prune old checkpoints
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
                if len(checkpoints) > 3: # Keep only the 3 most recent
                    checkpoints_to_delete = checkpoints[:-3]
                    self.logger.info(f"Pruning {len(checkpoints_to_delete)} old checkpoints...")
                    for old_checkpoint in checkpoints_to_delete:
                        self.logger.warning(f"Deleting old checkpoint: {old_checkpoint.name}")
                        shutil.rmtree(old_checkpoint)
                        
            except Exception as e:
                self.logger.error(f"Failed to save or prune checkpoint: {e}", exc_info=True)
                
            self.stats["checkpoints_saved"] += 1
    
    def run_optimization_phase(self):
        self.logger.info("=" * 80)
        self.logger.info("PHASE 3: INDEX OPTIMIZATION")
        self.logger.info("=" * 80)
        self.index_builder.optimize_index()
        self.logger.info("✔️ Index optimization complete")
    
    def run_final_validation_phase(self):
        self.logger.info("=" * 80)
        self.logger.info("PHASE 4: FINAL VALIDATION")
        self.logger.info("=" * 80)
        report = self.validator.run_full_validation()
        report["pipeline_stats"] = self.stats
        report["pipeline_stats"]["total_time_seconds"] = time.time() - self.stats["start_time"]
        report_path = Path(self.config.output_dir) / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"✔️ Validation report saved: {report_path}")
        return report
    
    def run(self):
        """Execute full pipeline"""
        try:
            checkpoint_loaded = self.load_latest_checkpoint()
        
            if checkpoint_loaded:
                self.logger.info("=" * 80)
                self.logger.info("RESUMING FROM CHECKPOINT")
                self.logger.info("=" * 80)
            else:
                self.index_builder._init_index()
            
            self.validator = ValidationSuite(self.config, self.index_builder, self.logger)
        
            # Phase 2: Full processing
            self.run_full_processing_phase()
            
            # Phase 3: Optimization
            self.run_optimization_phase()
            
            # Phase 4: Final validation
            report = self.run_final_validation_phase()
            
            # Save final index
            self.index_builder.save_final()
            
            self._print_summary(report)
            
            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False
    
    def _print_summary(self, report: dict):
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE!")
        self.logger.info("=" * 80)
        total_time = report["pipeline_stats"]["total_time_seconds"]
        articles = report["total_articles"]
        self.logger.info(f"Total articles: {articles:,}")
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"Throughput: {articles/(total_time/60):.0f} articles/min")
        self.logger.info(f"Search latency: {report['search_latency_ms']['mean']:.2f}ms")
        self.logger.info("")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"  - index.faiss")
        self.logger.info(f"  - metadata.db")
        self.logger.info(f"  - validation_report.json")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    config = Config()
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    if not any(Path(config.xml_chunk_dir).glob("*.xml*")):
        print(f"ERROR: No XML files found in {config.xml_chunk_dir}")
        print("Please put your .xml or .xml.bz2 file(s) in that directory.")
        sys.exit(1)
    
    pipeline = WikipediaEmbeddingsPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True) 
    main()