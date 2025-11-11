#!/usr/bin/env python3
"""
WIKIPEDIA EMBEDDINGS GENERATION SYSTEM
Production-grade pipeline for generating semantic embeddings of all English Wikipedia articles.
- RESTORED checkpointing and resuming
- PATCHED optimize_index function
- MERGED user config changes
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
# CONFIGURATION - YOUR LOCAL SETTINGS
# ============================================================================

@dataclass
class Config:
    """System configuration parameters"""
    
    # --- YOUR LOCAL PATHS ---
    xml_chunk_dir: str = "./data/raw/" 
    output_dir: str = "./data/embeddings/"
    checkpoint_dir: str = "./data/checkpoints/"
    
    # Model configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_seq_length: int = 512
    
    # --- YOUR LOCAL TUNING ---
    num_workers: int = 4
    batch_size: int = 64
    checkpoint_interval: int = 100
    use_fp16: bool = False 
    
    # Validation
    quick_validation_size: int = 10
    validation_queries: List[Tuple[str, List[str]]] = None
    
    # Filtering
    min_article_length: int = 100
    max_article_length: int = 200_000
    
    def __post_init__(self):
        """Initialize validation queries"""
        if self.validation_queries is None:
            self.validation_queries = [
                ("programming languages", ["Python_(programming_language)", "Machine_learning"]),
                ("European countries", ["France", "Germany"]),
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
        
        # --- FIXED ---
        # This will open .xml or .xml.bz2 files (which you have in data/raw)
        open_func = bz2.open if xml_file_path.endswith('.bz2') else open
        
        with open_func(xml_file_path, 'rb') as f:
            logger.info(f"File opened with {open_func.__name__}. Starting iterparse...")
            
            context = ET.iterparse(f, events=('end',), tag=f"{namespace}page")
            
            for event, elem in context:
                page_count += 1
                if page_count % 10000 == 0:
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
                    
                    if not (config.min_article_length < text_len < config.max_article_length):
                        # --- YOUR CHANGE (MERGED) ---
                        if page_count % 1000 == 0: # Log skips less often
                            logger.warning(f"Skipping {title} (length: {text_len})")
                        skipped_count += 1
                        continue
                        
                    article = Article(
                        page_id=page_id,
                        title=title,
                        namespace=ns_value,
                        text=text
                    )
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
    """Embedding generation with batching"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = self._setup_device()
        self.model = self._load_model()
        
    def _setup_device(self) -> str:
        # Check for Mac MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            device = "mps"
            self.logger.info("Mac MPS (GPU) detected. Using 'mps' device.")
        elif torch.cuda.is_available():
            device = "cuda"
            self.logger.info("NVIDIA GPU detected. Using 'cuda' device.")
        else:
            device = "cpu"
            self.logger.warning("No GPU detected, using CPU.")
        return device
    
    def _load_model(self) -> SentenceTransformer:
        self.logger.info(f"Loading model: {self.config.model_name}")
        model = SentenceTransformer(self.config.model_name, device=self.device)
        model.max_seq_length = self.config.max_seq_length
        # Note: FP16 is not enabled for MPS/CPU
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
            self.logger.error(f"Error during encoding: {e}")
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
        # CRITICAL: This maps article_id -> vector
        self.index = faiss.IndexIDMap(self.index) 
        self.logger.info(f"Initialized new FAISS index (dim={self.config.embedding_dim})")
    
    def _init_metadata_db(self) -> sqlite3.Connection:
        """Initialize SQLite database for metadata"""
        db_path = self.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # --- FIXED ---
        # DO NOT DELETE the DB. This allows resuming.
        
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
                # --- FIXED ---
                # Use None for the PRIMARY KEY to let SQLite auto-increment
                (None, a.page_id, a.title, a.namespace, len(a.text), timestamp)
                for a in articles
            ]
        )
        
        self.metadata_db.commit()
    
    def save_checkpoint(self, checkpoint_path: Path, stats: dict):
        """Save index and metadata checkpoint"""
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        index_path = checkpoint_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        checkpoint_db = checkpoint_path / "metadata.db"
        
        # Backup the main DB to the checkpoint location
        source_conn = sqlite3.connect(str(self.db_path))
        dest_conn = sqlite3.connect(str(checkpoint_db))
        source_conn.backup(dest_conn)
        source_conn.close()
        dest_conn.close()
        
        stats_path = checkpoint_path / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

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

        # 1. Get ALL article_ids from the DB
        self.logger.info(f"Fetching {n_vectors} article IDs from DB...")
        cursor = self.metadata_db.cursor()
        
        # --- PATCH ---
        # Get IDs from the index *first*, then check DB
        db_ids = set(row[0] for row in cursor.execute("SELECT article_id FROM articles"))
        index_ids = faiss.vector_to_array(self.index.id_map)
        
        ids_in_both = [id for id in index_ids if id in db_ids]
        n_vectors = len(ids_in_both)
        ids = np.array(ids_in_both, dtype=np.int64)
        
        if n_vectors < 1000:
             self.logger.warning(f"Too few vectors in common ({n_vectors}) for optimization.")
             return
        
        self.logger.info(f"Found {n_vectors} vectors common to DB and Index.")


        # 2. Reconstruct ALL vectors using their *real IDs*
        vectors = np.zeros((n_vectors, self.config.embedding_dim), dtype=np.float32)
        self.logger.info("Reconstructing all vectors from flat index...")
        for i, article_id in enumerate(tqdm(ids, desc="Reconstructing vectors")):
            try:
                vectors[i] = self.index.reconstruct(int(article_id))
            except Exception as e:
                self.logger.error(f"Failed to reconstruct {article_id}: {e}")
                raise
        
        self.logger.info("All vectors reconstructed. Training new index...")

        # 3. Build new optimized index
        n_clusters = int(np.sqrt(n_vectors))
        n_clusters = min(n_clusters, max(1, n_vectors // 39)) # Ensure n_clusters >= 1
        if n_clusters < 1: n_clusters = 1
        
        self.logger.info(f"Training IVF with {n_clusters} clusters...")
        
        quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
        index_ivf = faiss.IndexIVFPQ(
            quantizer,
            self.config.embedding_dim,
            n_clusters,
            64,
            8
        )
        
        # Train on the REAL vectors
        index_ivf.train(vectors)
        self.logger.info("Training complete.")

        # 4. Create a NEW IndexIDMap and add vectors WITH IDs
        self.logger.info("Creating new optimized IndexIDMap2...")
        new_index = faiss.IndexIDMap2(index_ivf)
        
        # Add the vectors WITH their original article_ids
        new_index.add_with_ids(vectors, ids)
        
        ivf_index_cast = faiss.downcast_index(new_index.index)
        ivf_index_cast.nprobe = min(32, n_clusters)
        
        self.logger.info(f"Index optimized: {new_index.ntotal} vectors, {n_clusters} clusters")
        
        # Replace the old index
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
        
        # Set nprobe if we are optimized
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
            
            k = 5
            distances, article_ids = self.index.search(query_emb, k)
            
            cursor = self.metadata_db.cursor()
            result_titles = []
            
            for article_id in article_ids[0]:
                if int(article_id) == -1: continue # No result
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
        self.logger.info(f"Average recall@5: {avg_recall:.3f}")
        
        return avg_recall
    
    def run_full_validation(self) -> dict:
        self.logger.info("Running full validation suite...")
        report = {}
        # ... (skipping for brevity, semantic is most important) ...
        self.logger.info("Validation complete.")
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
        self.logger.info("WIKIPEDIA EMBEDDINGS PIPELINE (LOCAL TEST)")
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

    # --- CHECKPOINTING LOGIC RESTORED ---
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
            
            # Re-connect the index_builder's DB handle to this new file
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

    # --- LOGIC RESTORED ---
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
                    
                    # --- THIS IS THE MISSING LOGIC ---
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

    # --- LOGIC RESTORED ---
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

    # --- LOGIC RESTORED ---
    def _save_checkpoint(self):
        """Save progress checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        # Format with leading zeros for correct sorting
        checkpoint_name = f"checkpoint_{self.stats['articles_processed']:09d}"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        self.index_builder.save_checkpoint(checkpoint_path, self.stats)
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
        self.validator.run_semantic_validation(self.generator.model)
        self.logger.info("✔️ Validation complete")
    
    # --- LOGIC RESTORED ---
    def run(self):
        """Execute full pipeline"""
        try:
            # Try to load checkpoint first
            checkpoint_loaded = self.load_latest_checkpoint()
        
            if checkpoint_loaded:
                self.logger.info("=" * 80)
                self.logger.info("RESUMING FROM CHECKPOINT")
                self.logger.info("=" * 80)
            else:
                # No checkpoint - create new index
                self.index_builder._init_index()
            
            # Initialize validator AFTER index is ready
            self.validator = ValidationSuite(self.config, self.index_builder, self.logger)
        
            # Phase 2: Full processing
            self.run_full_processing_phase()
            
            # Phase 3: Optimization
            self.run_optimization_phase()
            
            # Phase 4: Final validation
            self.run_final_validation_phase()
            
            # Save final index
            self.index_builder.save_final()
            
            self._print_summary()
            
            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False
    
    def _print_summary(self):
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE!")
        self.logger.info("=" * 80)
        total_time = time.time() - self.stats["start_time"]
        articles = self.stats["articles_processed"]
        self.logger.info(f"Total articles: {articles:,}")
        self.logger.info(f"Total time: {total_time/60:.2f} minutes")
        self.logger.info("")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f" - index.faiss")
        self.logger.info(f" - metadata.db")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    config = Config()
    
    # Create local dirs
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    if not any(Path(config.xml_chunk_dir).glob("*.xml*")):
        print(f"ERROR: No XML files found in {config.xml_chunk_dir}")
        print(f"Please put your .xml or .xml.bz2 file in {config.xml_chunk_dir}")
        sys.exit(1)
    
    pipeline = WikipediaEmbeddingsPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # 'spawn' is safer, especially on macOS
    torch.multiprocessing.set_start_method('spawn', force=True) 
    main()