#!/usr/bin/env python3
"""
WIKIPEDIA EMBEDDINGS GENERATION SYSTEM
Production-grade pipeline for generating semantic embeddings of all English Wikipedia articles.

Architecture:
- Multi-process XML parsing with 8 workers (Producer-Consumer)
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
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """System configuration parameters"""
    
    # Paths
    # This is now the DIRECTORY of split XML files
    xml_chunk_dir: str = "/mnt/data-large/wikipedia/raw/xml_chunks/" 
    output_dir: str = "/mnt/data-large/wikipedia/embeddings"
    checkpoint_dir: str = "/mnt/data-large/wikipedia/checkpoints"
    
    # Model configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_seq_length: int = 512
    
    # Performance tuning
    num_workers: int = 8
    batch_size: int = 512
    checkpoint_interval: int = 100_000
    use_fp16: bool = True
    
    # Validation
    quick_validation_size: int = 10_000
    validation_queries: List[Tuple[str, List[str]]] = None
    
    # Filtering
    min_article_length: int = 200
    max_article_length: int = 100_000
    
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
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set format
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__) # Return the main logger

def get_worker_logger(worker_id: int):
    """
    Creates a dedicated logger for a worker process.
    This is CRITICAL for getting logs out of multiprocessing.
    """
    # Create a unique name for the worker logger
    logger = logging.getLogger(f"Worker-{worker_id}")
    
    # Check if handlers are already configured (to avoid duplicates)
    if logger.hasHandlers():
        return logger

    # Find the file handler from the root logger to log to the same file
    root_logger = logging.getLogger()
    file_handler = None
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler):
            file_handler = h
            break
            
    if file_handler:
        # If we found the main log file, add it to our worker logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO) # Make sure it logs info messages
    else:
        # Fallback if we can't find it (shouldn't happen if setup_logging was called)
        # This will log to console, which is better than nothing
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Could not find root FileHandler for Worker-{worker_id}. Logging to console.")

    # Prevent logs from propagating up to the root logger's console handler
    # (which would create duplicates)
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
    text: str # This will be the CLEANED text

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
    worker_id: int,  # ADDED: A unique ID for logging
    xml_file_path: str, 
    queue: Queue,
    config: Config
):
    """
    This is the producer function.
    It runs in a separate process.
    It takes ONE XML chunk file, parses it, cleans it,
    and puts the results into the shared queue.
    """
    # 1. SET UP LOGGING FOR THIS WORKER
    logger = get_worker_logger(worker_id)
    
    logger.info(f"Worker has started. Processing file: {xml_file_path}")
    
    try:
        namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
        page_count = 0
        skipped_count = 0
        put_count = 0
        
        with open(xml_file_path, 'rb') as f:
            logger.info("File opened. Starting iterparse...")
            
            # 2. CORRECT, EFFICIENT PARSING
            # We add the 'tag' argument to only parse <page> tags
            context = ET.iterparse(f, events=('end',), tag=f"{namespace}page")
            
            for event, elem in context:
                page_count += 1
                if page_count % 1000 == 0:
                    logger.info(f"Parsed {page_count} pages so far... (Skipped: {skipped_count}, Queued: {put_count})")
                    
                try:
                    # --- Extract title ---
                    title_elem = elem.find(f"{namespace}title")
                    if title_elem is None or not title_elem.text:
                        logger.warning("SKIPPING: Page has no title")
                        skipped_count += 1
                        continue
                    title = title_elem.text.strip().replace(" ", "_")
                    # logger.info(f"Found title: {title}") # <-- This is TOO noisy, uncomment if desperate

                    # --- Extract namespace ---
                    ns_elem = elem.find(f"{namespace}ns")
                    
                    # 3. CORRECT VARIABLE NAMING
                    # We use 'ns_value' so it doesn't overwrite the 'namespace' string
                    ns_value = int(ns_elem.text) if ns_elem is not None else 0
                    
                    # --- Extract page ID ---
                    id_elem = elem.find(f"{namespace}id")
                    if id_elem is None:
                        logger.warning(f"SKIPPING: Page '{title}' has no ID")
                        skipped_count += 1
                        continue
                    page_id = int(id_elem.text)
                        
                    # --- Check for redirect ---
                    if elem.find(f"{namespace}redirect") is not None:
                        # logger.info(f"SKIPPING: Page '{title}' is a redirect")
                        skipped_count += 1
                        continue
                        
                    # --- Extract text content ---
                    revision = elem.find(f"{namespace}revision")
                    if revision is None:
                        logger.warning(f"SKIPPING: Page '{title}' has no revision")
                        skipped_count += 1
                        continue
                        
                    text_elem = revision.find(f"{namespace}text")
                    if text_elem is None or not text_elem.text:
                        logger.warning(f"SKIPPING: Page '{title}' has no text")
                        skipped_count += 1
                        continue
                        
                    # --- Filtering & Cleaning ---
                    if ns_value != 0:
                        # logger.info(f"SKIPPING: Page '{title}' is not in main namespace (ns={ns_value})")
                        skipped_count += 1
                        continue
                    if title.startswith("List_of_"):
                        # logger.info(f"SKIPPING: Page '{title}' is a List page")
                        skipped_count += 1
                        continue
                    if "(disambiguation)" in title.lower():
                        # logger.info(f"SKIPPING: Page '{title}' is a disambiguation page")
                        skipped_count += 1
                        continue

                    text = _clean_wikitext(text_elem.text)
                    text_len = len(text)
                    
                    if not (config.min_article_length < text_len < config.max_article_length):
                        # logger.info(f"SKIPPING: Page '{title}' wrong length ({text_len} chars)")
                        skipped_count += 1
                        continue
                        
                    # --- Send to Consumer ---
                    article = Article(
                        page_id=page_id,
                        title=title,
                        namespace=ns_value,
                        text=text # Store clean text
                    )
                    model_input_text = f"{title}. {text[:2000]}"
                        
                    # logger.info(f"SUCCESS: Adding '{title}' to queue")
                    queue.put((article, model_input_text))
                    put_count += 1
                    
                except Exception as e:
                    # Log and continue if a single page fails
                    logger.error(f"Failed to parse a page in {xml_file_path}: {e}")
                
                finally:
                    # Critical: clear element to free memory
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

    except Exception as e:
        logger.error(f"WORKER FAILED FATALLY on {xml_file_path}: {e}", exc_info=True)
    finally:
        logger.info(f"Worker finished file. Total pages: {page_count}. Skipped: {skipped_count}. Queued: {put_count}.")



# ============================================================================
# EMBEDDING GENERATION - GPU OPTIMIZED
# ============================================================================

class EmbeddingGenerator:
    """GPU-optimized embedding generation with batching"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = self._setup_device()
        self.model = self._load_model()
        
    def _setup_device(self) -> str:
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
        self.metadata_db = None
        self._init_index()
        self._init_metadata_db()
        
    def _init_index(self):
        """Initialize FAISS index"""
        self.index = faiss.IndexFlatIP(self.config.embedding_dim)
        self.index = faiss.IndexIDMap(self.index)
        self.logger.info(f"Initialized FAISS index (dim={self.config.embedding_dim})")
    
    def _init_metadata_db(self):
        """Initialize SQLite database for metadata"""
        db_path = Path(self.config.output_dir) / "wikipedia_metadata.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metadata_db = sqlite3.connect(str(db_path))
        cursor = self.metadata_db.cursor()
        
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
        
        self.metadata_db.commit()
        self.logger.info("Initialized metadata database")
    
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
            """INSERT OR REPLACE INTO articles 
               (idx, article_id, title, namespace, char_count, embedding_timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (i, a.page_id, a.title, a.namespace, len(a.text), timestamp)
                for i, a in enumerate(articles, start=self.index.ntotal - len(articles))
            ]
        )
        
        self.metadata_db.commit()
    
    def save_checkpoint(self, checkpoint_path: Path, stats: dict):
        """Save index and metadata checkpoint"""
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        index_path = checkpoint_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        db_path = Path(self.config.output_dir) / "wikipedia_metadata.db"
        checkpoint_db = checkpoint_path / "metadata.db"
        
        source_conn = sqlite3.connect(str(db_path))
        dest_conn = sqlite3.connect(str(checkpoint_db))
        source_conn.backup(dest_conn)
        source_conn.close()
        dest_conn.close()
        
        stats_path = checkpoint_path / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def optimize_index(self):
        """Convert to optimized IVF+PQ index for production"""
        self.logger.info("Optimizing index with IVF+PQ...")
        
        n_vectors = self.index.ntotal
        
        if n_vectors < 1000:
            self.logger.warning("Too few vectors for optimization, keeping flat index")
            return
        
        n_clusters = int(np.sqrt(n_vectors))
        n_clusters = min(n_clusters, n_vectors // 39)
        
        self.logger.info(f"Training IVF with {n_clusters} clusters...")
        
        vectors = np.zeros((n_vectors, self.config.embedding_dim), dtype=np.float32)
        for i in range(n_vectors):
            vectors[i] = self.index.reconstruct(i)
        
        quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
        index_ivf = faiss.IndexIVFPQ(
            quantizer,
            self.config.embedding_dim,
            n_clusters,
            64,  # Number of sub-quantizers
            8    # Bits per sub-quantizer
        )
        
        index_ivf.train(vectors)
        index_ivf.add(vectors)
        index_ivf.nprobe = 32
        
        self.logger.info(f"Index optimized: {n_vectors} vectors, {n_clusters} clusters")
        self.index = index_ivf
    
    def save_final(self):
        """Save final optimized index"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        index_path = output_path / "wikipedia_embeddings.faiss"
        faiss.write_index(self.index, str(index_path))
        
        self.metadata_db.close()
        
        self.logger.info(f"Final index saved: {index_path}")
        self.logger.info(f"Metadata DB: {output_path / 'wikipedia_metadata.db'}")

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
    
    def run_quick_validation(self, embeddings: np.ndarray, articles: List[Article]):
        """Quick statistical validation of embeddings"""
        self.logger.info("Running quick validation...")
        
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("Invalid embeddings detected (NaN or Inf)")
        
        mean = np.mean(embeddings)
        std = np.std(embeddings)
        magnitude = np.mean(np.linalg.norm(embeddings, axis=1))
        
        self.logger.info(f"Embedding stats: mean={mean:.4f}, std={std:.4f}, magnitude={magnitude:.4f}")
        
        if not (0.95 < magnitude < 1.05):
            self.logger.warning(f"Unexpected embedding magnitude: {magnitude}")
        
        return True
    
    def run_semantic_validation(self, model: SentenceTransformer):
        """Validate semantic search quality"""
        self.logger.info("Running semantic validation...")
        
        results = []
        
        for query_text, expected_titles in self.config.validation_queries:
            query_emb = model.encode([query_text], normalize_embeddings=True)
            
            if not isinstance(query_emb, np.ndarray):
                query_emb = np.array(query_emb)
            query_emb = query_emb.astype(np.float32)
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            
            k = 10
            distances, indices = self.index.search(query_emb, k)
            
            cursor = self.metadata_db.cursor()
            result_titles = []
            
            for idx in indices[0]:
                cursor.execute("SELECT title FROM articles WHERE idx = ?", (int(idx),))
                row = cursor.fetchone()
                if row:
                    result_titles.append(row[0])
            
            hits = sum(1 for title in expected_titles if title in result_titles)
            recall = hits / len(expected_titles)
            
            self.logger.info(f"Query '{query_text}': {hits}/{len(expected_titles)} recall={recall:.2f}")
            results.append(recall)
        
        avg_recall = np.mean(results)
        self.logger.info(f"Average recall@10: {avg_recall:.3f}")
        
        if avg_recall < 0.5:
            self.logger.warning("Low recall detected - search quality may be poor")
s
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
        self.logger.info("WIKIPEDIA EMBEDDINGS GENERATION PIPELINE")
        self.logger.info("=" * 80)
        
        # NO parser here!
        self.generator = EmbeddingGenerator(config, self.logger)
        self.index_builder = FAISSIndexBuilder(config, self.logger)
        self.validator = ValidationSuite(config, self.index_builder, self.logger)
        
        self.stats = {
            "start_time": time.time(),
            "articles_processed": 0,
            "articles_skipped": 0,
            "batches_processed": 0,
            "checkpoints_saved": 0
        }
    
    def run_full_processing_phase(self):
        """Phase 2: Process all articles using Producer-Consumer"""
        self.logger.info("=" * 80)
        self.logger.info("PHASE 2: FULL PROCESSING (Producer-Consumer)")
        self.logger.info("=" * 80)
        
        # Get list of all XML chunk files
        xml_files = sorted(glob(f"{self.config.xml_chunk_dir}/*.xml*")) # Use sorted for deterministic order
        if not xml_files:
            self.logger.error(f"No XML chunks found in {self.config.xml_chunk_dir}")
            raise FileNotFoundError(f"No XML chunks found in {self.config.xml_chunk_dir}. Run chunk.py first.")
            
        self.logger.info(f"Found {len(xml_files)} XML chunks to process.")

        # Create a shared queue
        manager = Manager()
        article_queue = manager.Queue(maxsize=1024) # Max 1024 items in memory

        # Start producer pool
        producer_pool = Pool(self.config.num_workers)
        
        # MODIFIED: Pass a worker ID (i) to each worker for logging
        producer_args = [(i, f, article_queue, self.config) for i, f in enumerate(xml_files)]
        
        producer_pool.starmap_async(xml_parser_worker, producer_args)
        producer_pool.close() # No more tasks will be added
        
        self.logger.info(f"Started {self.config.num_workers} producers...")

        # --- This (the main thread) is now the CONSUMER ---
        
        pbar = tqdm(
            desc="Processing articles",
            unit="article",
            total=6_900_000, # Approx
            dynamic_ncols=True
        )
        
        article_buffer = []
        texts_buffer = []
        last_checkpoint = 0
        
        try:
            while True:
                try:
                    # Get data from the queue
                    article, model_input_text = article_queue.get(timeout=30)
                    
                    article_buffer.append(article)
                    texts_buffer.append(model_input_text)
                    
                    # Process batch when buffer is full
                    if len(article_buffer) >= self.config.batch_size:
                        self._process_batch(article_buffer, texts_buffer, pbar)
                        article_buffer, texts_buffer = [], []
                    
                    # Checkpoint if needed
                    if (self.stats["articles_processed"] - last_checkpoint >= 
                        self.config.checkpoint_interval):
                        self._save_checkpoint()
                        last_checkpoint = self.stats["articles_processed"]

                except Empty:
                    # If queue is empty, check if producers are done
                    # This is an internal check, but it's one way to see if the pool is done
                    if not producer_pool._cache: 
                        self.logger.info("Queue is empty and producers are finished.")
                        break
                    else:
                        self.logger.info("Queue empty, waiting for producers...")
                        time.sleep(5) # Wait a bit before checking queue again
            
            # Process any remaining articles
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
            self._save_checkpoint()
            raise
        except Exception as e:
            self.logger.error(f"Fatal error in consumer: {e}", exc_info=True)
            producer_pool.terminate()
            pbar.close()
            self._save_checkpoint()
            raise
        
        self.logger.info(f"✔ Processed {self.stats['articles_processed']:,} articles")

a
    def _process_batch(self, articles: List[Article], texts: List[str], pbar: tqdm):
        """Process a batch of articles (Consumer side)"""
        
        embeddings = self.generator.encode_batch(texts)
        
        if embeddings is not None:
            self.index_builder.add_batch(embeddings, articles)
            self.stats["articles_processed"] += len(articles)
            self.stats["batches_processed"] += 1
            pbar.update(len(articles))

t
    def _save_checkpoint(self):
        """Save progress checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_name = f"checkpoint_{self.stats['articles_processed']:08d}"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        self.index_builder.save_checkpoint(checkpoint_path, self.stats)
        self.stats["checkpoints_saved"] += 1
    
    def run_optimization_phase(self):
        # (Unchanged)
        self.logger.info("=" * 80)
        self.logger.info("PHASE 3: INDEX OPTIMIZATION")
        self.logger.info("=" * 80)
        self.index_builder.optimize_index()
        self.logger.info("✔ Index optimization complete")
    
    def run_final_validation_phase(self):
        # (Unchanged)
        self.logger.info("=" * 80)
        self.logger.info("PHASE 4: FINAL VALIDATION")
        self.logger.info("=" * 80)
        report = self.validator.run_full_validation()
        report["pipeline_stats"] = self.stats
        report["pipeline_stats"]["total_time_seconds"] = time.time() - self.stats["start_time"]
        report_path = Path(self.config.output_dir) / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"✔ Validation report saved: {report_path}")
        return report
    
    def run(self):
        """Execute full pipeline"""
        try:
            # Phase 2: Full processing
            self.run_full_processing_phase()
            
            # Phase 3: Optimization
            self.run_optimization_phase()
            
            # Phase 4: Final validation
CH
            report = self.run_final_validation_phase()
            
            # Save final index
            self.index_builder.save_final()
            
            self._print_summary(report)
            
            return True
           s
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False
    
    def _print_summary(self, report: dict):
        # (Unchanged)
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE!")
        self.logger.info("=" * 80)
        total_time = report["pipeline_stats"]["total_time_seconds"]
s        articles = report["total_articles"]
        self.logger.info(f"Total articles: {articles:,}")
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"Throughput: {articles/(total_time/60):.0f} articles/min")
        self.logger.info(f"Search latency: {report['search_latency_ms']['mean']:.2f}ms")
        self.logger.info("")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"  - wikipedia_embeddings.faiss")
        self.logger.info(f"  - wikipedia_metadata.db")
        self.logger.info(f"  - validation_report.json")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
DELETEME
    """Main entry point"""
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    config = Config()
    
    if not Path(config.xml_chunk_dir).exists():
        print(f"ERROR: Wikipedia chunk dir not found: {config.xml_chunk_dir}")
        print("Please ensure the chunk.py script has been run")
        sys.exit(1)
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    pipeline = WikipediaEmbeddingsPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()