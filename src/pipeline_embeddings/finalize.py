#!/usr/bin/env python3
"""
WIKIPEDIA EMBEDDINGS FINALIZER

This script loads the latest *un-optimized* checkpoint, runs the
optimization and compression (IVFPQ), saves the final production-ready
artifacts, and runs a final validation.

Run this script if the main 'generate_wiki_embeddings.py' script
crashed or was stopped *after* Phase 2 (parsing) but *before*
it could save the final optimized index.
"""

import os
import sys
import json
import time
import sqlite3
import logging
import re
import shutil
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

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
    
    # --- PRODUCTION PATHS ---
    # !! These must match your main script !!
    output_dir: str = "/mnt/data/wikipedia/embeddings/"
    checkpoint_dir: str = "/mnt/data/wikipedia/checkpoints/"
    
    # Model configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Validation queries (copied from main script)
    validation_queries: List[Tuple[str, List[str]]] = None
    
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
    log_file = log_dir / f"finalize_{timestamp}.log"
    
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

# ============================================================================
# EMBEDDING GENERATOR (Needed for Validation)
# ============================================================================

class EmbeddingGenerator:
    """Loads model for validation queries"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = self._setup_device()
        self.model = self._load_model()
        
    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU detected: {gpu_name}")
        else:
            device = "cpu"
            self.logger.warning("No GPU detected, using CPU")
        return device
    
    def _load_model(self) -> SentenceTransformer:
        self.logger.info(f"Loading model: {self.config.model_name}")
        model = SentenceTransformer(self.config.model_name, device=self.device)
        if self.device == "cuda":
            model = model.half() # Use FP16 for validation query
        return model

# ============================================================================
# FAISS INDEX BUILDER (with Patched Optimize Function)
# ============================================================================

class FAISSIndexBuilder:
    """Loads, optimizes, and saves the final FAISS index"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.index = None # Will be loaded from checkpoint
        self.db_path = Path(self.config.output_dir) / "metadata.db"
        self.metadata_db = self._init_metadata_db()
        
    def _init_metadata_db(self) -> sqlite3.Connection:
        """Initialize SQLite database for metadata"""
        db_path = self.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_db = sqlite3.connect(str(db_path))
        cursor = metadata_db.cursor()
        
        # Create table just in case, but it should exist
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
        metadata_db.commit()
        self.logger.info(f"Initialized metadata database at {db_path}")
        return metadata_db
    
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
                # Reconstruct using the *internal* ID (i)
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
        
        # Use half of available RAM for training if possible
        # res = faiss.StandardGpuResources()
        # index_ivf.train_gpu(res, vectors)
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
        
        self.metadata_db.commit()
        self.metadata_db.close()
        
        self.logger.info(f"Final optimized index saved: {index_path}")
        self.logger.info(f"Final metadata DB saved: {self.db_path}")

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
    
    def run_full_validation(self, model: SentenceTransformer) -> dict:
        """Comprehensive post-processing validation"""
        self.logger.info("Running full validation suite...")
        
        report = {
            "total_articles": self.index.ntotal,
            "embedding_dimension": self.config.embedding_dim,
            "index_type": type(self.index.index).__name__, # Get optimized type
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
        
        # Run semantic validation
        self.run_semantic_validation(model)

        # Run latency test
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

    def run_semantic_validation(self, model: SentenceTransformer):
        """Validate semantic search quality"""
        self.logger.info("Running semantic validation...")
        results = []
        
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_latest_checkpoint(config: Config, logger: logging.Logger) -> Tuple[str, dict]:
    """Finds the latest checkpoint, copies its DB, and returns index path & stats"""
    checkpoint_dir = Path(config.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None, None
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_dir}")
        return None, None
    
    latest_checkpoint = checkpoints[-1]
    logger.info(f"Found latest checkpoint: {latest_checkpoint.name}")
    
    index_file = latest_checkpoint / "index.faiss"
    checkpoint_db = latest_checkpoint / "metadata.db"
    stats_file = latest_checkpoint / "stats.json"
    
    if not index_file.exists() or not checkpoint_db.exists():
        logger.error(f"Checkpoint is incomplete. Missing index or db in {latest_checkpoint}")
        return None, None
    
    # This is the critical step: copy the checkpoint DB to the final output path
    # The FAISSIndexBuilder will then open this copied file
    working_db = Path(config.output_dir) / "metadata.db"
    logger.info(f"Copying metadata DB from checkpoint to {working_db}")
    shutil.copy2(checkpoint_db, working_db)
    
    stats = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            stats["start_time"] = time.time() # Reset timer
    
    logger.info(f"Loaded checkpoint with {stats.get('articles_processed', 'N/A')} articles")
    return str(index_file), stats

def _print_summary(report: dict, stats: dict, logger: logging.Logger, config: Config):
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)
    total_time = time.time() - stats["start_time"]
    articles = report["total_articles"]
    logger.info(f"Total articles: {articles:,}")
    logger.info(f"Finalization time: {total_time/60:.2f} minutes")
    logger.info(f"Optimized Index: {report['index_type']}")
    logger.info(f"Search latency (mean): {report['search_latency_ms']['mean']:.2f}ms")
    logger.info("")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"  - index.faiss")
    logger.info(f"  - metadata.db")
    logger.info(f"  - validation_report.json")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    config = Config()
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("WIKIPEDIA EMBEDDINGS FINALIZER (PRODUCTION)")
    logger.info("=" * 80)
    
    try:
        # --- PHASE 1: LOAD LATEST CHECKPOINT ---
        logger.info("PHASE 1: Loading latest checkpoint...")
        index_path, stats = load_latest_checkpoint(config, logger)
        if index_path is None:
            sys.exit(1)
        
        index_builder = FAISSIndexBuilder(config, logger)
        logger.info(f"Loading FAISS index from {index_path}")
        index_builder.index = faiss.read_index(index_path)
        logger.info(f"Successfully loaded index with {index_builder.index.ntotal} vectors")

        # --- PHASE 2: OPTIMIZATION ---
        logger.info("PHASE 2: Starting index optimization...")
        start_time = time.time()
        index_builder.optimize_index()
        logger.info(f"Optimization complete in {(time.time() - start_time)/60:.2f} minutes")

        # --- PHASE 3: VALIDATION ---
        logger.info("PHASE 3: Running final validation...")
        generator = EmbeddingGenerator(config, logger)
        validator = ValidationSuite(config, index_builder, logger)
        report = validator.run_full_validation(generator.model)
        
        report["pipeline_stats"] = stats
        report_path = Path(config.output_dir) / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✔️ Validation report saved: {report_path}")

        # --- PHASE 4: SAVE FINAL ARTIFACTS ---
        logger.info("PHASE 4: Saving final production artifacts...")
        index_builder.save_final()
        
        _print_summary(report, stats, logger, config)
        
    except Exception as e:
        logger.error(f"Finalization pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()