#!/usr/bin/env python3
"""
WIKIPEDIA CHECKPOINT VALIDATION SCRIPT

This script is designed to be run on a live, in-progress system.
It automatically finds the *latest* checkpoint directory inside
CHECKPOINT_DIR, loads that checkpoint's index and database,
and runs a series of semantic queries to validate its quality.
"""

import sys
import sqlite3
import time
from pathlib import Path
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Base directory where all checkpoints are saved
CHECKPOINT_DIR = Path("./data/checkpoints")
TOP_K = 5 # How many results to fetch for each query

# --- VALIDATION QUERIES ---
# (Query Text, [List of expected article titles])
# Add more queries as your script processes more of the alphabet
QUERIES = [
    ("Tibetan Buddhism", ["Dalai_Lama"]),
    ("natural catastrophe", ["Disaster"]),
    ("Italian soccer goalie", ["Dino_Zoff"]),
    ("part of a brain cell", ["Dendrite"]),
    ("regional language differences", ["Dialect"]),
    ("heart medication", ["Digitalis"]),
    ("comic strip about office life", ["Dilbert"]),
    ("Greek alphabet letter", ["Digamma"]),
    ("money for wrongdoing", ["Damages"]),
]

def main():
    print("=" * 80)
    print("WIKIPEDIA CHECKPOINT VALIDATION SCRIPT")
    print("=" * 80)

    # --- NEW LOGIC: Find the latest checkpoint ---
    print(f"Scanning for checkpoints in: {CHECKPOINT_DIR}")
    try:
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*"))
        if not checkpoints:
            print(f"ERROR: No checkpoints found in {CHECKPOINT_DIR}")
            print("Did you run the generate script yet?")
            sys.exit(1)
        
        latest_checkpoint_dir = checkpoints[-1]
        print(f"Found latest checkpoint: {latest_checkpoint_dir.name}\n")
    
    except Exception as e:
        print(f"ERROR: Could not scan for checkpoints: {e}")
        sys.exit(1)

    # --- Set paths dynamically based on the latest checkpoint ---
    INDEX_PATH = latest_checkpoint_dir / "index.faiss"
    DB_PATH = latest_checkpoint_dir / "metadata.db"

    if not INDEX_PATH.exists():
        print(f"ERROR: index.faiss not found in {latest_checkpoint_dir}")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"ERROR: metadata.db not found in {latest_checkpoint_dir}")
        sys.exit(1)

    # 1. Load Model
    print(f"Loading model: {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 2. Load FAISS Index
    print(f"Loading index: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    
    # Set nprobe if the index is IVF (optimized)
    try:
        ivf_index = faiss.downcast_index(index)
        ivf_index.nprobe = 16
        print(f"Index is IVF. Set nprobe = {ivf_index.nprobe}")
    except:
         try:
            ivf_index = faiss.downcast_index(index.index)
            ivf_index.nprobe = 16
            print(f"Index is IndexIDMap(IVF). Set nprobe = {ivf_index.nprobe}")
         except:
            print("Index is Flat. nprobe not applicable.")

    # 3. Connect to DB
    print(f"Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    print("\n--- RUNNING VALIDATION QUERIES ---\n")
    
    total_queries = len(QUERIES)
    hits = 0

    for query_text, expected_titles in QUERIES:
        print(f"Query: '{query_text}'")
        print(f"  > Expected: {expected_titles[0]}")
        
        # 4. Encode Query
        query_emb = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # 5. Search FAISS
        distances, article_ids = index.search(query_emb, TOP_K)
        
        article_ids_list = [int(idx) for idx in article_ids[0] if idx >= 0]
        if not article_ids_list:
            print("    <No valid results from FAISS>")
            print("  > ❌ FAIL\n")
            continue
            
        placeholders = ",".join("?" * len(article_ids_list))
        
        # 6. Fetch Metadata
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT title, article_id FROM articles WHERE article_id IN ({placeholders})",
            article_ids_list
        )
        
        id_to_title = {article_id: title for title, article_id in cursor.fetchall()}
        
        # 7. Print Results & Score
        found_titles = []
        for i, article_id in enumerate(article_ids_list):
            if article_id in id_to_title:
                title = id_to_title[article_id]
                score = distances[0][i]
                found_titles.append(title)
                print(f"    {i+1}. {title} (Score: {score:.4f})")
        
        if not found_titles:
            print("    <FAISS returned IDs, but none were in the DB>")

        # Check for hit
        if any(expected in found_titles for expected in expected_titles):
            hits += 1
            print("  > ✅ PASS\n")
        else:
            print("  > ❌ FAIL\n")
            
    conn.close()
    
    # 8. Final Report
    print("--- VALIDATION COMPLETE ---")
    print(f"Score: {hits} / {total_queries} correct")
    print("=" * 80)

if __name__ == "__main__":
    main()