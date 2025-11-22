#!/usr/bin/env python3
"""
Wikipedia Semantic Search
Searches the generated FAISS index and returns the top 5 most similar articles.
"""

import sys
import time
import sqlite3
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os

# --- 1. CRITICAL FIX for macOS (Part A) ---
# Disables tokenizer parallelism to prevent segfaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 2. CRITICAL FIX for macOS (Part B) ---
# Forces 'spawn' start method for multiprocessing to prevent
# segfaults and "leaked semaphore" warnings from PyTorch/fork.
# This MUST be in the 'if __name__ == "__main__":' block.


# --- Configuration ---
# local
DB_PATH = "data/embeddings/metadata.db"
INDEX_PATH = "data/embeddings/index.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5 # Number of results to return
# ---------------------

def search(query_string: str):
    """
    Loads the models and DBs, encodes the query, and performs the search.
    """
    
    db_conn = None
    try:
        # --- 1. Get Query from User ---
        if not query_string:
            print("Usage: ./search.py \"<your query>\"", file=sys.stderr)
            sys.exit(1)
            
        print(f"Searching for: \"{query_string}\"")
        print("-" * 40)

        # --- 2. Load Model ---
        print(f"Loading sentence transformer model: {MODEL_NAME}...")
        start_time = time.time()
        
        # Auto-detect Mac Apple Silicon (MPS)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Model loaded to {device.upper()} in {time.time() - start_time:.2f}s")

        # --- 3. Load FAISS Index ---
        print(f"Loading FAISS index: {INDEX_PATH}...")
        start_time = time.time()
        index = faiss.read_index(INDEX_PATH)
        
        # Set nprobe if the index is IVF (optimized)
        try:
            ivf_index = faiss.downcast_index(index.index) 
            ivf_index.nprobe = 16
            print(f"Index is IndexIDMap(IVF). Set nprobe = {ivf_index.nprobe}")
        except:
            print("Index is Flat. nprobe not applicable.")
            
        print(f"Index loaded in {time.time() - start_time:.2f}s (Total vectors: {index.ntotal})")
        
        # --- 4. Connect to Metadata DB ---
        print(f"Connecting to metadata DB: {DB_PATH}...")
        db_conn = sqlite3.connect(DB_PATH)
        cursor = db_conn.cursor()
        print("Database connected.")

        # --- 5. Encode Query ---
        print("\nEncoding query string...")
        start_time = time.time()
        query_vector = model.encode(
            [query_string], 
            normalize_embeddings=True, # Critical for Inner Product (IP) search
            convert_to_numpy=True
        ).astype(np.float32)
        print(f"Query encoded in {time.time() - start_time:.2f}s")

        # --- 6. Search FAISS ---
        print(f"Searching for top {TOP_K} results...")
        start_time = time.time()
        # 'distances' will be the cosine similarity scores (higher is better)
        # 'indices' will be the 'article_id's we stored
        distances, indices = index.search(query_vector, TOP_K)
        print(f"Search completed in {time.time() - start_time:.2f}s")

        # --- 7. Fetch & Display Results ---
        print("\n--- Top Results ---")
        
        # Get the lists of scores and IDs from the 2D arrays
        scores = distances[0]
        article_ids = indices[0]
        
        for i in range(TOP_K):
            score = scores[i]
            article_id = article_ids[i]
            
            # Query the DB to get the title
            cursor.execute("SELECT title, char_count FROM articles WHERE article_id = ?", (int(article_id),))
            row = cursor.fetchone()
            
            if row:
                title, char_count = row
                print(f"\n{i+1}. {title}")
                print(f"   (Similarity: {score:.4f}, Article ID: {article_id}, Length: {char_count} chars)")
            else:
                print(f"\n{i+1}. Error: Could not find metadata for Article ID {article_id}")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
    
    finally:
        if db_conn:
            db_conn.close()
            print("\n" + "-" * 40)
            print("Database connection closed.")

if __name__ == "__main__":
    # --- ADD THIS LINE ---
    # Apply the multiprocessing fix (must be inside __name__ == "__main__")
    torch.multiprocessing.set_start_method('spawn', force=True) 
    
    # Get the query string from all command-line arguments joined together
    query = " ".join(sys.argv[1:])
    search(query)