#!/usr/bin/env python3
"""
WIKIPEDIA PAGERANK CALCULATOR (Corrected for macOS/actual schema)

Calculates PageRank scores using the link graph extracted by extract_link_graph.py.
Works with the 'pagelinks' table schema.

Expected runtime: ~30-60 minutes for 6.7M articles
"""

import sqlite3
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

METADATA_DB = "../../../data/metadata.db"
DAMPING_FACTOR = 0.85  # Standard PageRank damping
ITERATIONS = 50        # Number of power iterations
CONVERGENCE_THRESHOLD = 1e-6  # Early stopping threshold

# ============================================================================
# BUILD LINK MATRIX
# ============================================================================

def build_link_matrix():
    """
    Build sparse adjacency matrix from pagelinks table.
    
    Matrix structure:
    - Rows = target articles (who receives the rank)
    - Cols = source articles (who distributes the rank)
    - Value = 1/outlinks (each source distributes rank equally)
    
    Returns:
        matrix: Sparse CSR matrix (memory efficient)
        article_ids: List of article_ids (index → article_id mapping)
    """
    
    print("\n" + "="*80)
    print("BUILDING LINK MATRIX")
    print("="*80)
    
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    
    # Get all article IDs (ordered for consistent indexing)
    print("\nLoading article IDs...")
    cursor.execute("SELECT article_id FROM articles ORDER BY article_id")
    article_ids = [row[0] for row in cursor.fetchall()]
    
    # Create bidirectional mapping
    id_to_idx = {article_id: idx for idx, article_id in enumerate(article_ids)}
    
    n = len(article_ids)
    print(f"Articles: {n:,}")
    
    # Count outlinks per source (for normalization)
    print("\nCounting outlinks per source...")
    cursor.execute("""
        SELECT source_id, COUNT(*) 
        FROM pagelinks 
        GROUP BY source_id
    """)
    
    outlink_counts = {}
    for source_id, count in tqdm(cursor.fetchall(), desc="Counting"):
        outlink_counts[source_id] = count
    
    print(f"Sources with outlinks: {len(outlink_counts):,}")
    
    # Build sparse matrix
    print("\nBuilding sparse matrix...")
    cursor.execute("SELECT source_id, target_id FROM pagelinks")
    
    # Use LIL (List of Lists) for efficient construction
    matrix = lil_matrix((n, n), dtype=np.float32)
    
    links_processed = 0
    links_skipped = 0
    
    for source_id, target_id in tqdm(cursor.fetchall(), desc="Building matrix"):
        if source_id in id_to_idx and target_id in id_to_idx:
            source_idx = id_to_idx[source_id]
            target_idx = id_to_idx[target_id]
            
            # Each source distributes 1/outlinks to each target
            weight = 1.0 / outlink_counts[source_id]
            matrix[target_idx, source_idx] = weight
            
            links_processed += 1
        else:
            links_skipped += 1
    
    print(f"Links processed: {links_processed:,}")
    if links_skipped > 0:
        print(f"Links skipped (IDs not in articles table): {links_skipped:,}")
    
    # Convert to CSR for efficient matrix-vector multiplication
    print("\nConverting to CSR format (optimized for computation)...")
    matrix = matrix.tocsr()
    
    conn.close()
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero entries: {matrix.nnz:,}")
    print(f"Sparsity: {100 * (1 - matrix.nnz / (n*n)):.2f}%")
    
    return matrix, article_ids

# ============================================================================
# PAGERANK ALGORITHM
# ============================================================================

def calculate_pagerank(matrix, damping=DAMPING_FACTOR, max_iterations=ITERATIONS):
    """
    Calculate PageRank using power iteration.
    
    Algorithm:
        PR(i) = (1-d)/N + d * SUM(PR(j) / L(j))
        
    Where:
        PR(i) = PageRank of page i
        d = damping factor (0.85)
        N = total number of pages
        PR(j) = PageRank of page j that links to i
        L(j) = number of outlinks from page j
    
    Returns:
        rank: NumPy array of PageRank scores (same order as article_ids)
    """
    
    print("\n" + "="*80)
    print("CALCULATING PAGERANK")
    print("="*80)
    print(f"Damping factor: {damping}")
    print(f"Max iterations: {max_iterations}")
    print()
    
    n = matrix.shape[0]
    
    # Initialize with uniform distribution
    rank = np.ones(n, dtype=np.float32) / n
    
    # Teleportation vector (for dangling nodes and random jumps)
    teleport = (1 - damping) / n
    
    print("Running power iteration...")
    for iteration in tqdm(range(max_iterations), desc="Iterations"):
        prev_rank = rank.copy()
        
        # PageRank update: PR = d*M*PR + (1-d)/N
        rank = damping * matrix.dot(rank) + teleport
        
        # Normalize (should sum to 1)
        rank = rank / rank.sum()
        
        # Check convergence
        delta = np.abs(rank - prev_rank).sum()
        
        if iteration % 10 == 0:
            tqdm.write(f"  Iteration {iteration}: delta = {delta:.8f}")
        
        if delta < CONVERGENCE_THRESHOLD:
            print(f"\n✓ Converged after {iteration + 1} iterations (delta = {delta:.8f})")
            break
    
    return rank

# ============================================================================
# UPDATE DATABASE
# ============================================================================

def update_database(article_ids, pagerank_scores):
    """
    Update metadata.db with PageRank scores.
    
    Scores are scaled to 0-100 for easier interpretation:
    - 0 = least important
    - 100 = most important (highest PageRank)
    """
    
    print("\n" + "="*80)
    print("UPDATING DATABASE")
    print("="*80)
    
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    
    # Add pagerank column if needed
    cursor.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'pagerank' not in columns:
        print("Adding 'pagerank' column...")
        cursor.execute("ALTER TABLE articles ADD COLUMN pagerank REAL DEFAULT 0.0")
        conn.commit()
    
    # Scale scores to 0-100
    max_score = pagerank_scores.max()
    min_score = pagerank_scores.min()
    
    print(f"\nPageRank range: {min_score:.10f} to {max_score:.10f}")
    
    scaled_scores = ((pagerank_scores - min_score) / (max_score - min_score)) * 100
    
    print(f"Scaled range: {scaled_scores.min():.2f} to {scaled_scores.max():.2f}")
    
    # Update database
    print("\nUpdating articles...")
    updates = [
        (float(score), int(article_id))
        for article_id, score in zip(article_ids, scaled_scores)
    ]
    
    batch_size = 10000
    for i in tqdm(range(0, len(updates), batch_size), desc="Updating"):
        batch = updates[i:i+batch_size]
        cursor.executemany(
            "UPDATE articles SET pagerank = ? WHERE article_id = ?",
            batch
        )
    
    conn.commit()
    conn.close()
    
    print("✓ Database updated")

# ============================================================================
# VALIDATION & RESULTS
# ============================================================================

def show_results():
    """Display top articles by PageRank and compare to backlinks."""
    
    print("\n" + "="*80)
    print("TOP 30 ARTICLES BY PAGERANK")
    print("="*80)
    print()
    
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT title, pagerank, backlinks, char_count
        FROM articles
        ORDER BY pagerank DESC
        LIMIT 30
    """)
    
    print(f"{'Rank':<5} {'PageRank':<10} {'Backlinks':<12} {'Chars':<10} {'Title'}")
    print("-"*80)
    
    for i, (title, pr, bl, cc) in enumerate(cursor.fetchall(), 1):
        bl_str = f"{bl:,}" if bl else "N/A"
        cc_str = f"{cc:,}" if cc else "N/A"
        print(f"{i:<5} {pr:>8.2f}   {bl_str:>10}   {cc_str:>8}   {title}")
    
    # Compare with backlinks (if available)
    cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
    if cursor.fetchone()[0] > 0:
        print("\n" + "="*80)
        print("CORRELATION: PageRank vs Backlinks")
        print("="*80)
        
        cursor.execute("""
            SELECT pagerank, backlinks 
            FROM articles 
            WHERE backlinks > 0 AND pagerank > 0
        """)
        
        data = cursor.fetchall()
        if len(data) > 100:  # Need enough data points
            pr_scores = np.array([row[0] for row in data])
            bl_counts = np.array([row[1] for row in data])
            
            correlation = np.corrcoef(pr_scores, bl_counts)[0, 1]
            print(f"\nPearson correlation: {correlation:.4f}")
            
            if correlation > 0.8:
                print("✓ Strong correlation - PageRank captures similar signal as backlinks")
            elif correlation > 0.6:
                print("~ Moderate correlation - PageRank adds new information")
            else:
                print("⚠ Weak correlation - PageRank captures very different importance")
    
    conn.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    print("="*80)
    print("WIKIPEDIA PAGERANK CALCULATOR")
    print("="*80)
    print()
    print("This will calculate importance scores for all articles using")
    print("the PageRank algorithm (same as Google).")
    print()
    
    start_time = time.time()
    
    # Check prerequisites
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pagelinks'")
    if not cursor.fetchone():
        print("ERROR: No 'pagelinks' table found!")
        print("Run extract_link_graph.py first to build the link graph.")
        conn.close()
        return
    
    cursor.execute("SELECT COUNT(*) FROM pagelinks")
    link_count = cursor.fetchone()[0]
    print(f"Found {link_count:,} links in pagelinks table")
    
    conn.close()
    
    # Build matrix
    matrix, article_ids = build_link_matrix()
    
    # Calculate PageRank
    pagerank = calculate_pagerank(matrix, damping=DAMPING_FACTOR, max_iterations=ITERATIONS)
    
    # Update database
    update_database(article_ids, pagerank)
    
    # Show results
    show_results()
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✓ PAGERANK CALCULATION COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print()
    print("The 'pagerank' column in metadata.db now contains importance scores (0-100)")
    print("Higher scores = more important/central articles")
    print()

if __name__ == "__main__":
    main()