#!/usr/bin/env python3
"""
Backlink Computation Script (Direct) - v4 (Corrected Regex)

This script replaces the entire 'temp_wiki.db' workflow.

It reads the 'pagelinks.sql' dump ONCE, builds a dictionary of 
backlink counts in memory, and then updates the 'metadata.db'
directly.

FIXES:
1.  Uses the correct 3-field regex `(INT, INT, INT)` based on
    diagnostic output of the pagelinks.sql file.
2.  Handles the 'INSERT INTO `pagelinks`' line with backticks.
3.  Fixes the final stats-printing bug for NoneType.
"""
import sqlite3
import time
import sys
import os
import re
from collections import defaultdict

# --- CONFIG ---
PAGELINKS_SQL_PATH = "data/dumps_sql/enwiki-latest-pagelinks.sql"
METADATA_DB_PATH = "data/metadata.db"
# ----------------

# --- THIS IS THE CRITICAL FIX ---
# Pre-compile regex for finding 3-field integer tuples:
# (pl_from, pl_from_namespace, pl_target_id)
# e.g., (1939,0,2)
TUPLE_REGEX = re.compile(r"\((\d+),(\d+),(\d+)\)")
# --------------------------------

def compute_backlinks():
    print("=" * 80)
    print("BACKLINK PROCESSOR (Direct-to-DB, v4)")
    print("=" * 80)
    print()
    
    if not os.path.exists(PAGELINKS_SQL_PATH):
        print(f"✗ ERROR: {PAGELINKS_SQL_PATH} not found")
        print(f"  (Looking for it from the project root directory)")
        sys.exit(1)
        
    if not os.path.exists(METADATA_DB_PATH):
        print(f"✗ ERROR: {METADATA_DB_PATH} not found. Run Stage 1 (embeddings) first.")
        sys.exit(1)
    
    file_size = os.path.getsize(PAGELINKS_SQL_PATH)
    print(f"Parsing: {PAGELINKS_SQL_PATH} ({file_size / (1024**3):.2f} GB)")
    print(f"Target DB: {METADATA_DB_PATH}")
    print()
    
    total_start = time.time()
    
    # --- Phase 1: Parse SQL and build counts in memory ---
    print("Phase 1: Reading pagelinks and counting backlinks...")
    print("This will take 2-4 hours and use RAM for the count dictionary.")
    print()
    
    backlink_counts = defaultdict(int)
    total_links_scanned = 0
    total_lines = 0
    
    try:
        with open(PAGELINKS_SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Use the exact line start from your diagnostic
                if not line.startswith('INSERT INTO `pagelinks`'):
                    continue
                
                total_lines += 1
                matches = TUPLE_REGEX.finditer(line)
                
                for match in matches:
                    try:
                        # --- FIELDS ARE NOW CORRECT ---
                        # pl_from = int(match.group(1)) # We don't need this
                        pl_from_namespace = int(match.group(2))
                        
                        # We only care about links FROM namespace 0 (main articles)
                        if pl_from_namespace == 0:
                            pl_target_id = int(match.group(3))
                            backlink_counts[pl_target_id] += 1
                            
                        total_links_scanned += 1
                        
                        if total_links_scanned % 10_000_000 == 0:
                            elapsed = time.time() - total_start
                            rate = total_links_scanned / elapsed
                            print(f"  Scanned: {total_links_scanned:12,} links | Found: {len(backlink_counts):9,} target articles | {rate:,.0f}/sec", end='\r')
                            
                    except Exception:
                        pass # Ignore parsing errors on individual tuples
        
        elapsed_parse = time.time() - total_start
        print(f"\n\nPhase 1 Complete in {elapsed_parse/60:.1f} minutes.")
        print(f"  Scanned {total_links_scanned:,} total pagelinks from {total_lines:,} INSERT statements.")
        print(f"  Found {len(backlink_counts):,} unique articles with backlinks.")
        print()
        
    except KeyboardInterrupt:
        print("\n\n!!! INTERRUPTED BY USER !!!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Update the main database ---
    print("Phase 2: Updating metadata.db...")
    conn = None
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")

        print("  Resetting all backlinks to 0...")
        cursor.execute("UPDATE articles SET backlinks = 0")
        
        print(f"  Preparing {len(backlink_counts):,} updates...")
        # Create an iterator of tuples (count, article_id)
        update_data = [(count, article_id) for article_id, count in backlink_counts.items()]
        
        print("  Applying updates in a single transaction (this may take a few minutes)...")
        start_update = time.time()
        
        cursor.executemany(
            """
            UPDATE articles 
            SET backlinks = ? 
            WHERE article_id = ?
            """, 
            update_data
        )
        
        conn.commit()
        elapsed_update = time.time() - start_update
        print(f"  ✓ Database update complete in {elapsed_update:.1f} seconds.")
        print()

        # --- Step 3: Show statistics ---
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*), AVG(backlinks), MAX(backlinks), SUM(backlinks) FROM articles WHERE backlinks > 0")
        results = cursor.fetchone()
        
        count = results[0]
        
        if count > 0 and results[1] is not None:
            avg, max_links, total_links = results[1], results[2], results[3]
            print(f"Total articles in metadata.db: {total_articles:,}")
            print(f"Articles with backlinks: {count:,} ({100*count/total_articles:.1f}%)")
            print(f"Average backlinks: {avg:.1f}")
            print(f"Max backlinks: {max_links:,}")
            print(f"Total backlink count: {total_links:,}")
        else:
            print(f"Total articles in metadata.db: {total_articles:,}")
            print(f"Articles with backlinks: 0 (0.0%)")
            print("  No backlink data was successfully merged.")
            
        print()
        
        # Top 20 most linked articles
        print("Top 20 most linked articles:")
        cursor.execute("""
            SELECT title, backlinks 
            FROM articles 
            WHERE backlinks > 0 
            ORDER BY backlinks DESC 
            LIMIT 20
        """)
        for i, (title, backlinks) in enumerate(cursor.fetchall(), 1):
            print(f"  {i:2d}. {backlinks:>8,} - {title}")
        print()
        
        print("=" * 80)
        print("COMPUTE BACKLINKS COMPLETE")
        print("=" * 80)
        print(f"Total time: {(time.time() - total_start)/60:.1f} minutes")
        print()
        
        if count < total_articles * 0.5:
            print("✗ WARNING: Less than 50% of articles have backlinks")
            print("This may indicate an issue with the merge.")
        else:
            print("✓ Backlink data successfully merged into metadata.db")

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
        sys.exit(1)
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    compute_backlinks()