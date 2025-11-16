#!/usr/bin/env python3
import sqlite3
import time
import sys
import os

# --- CONFIG ---
DB_PATH = "data/metadata.db"
TEMP_DB_PATH = "data/temp_wiki.db"
# ----------------

def compute_backlinks():
    print("=" * 80)
    print("MERGE TEMP_WIKI.DB INTO METADATA.DB")
    print("=" * 80)
    print()
    
    total_start = time.time()
    conn = None
    
    try:
        # Verify files exist
        if not os.path.exists(TEMP_DB_PATH):
            print(f"ERROR: {TEMP_DB_PATH} does not exist!")
            print("Run processPage.py and processPagelinks.py first")
            sys.exit(1)
        
        if not os.path.exists(DB_PATH):
            print(f"ERROR: {DB_PATH} does not exist!")
            sys.exit(1)
        
        # Verify temp database has data
        temp_conn = sqlite3.connect(TEMP_DB_PATH)
        temp_cursor = temp_conn.cursor()
        
        temp_cursor.execute("SELECT COUNT(*) FROM page")
        page_count = temp_cursor.fetchone()[0]
        
        temp_cursor.execute("SELECT COUNT(*) FROM pagelinks")
        link_count = temp_cursor.fetchone()[0]
        
        temp_conn.close()
        
        print("Temp database contents:")
        print(f"  Page records: {page_count:,}")
        print(f"  Pagelink records: {link_count:,}")
        print()
        
        if page_count < 1000000:
            print(f"ERROR: Only {page_count:,} page records (expected ~18M)")
            print("Run processPage.py first")
            sys.exit(1)
        
        if link_count < 1000000:
            print(f"ERROR: Only {link_count:,} pagelink records (expected ~500M)")
            print("Run processPagelinks.py first")
            sys.exit(1)
        
        print("✓ Temp database validation passed")
        print()
        
        # Connect to main database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Detach if already attached
        try:
            cursor.execute("DETACH DATABASE wiki")
        except:
            pass
        
        # Attach temp database
        print(f"Attaching {TEMP_DB_PATH}...")
        cursor.execute(f"ATTACH DATABASE '{TEMP_DB_PATH}' AS wiki")
        print()
        
        # Compute backlinks
        print("Computing backlink counts...")
        print("This may take 20-60 minutes depending on your disk speed")
        print()
        
        start_time = time.time()
        
        # Reset all backlinks to 0
        print("Step 1: Resetting backlinks to 0...")
        cursor.execute("UPDATE articles SET backlinks = 0")
        conn.commit()
        print(f"  Done in {time.time() - start_time:.1f}s")
        print()
        
        # Compute backlinks using JOIN
        print("Step 2: Computing backlink counts...")
        step_start = time.time()
        
        cursor.execute("""
            UPDATE articles
            SET backlinks = (
                SELECT COUNT(*)
                FROM wiki.pagelinks pl
                WHERE pl.pl_target_id = articles.article_id
                  AND pl.pl_from_namespace = 0
            )
        """)
        
        conn.commit()
        print(f"  Done in {(time.time() - step_start)/60:.1f} minutes")
        print()
        
        # Show statistics
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*), AVG(backlinks), MAX(backlinks), SUM(backlinks) FROM articles WHERE backlinks > 0")
        count, avg, max_links, total_links = cursor.fetchone()
        
        print(f"Total articles in metadata.db: {total_articles:,}")
        print(f"Articles with backlinks: {count:,} ({100*count/total_articles:.1f}%)")
        print(f"Average backlinks: {avg:.1f}")
        print(f"Max backlinks: {max_links:,}")
        print(f"Total backlink count: {total_links:,}")
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
        
        # Backlink distribution
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN backlinks = 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN backlinks BETWEEN 1 AND 10 THEN 1 ELSE 0 END),
                SUM(CASE WHEN backlinks BETWEEN 11 AND 100 THEN 1 ELSE 0 END),
                SUM(CASE WHEN backlinks BETWEEN 101 AND 1000 THEN 1 ELSE 0 END),
                SUM(CASE WHEN backlinks BETWEEN 1001 AND 10000 THEN 1 ELSE 0 END),
                SUM(CASE WHEN backlinks > 10000 THEN 1 ELSE 0 END)
            FROM articles
        """)
        zero, low, med, high, very_high, extreme = cursor.fetchone()
        
        print("Backlink distribution:")
        print(f"  0 backlinks: {zero:,} ({100*zero/total_articles:.1f}%)")
        print(f"  1-10: {low:,} ({100*low/total_articles:.1f}%)")
        print(f"  11-100: {med:,} ({100*med/total_articles:.1f}%)")
        print(f"  101-1000: {high:,} ({100*high/total_articles:.1f}%)")
        print(f"  1001-10000: {very_high:,} ({100*very_high/total_articles:.1f}%)")
        print(f"  >10000: {extreme:,} ({100*extreme/total_articles:.1f}%)")
        print()
        
        # Detach
        cursor.execute("DETACH DATABASE wiki")
        
        print("=" * 80)
        print("MERGE COMPLETE")
        print("=" * 80)
        print(f"Total time: {(time.time() - total_start)/60:.1f} minutes")
        print()
        
        if count < total_articles * 0.5:
            print("⚠ WARNING: Less than 50% of articles have backlinks")
            print("This may indicate an issue with the merge")
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