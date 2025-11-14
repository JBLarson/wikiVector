import sqlite3
import time
import sys
import os

# --- CONFIG ---
DB_PATH = "data/embeddings/metadata.db"
TEMP_DB_PATH = "data/temp_wiki.db"
# ----------------

def compute_backlinks():
    print("Starting backlink computation...")
    total_start_time = time.time()
    
    conn = None
    
    try:
        if not os.path.exists(TEMP_DB_PATH):
            print(f"Error: {TEMP_DB_PATH} does not exist!")
            print("Please run processPage.py and processPagelinks.py first.")
            sys.exit(1)
        
        if not os.path.exists(DB_PATH):
            print(f"Error: {DB_PATH} does not exist!")
            sys.exit(1)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Attach temp database
        print(f"Attaching {TEMP_DB_PATH}...")
        cursor.execute(f"ATTACH DATABASE '{TEMP_DB_PATH}' AS wiki")
        
        # Verify tables exist
        cursor.execute("SELECT name FROM wiki.sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Found tables in temp database: {tables}")
        
        if 'page' not in tables or 'pagelinks' not in tables:
            print("Error: Missing page or pagelinks table!")
            sys.exit(1)
        
        # Reset all backlinks
        print("Resetting all backlinks to 0...")
        cursor.execute("UPDATE articles SET backlinks = 0")
        conn.commit()
        
        # Compute backlinks
        print("Computing backlink counts (this may take 20-60 minutes)...")
        start_time = time.time()
        
        cursor.execute("""
            UPDATE articles
            SET backlinks = (
                SELECT COUNT(*)
                FROM wiki.pagelinks pl
                WHERE pl.pl_target_id = articles.article_id
            )
        """)
        
        conn.commit()
        
        print(f"Backlink computation complete in {time.time() - start_time:.2f}s")
        
        # Show stats
        cursor.execute("SELECT COUNT(*), AVG(backlinks), MAX(backlinks) FROM articles WHERE backlinks > 0")
        count, avg, max_links = cursor.fetchone()
        print(f"\nStats: {count:,} articles with backlinks, avg={avg:.1f}, max={max_links:,}")
        
        # Show top articles
        print("\nTop 10 most linked articles:")
        cursor.execute("""
            SELECT title, backlinks 
            FROM articles 
            WHERE backlinks > 0 
            ORDER BY backlinks DESC 
            LIMIT 10
        """)
        for title, backlinks in cursor.fetchall():
            print(f"  {backlinks:,} - {title}")
        
        cursor.execute("DETACH DATABASE wiki")
        
        print(f"\nTotal processing complete in {time.time() - total_start_time:.2f}s")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    compute_backlinks()