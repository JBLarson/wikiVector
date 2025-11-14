import sqlite3
import time
import bz2
import io
import sys
import os
import glob
from collections import defaultdict

# --- CONFIG ---
DB_PATH = "data/embeddings/metadata.db"
PAGEVIEW_DIR = "data/pageviews" # Directory to read files from
# ----------------

def main():
    print(f"Starting pageview processor from local directory: {PAGEVIEW_DIR}...")
    start_time = time.time()
    
    conn = None
    try:
        # --- 1. Find local files to process ---
        file_pattern = os.path.join(PAGEVIEW_DIR, "pageviews-*-user.bz2")
        files_to_process = glob.glob(file_pattern)
        
        if not files_to_process:
            print(f"Error: No '...-user.bz2' files found in {PAGEVIEW_DIR}")
            print("Please run the `download_pageviews.py` script first.")
            sys.exit(1)
            
        print(f"Found {len(files_to_process)} local files to process.")
        
        total_views = defaultdict(int)

        # --- 2. Process files locally ---
        for i, filepath in enumerate(files_to_process):
            print(f"  ...processing file {i+1}/{len(files_to_process)}: {os.path.basename(filepath)}", end='\r')
            with bz2.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Format: project page_title hourly_counts_string
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        project, title, counts_str = parts[0], parts[1], parts[2]
                        
                        if project == "en" and not ":" in title:
                            try:
                                total_day_views = 0
                                for item in counts_str.split(','):
                                    if item and len(item) > 1:
                                        total_day_views += int(item[1:])
                                
                                lookup_key = title.lower()
                                total_views[lookup_key] += total_day_views
                            except (ValueError, TypeError, IndexError):
                                continue
        
        print(f"\nFinished processing all files. Found {len(total_views):,} unique articles.")

        # --- 3. Update our metadata.db ---
        print(f"Connecting to {DB_PATH} to update pageview counts...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        updates = [
            (count, lookup_key) 
            for lookup_key, count in total_views.items()
            if count > 0
        ]
        
        print(f"Updating {len(updates):,} articles with their pageview counts...")
        
        print("Resetting all pageviews to 0...")
        cursor.execute("UPDATE articles SET pageviews = 0")
        
        print("Applying new pageview counts...")
        cursor.executemany("""
            UPDATE articles 
            SET pageviews = ? 
            WHERE lookup_title = ?
        """, updates)
        
        conn.commit()
        
        print(f"\nPageview processing complete in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
            
if __name__ == "__main__":
    main()