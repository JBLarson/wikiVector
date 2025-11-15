import sqlite3
import time
import bz2
import sys
import os
import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- CONFIG ---
DB_PATH = "data/metadata.db"
PAGEVIEW_DIR = "data/pageviews"
MAX_WORKERS = 8  # Adjust based on your CPU cores
# ----------------

def process_single_file(filepath):
    """Process a single pageview file and return view counts."""
    local_views = defaultdict(int)
    filename = os.path.basename(filepath)
    
    try:
        with bz2.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts) >= 3:
                    project, title, counts_str = parts[0], parts[1], parts[2]
                    
                    if project == "en" and ":" not in title:
                        try:
                            total_day_views = sum(
                                int(item[1:]) 
                                for item in counts_str.split(',') 
                                if item and len(item) > 1
                            )
                            
                            if total_day_views > 0:
                                lookup_key = title.lower()
                                local_views[lookup_key] += total_day_views
                                
                        except (ValueError, TypeError, IndexError):
                            continue
        
        return (filename, local_views, None)
        
    except Exception as e:
        return (filename, None, str(e))

def merge_results(total_views, file_views):
    """Merge views from one file into the total."""
    for title, count in file_views.items():
        total_views[title] += count

def main():
    print(f"Starting multithreaded pageview processor")
    print(f"Directory: {PAGEVIEW_DIR}")
    print(f"Max workers: {MAX_WORKERS}")
    total_start_time = time.time()
    
    conn = None
    
    try:
        # --- Step 1: Find files ---
        file_pattern = os.path.join(PAGEVIEW_DIR, "pageviews-*-user.bz2")
        files_to_process = sorted(glob.glob(file_pattern))
        
        if not files_to_process:
            print(f"Error: No pageview files found in {PAGEVIEW_DIR}")
            sys.exit(1)
        
        print(f"Found {len(files_to_process)} files to process\n")
        
        # --- Step 2: Process files in parallel ---
        print("Phase 1: Processing files in parallel...")
        start_time = time.time()
        
        total_views = defaultdict(int)
        completed = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(process_single_file, filepath): filepath 
                for filepath in files_to_process
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                filename, file_views, error = future.result()
                
                completed += 1
                
                if error:
                    failed += 1
                    print(f"  [{completed}/{len(files_to_process)}] ✗ {filename}: {error}")
                else:
                    merge_results(total_views, file_views)
                    articles_found = len(file_views)
                    print(f"  [{completed}/{len(files_to_process)}] ✓ {filename} ({articles_found:,} articles)")
        
        elapsed = time.time() - start_time
        print(f"\nPhase 1 complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"  Processed: {completed - failed}/{len(files_to_process)} files")
        print(f"  Failed: {failed}")
        print(f"  Unique articles with views: {len(total_views):,}")
        
        # --- Step 3: Update database ---
        print(f"\nPhase 2: Updating database...")
        start_time = time.time()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        print("  Resetting all pageviews to 0...")
        cursor.execute("UPDATE articles SET pageviews = 0")
        
        print(f"  Preparing {len(total_views):,} updates...")
        updates = [
            (count, lookup_key) 
            for lookup_key, count in total_views.items()
        ]
        
        print("  Applying pageview counts...")
        cursor.executemany("""
            UPDATE articles 
            SET pageviews = ? 
            WHERE lookup_title = ?
        """, updates)
        
        conn.commit()
        
        elapsed = time.time() - start_time
        print(f"Phase 2 complete in {elapsed:.1f}s")
        
        # --- Step 4: Show statistics ---
        cursor.execute("""
            SELECT COUNT(*), AVG(pageviews), MAX(pageviews) 
            FROM articles 
            WHERE pageviews > 0
        """)
        count, avg, max_views = cursor.fetchone()
        
        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)
        print(f"Total time: {(time.time() - total_start_time)/60:.1f} minutes")
        print(f"\nDatabase statistics:")
        print(f"  Articles with pageviews: {count:,}")
        print(f"  Average pageviews: {avg:,.1f}")
        print(f"  Max pageviews: {max_views:,}")
        
        # Show top 10
        print(f"\nTop 10 most viewed articles:")
        cursor.execute("""
            SELECT title, pageviews 
            FROM articles 
            WHERE pageviews > 0 
            ORDER BY pageviews DESC 
            LIMIT 10
        """)
        for title, views in cursor.fetchall():
            print(f"  {views:>12,} - {title}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
        sys.exit(1)
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()