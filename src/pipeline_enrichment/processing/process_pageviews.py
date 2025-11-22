#!/usr/bin/env python3
"""
Process Wikipedia pageview dumps - v4 (Corrected Join Key)

This script reads all pageview files (bz2 or decompressed) from the
pageview directory, aggregates all counts into an in-memory dictionary,
and then updates the metadata.db in a single, atomic transaction.

FIX: The join key is title.lower(), preserving underscores, to match
the 'lookup_title' column in metadata.db.
"""
import sqlite3
import time
import bz2
import sys
import os
import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote

# --- CONFIG ---
METADATA_DB_PATH = "data/metadata.db"
PAGEVIEW_DIR = "data/pageviews"
MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
# ----------------

def smart_open(filepath, mode='rt'):
    """Open file - automatically handles .bz2 or plain text"""
    if filepath.endswith('.bz2'):
        return bz2.open(filepath, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filepath, mode, encoding='utf-8', errors='ignore')

def format_bytes(bytes_val):
    """Format bytes into human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def process_single_file(filepath):
    """
    Process a single pageview file and return a dict of {lookup_key: view_count}.
    
    Format:
    project title domain device_type view_count hourly_breakdown
    """
    local_views = defaultdict(int)
    filename = os.path.basename(filepath)
    
    lines_processed = 0
    lines_matched = 0
    lines_skipped = 0
    
    try:
        with smart_open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                lines_processed += 1
                parts = line.strip().split()
                
                if len(parts) < 5:
                    lines_skipped += 1
                    continue
                
                project = parts[0]
                title_encoded = parts[1]
                view_count_str = parts[4]
                
                if not (project == "en.wikipedia" or project == "en"):
                    continue
                
                if ":" in title_encoded or title_encoded == "-":
                    continue
                
                try:
                    view_count = int(view_count_str)
                    
                    if view_count > 0:
                        try:
                            title = unquote(title_encoded)
                        except Exception:
                            title = title_encoded
                        
                        # --- THIS IS THE FIX ---
                        # Your diagnostic proves the join key is just lowercased.
                        # The .replace('_', ' ') was the bug.
                        lookup_key = title.lower()
                        # -----------------------

                        local_views[lookup_key] += view_count
                        lines_matched += 1
                        
                except (ValueError, IndexError):
                    lines_skipped += 1
                    continue
        
        summary = {
            'lines_processed': lines_processed,
            'lines_matched': lines_matched,
            'lines_skipped': lines_skipped,
            'unique_articles': len(local_views)
        }
        
        return (filename, local_views, None, summary)
        
    except Exception as e:
        return (filename, None, str(e), None)

def merge_results(total_views, file_views):
    """Merge views from one file into the total."""
    for title, count in file_views.items():
        total_views[title] += count

def main():
    print("Wikipedia Pageview Processor (v4 - Corrected Join Key)")
    print("=" * 80)
    print(f"Directory: {PAGEVIEW_DIR}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Database: {METADATA_DB_PATH}")
    print()
    
    total_start_time = time.time()
    
    conn = None
    
    try:
        # --- Step 1: Find files (both .bz2 and decompressed) ---
        print("Phase 1: Finding pageview files...")
        bz2_pattern = os.path.join(PAGEVIEW_DIR, "pageviews-*-user.bz2")
        txt_pattern = os.path.join(PAGEVIEW_DIR, "pageviews-*-user")
        
        bz2_files = set(glob.glob(bz2_pattern))
        txt_files = set(glob.glob(txt_pattern))
        
        files_to_process = []
        for bz2_file in sorted(bz2_files):
            decompressed = bz2_file[:-4]  # Remove .bz2
            if decompressed in txt_files:
                files_to_process.append(decompressed)  # Use decompressed
            else:
                files_to_process.append(bz2_file)  # Use compressed
        
        for txt_file in sorted(txt_files):
            if txt_file + ".bz2" not in bz2_files and txt_file not in files_to_process:
                files_to_process.append(txt_file)
        
        if not files_to_process:
            print(f"✗ ERROR: No pageview files found in {PAGEVIEW_DIR}")
            sys.exit(1)
        
        print(f"  ✓ Found {len(files_to_process)} files to process")
        total_size = sum(os.path.getsize(f) for f in files_to_process)
        print(f"  Total size: {format_bytes(total_size)}")
        print()
        
        # --- Step 2: Process files in parallel ---
        print("Phase 2: Processing all files (this will take ~40-50 minutes)...")
        print("-" * 80)
        start_time = time.time()
        
        total_views = defaultdict(int)
        completed = 0
        failed = 0
        total_lines = 0
        total_matched = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(process_single_file, filepath): filepath 
                for filepath in files_to_process
            }
            
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                filename, file_views, error, summary = future.result()
                
                completed += 1
                
                if error:
                    failed += 1
                    print(f"  [{completed}/{len(files_to_process)}] ✗ {filename}")
                    print(f"      Error: {error}")
                else:
                    merge_results(total_views, file_views)
                    total_lines += summary['lines_processed']
                    total_matched += summary['lines_matched']
                    
                    file_type = "bz2" if filepath.endswith('.bz2') else "txt"
                    print(f"  [{completed}/{len(files_to_process)}] ✓ {filename} ({file_type})")
                    print(f"      Articles: {summary['unique_articles']:,} | Matched: {summary['lines_matched']:,}/{summary['lines_processed']:,}")
        
        elapsed = time.time() - start_time
        
        print()
        print(f"Phase 2 complete in {elapsed/60:.1f} minutes")
        print(f"  ✓ Processed: {completed - failed}/{len(files_to_process)} files (Failed: {failed})")
        print(f"  Total lines scanned: {total_lines:,}")
        print(f"  Total lines matched: {total_matched:,}")
        print(f"  ✓ Found {len(total_views):,} unique articles with pageviews.")
        print()
        
        if len(total_views) == 0:
            print("✗ ERROR: No pageviews extracted from any file!")
            sys.exit(1)
        
        # --- Step 3: Update database ---
        print("Phase 3: Updating database in a single transaction...")
        print("-" * 80)
        start_time = time.time()
        
        if not os.path.exists(METADATA_DB_PATH):
            print(f"✗ ERROR: Database not found: {METADATA_DB_PATH}")
            sys.exit(1)
        
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        
        print("  Resetting all pageviews to 0...")
        cursor.execute("UPDATE articles SET pageviews = 0")
        
        print(f"  Preparing {len(total_views):,} updates...")
        updates = [
            (count, lookup_key) 
            for lookup_key, count in total_views.items()
        ]
        
        print("  Applying pageview counts (this may take a minute)...")
        cursor.executemany("""
            UPDATE articles 
            SET pageviews = ? 
            WHERE lookup_title = ?
        """, updates)
        
        conn.commit()
        
        elapsed = time.time() - start_time
        print(f"  ✓ Phase 3 complete in {elapsed:.1f}s")
        print()
        
        # --- Step 4: Show statistics ---
        cursor.execute("""
            SELECT COUNT(*), AVG(pageviews), MAX(pageviews), SUM(pageviews)
            FROM articles 
            WHERE pageviews > 0
        """)
        count, avg, max_views, total = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        
        print("=" * 80)
        print("Processing Complete!")
        print("=" * 80)
        print(f"Total time: {(time.time() - total_start_time)/60:.1f} minutes")
        print()
        print("Database statistics:")
        print(f"  Total articles in DB: {total_articles:,}")
        print(f"  Articles with pageviews: {count:,} ({100*count/total_articles:.1f}%)")
        print(f"  Average pageviews: {avg:,.1f}")
        print(f"  Max pageviews: {max_views:,}")
        print(f"  Total pageviews: {total:,}")
        print()
        
        # Show top 20
        print("Top 20 most viewed articles:")
        cursor.execute("""
            SELECT title, pageviews 
            FROM articles 
            WHERE pageviews > 0 
            ORDER BY pageviews DESC 
            LIMIT 20
        """)
        for i, (title, views) in enumerate(cursor.fetchall(), 1):
            print(f"  {i:2d}. {views:>12,} - {title}")
        print()
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Process cancelled by user")
        if conn:
            conn.rollback()
        sys.exit(1)
        
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