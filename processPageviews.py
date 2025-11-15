#!/usr/bin/env python3
"""
Process Wikipedia pageview dumps - handles both .bz2 and decompressed files
Corrected format parsing based on actual dump structure
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
DB_PATH = "data/metadata.db"
PAGEVIEW_DIR = "data/pageviews"
MAX_WORKERS = 8  # Adjust based on your CPU cores
DEBUG_SAMPLES = 10  # Number of sample lines to show in debug mode
# ----------------

def smart_open(filepath, mode='rt'):
    """Open file - automatically handles .bz2 or plain text"""
    if filepath.endswith('.bz2'):
        return bz2.open(filepath, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filepath, mode, encoding='utf-8', errors='ignore')

def process_single_file(filepath, debug=False):
    """
    Process a single pageview file and return view counts.
    
    Actual format from October 2025 dumps:
    project title domain device_type view_count hourly_breakdown
    
    Example:
    aa.wikibooks File:xxx null desktop 1 J1
    en.wikipedia Python_(programming_language) null mobile-web 12345 A123B456...
    """
    local_views = defaultdict(int)
    filename = os.path.basename(filepath)
    
    lines_processed = 0
    lines_matched = 0
    lines_skipped = 0
    format_samples = []
    
    try:
        with smart_open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                lines_processed += 1
                
                parts = line.strip().split()
                
                # Collect debug samples
                if debug and len(format_samples) < DEBUG_SAMPLES:
                    format_samples.append({
                        'line_num': line_num,
                        'raw': line.strip(),
                        'parts': parts[:7] if len(parts) >= 7 else parts,
                        'count': len(parts)
                    })
                
                # Need at least 5 fields: project title domain device_type view_count
                if len(parts) < 5:
                    lines_skipped += 1
                    continue
                
                project = parts[0]
                title_encoded = parts[1]
                # parts[2] is domain (often "null")
                # parts[3] is device_type (desktop/mobile-web/etc)
                view_count_str = parts[4]  # THE ACTUAL VIEW COUNT
                
                # Filter for English Wikipedia only
                # Can be "en.wikipedia" or "en" depending on dump version
                if not (project == "en.wikipedia" or project == "en"):
                    continue
                
                # Skip special pages (contain colons) and files
                if ":" in title_encoded:
                    continue
                
                # Skip the main page dash entry
                if title_encoded == "-":
                    continue
                
                # Parse view count
                try:
                    view_count = int(view_count_str)
                    
                    if view_count > 0:
                        # URL decode and normalize the title
                        try:
                            title = unquote(title_encoded)
                        except Exception:
                            title = title_encoded
                        
                        # Convert underscores to spaces and lowercase for lookup
                        lookup_key = title.replace('_', ' ').lower()
                        local_views[lookup_key] += view_count
                        lines_matched += 1
                        
                except (ValueError, IndexError):
                    lines_skipped += 1
                    continue
        
        summary = {
            'lines_processed': lines_processed,
            'lines_matched': lines_matched,
            'lines_skipped': lines_skipped,
            'unique_articles': len(local_views),
            'samples': format_samples if debug else []
        }
        
        return (filename, local_views, None, summary)
        
    except Exception as e:
        return (filename, None, str(e), None)

def merge_results(total_views, file_views):
    """Merge views from one file into the total."""
    for title, count in file_views.items():
        total_views[title] += count

def format_bytes(bytes_val):
    """Format bytes into human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def main():
    print("Wikipedia Pageview Processor (Fixed Format)")
    print("=" * 80)
    print(f"Directory: {PAGEVIEW_DIR}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Database: {DB_PATH}")
    print()
    
    total_start_time = time.time()
    
    conn = None
    
    try:
        # --- Step 1: Find files (both .bz2 and decompressed) ---
        bz2_pattern = os.path.join(PAGEVIEW_DIR, "pageviews-*-user.bz2")
        txt_pattern = os.path.join(PAGEVIEW_DIR, "pageviews-*-user")
        
        bz2_files = set(glob.glob(bz2_pattern))
        txt_files = set(glob.glob(txt_pattern))
        
        # Remove .bz2 files if decompressed version exists
        files_to_process = []
        for bz2_file in sorted(bz2_files):
            decompressed = bz2_file[:-4]  # Remove .bz2
            if decompressed in txt_files:
                files_to_process.append(decompressed)  # Use decompressed
                print(f"  Using decompressed: {os.path.basename(decompressed)}")
            else:
                files_to_process.append(bz2_file)  # Use compressed
        
        # Add any txt files that don't have .bz2 equivalents
        for txt_file in sorted(txt_files):
            if txt_file + ".bz2" not in bz2_files and txt_file not in files_to_process:
                files_to_process.append(txt_file)
        
        if not files_to_process:
            print(f"[ERROR] No pageview files found in {PAGEVIEW_DIR}")
            print(f"Looking for: pageviews-*-user.bz2 or pageviews-*-user")
            
            if os.path.exists(PAGEVIEW_DIR):
                all_files = os.listdir(PAGEVIEW_DIR)
                print(f"\nFiles in directory: {len(all_files)}")
                for f in all_files[:10]:
                    print(f"  - {f}")
            else:
                print(f"\nDirectory does not exist: {PAGEVIEW_DIR}")
            
            sys.exit(1)
        
        print(f"\nFound {len(files_to_process)} files to process")
        
        # Show file info
        compressed_count = sum(1 for f in files_to_process if f.endswith('.bz2'))
        decompressed_count = len(files_to_process) - compressed_count
        print(f"  Compressed (.bz2): {compressed_count}")
        print(f"  Decompressed: {decompressed_count}")
        
        total_size = sum(os.path.getsize(f) for f in files_to_process)
        print(f"  Total size: {format_bytes(total_size)}")
        print()
        
        # --- Step 1.5: Debug first file ---
        print("Running diagnostic on first file...")
        print("-" * 80)
        first_file = files_to_process[0]
        _, test_views, error, summary = process_single_file(first_file, debug=True)
        
        if error:
            print(f"[ERROR] Failed to process test file: {error}")
            sys.exit(1)
        
        # Show format samples
        print("\nFormat analysis from first file:")
        print(f"File: {os.path.basename(first_file)}")
        print(f"Type: {'Compressed' if first_file.endswith('.bz2') else 'Decompressed'}")
        print()
        
        if summary['samples']:
            print("Sample lines showing field structure:")
            for sample in summary['samples']:
                print(f"\nLine {sample['line_num']}: ({sample['count']} fields)")
                print(f"  Raw: {sample['raw'][:120]}...")
                if len(sample['parts']) >= 5:
                    print(f"  [0] project:     {sample['parts'][0]}")
                    print(f"  [1] title:       {sample['parts'][1]}")
                    print(f"  [2] domain:      {sample['parts'][2]}")
                    print(f"  [3] device:      {sample['parts'][3]}")
                    print(f"  [4] view_count:  {sample['parts'][4]}")
                    if len(sample['parts']) >= 6:
                        print(f"  [5] hourly:      {sample['parts'][5][:20]}...")
        
        print()
        print("Processing results:")
        print(f"  Lines processed: {summary['lines_processed']:,}")
        print(f"  Lines matched (en.wikipedia): {summary['lines_matched']:,}")
        print(f"  Lines skipped: {summary['lines_skipped']:,}")
        print(f"  Unique articles found: {summary['unique_articles']:,}")
        
        # Show some example matches
        if test_views:
            print(f"\nExample articles found:")
            for i, (title, count) in enumerate(sorted(test_views.items(), key=lambda x: x[1], reverse=True)[:10]):
                print(f"  {count:>8,} views - {title}")
        
        print()
        
        if summary['unique_articles'] == 0:
            print("[ERROR] No pageviews extracted!")
            print("\nThe format appears to be:")
            print("  project title domain device_type view_count hourly_data")
            print("\nBut no 'en.wikipedia' or 'en' entries without colons were found.")
            print("\nPossible issues:")
            print("1. Different dump type than expected")
            print("2. All entries are special pages (contain ':')")
            print("3. Project field format is different")
            sys.exit(1)
        
        print(f"✓ Format validation passed!")
        print(f"✓ Found {summary['unique_articles']:,} unique English Wikipedia articles")
        print(f"\nProceeding with all {len(files_to_process)} files...")
        print()
        
        # --- Step 2: Process files in parallel ---
        print("Phase 1: Processing all files")
        print("-" * 80)
        start_time = time.time()
        
        total_views = defaultdict(int)
        completed = 0
        failed = 0
        total_lines = 0
        total_matched = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(process_single_file, filepath, False): filepath 
                for filepath in files_to_process
            }
            
            # Process results as they complete
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
                    print(f"      Articles: {summary['unique_articles']:,} | "
                          f"Matched: {summary['lines_matched']:,}/{summary['lines_processed']:,}")
        
        elapsed = time.time() - start_time
        
        print()
        print(f"Phase 1 complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"  Processed: {completed - failed}/{len(files_to_process)} files")
        print(f"  Failed: {failed}")
        print(f"  Total lines: {total_lines:,}")
        print(f"  Lines matched: {total_matched:,}")
        print(f"  Match rate: {100*total_matched/total_lines:.2f}%")
        print(f"  Unique articles with views: {len(total_views):,}")
        print()
        
        if len(total_views) == 0:
            print("[ERROR] No pageviews extracted from any file!")
            sys.exit(1)
        
        # --- Step 3: Update database ---
        print("Phase 2: Updating database")
        print("-" * 80)
        start_time = time.time()
        
        if not os.path.exists(DB_PATH):
            print(f"[ERROR] Database not found: {DB_PATH}")
            print("Please run prepDb.py first to create the required schema.")
            sys.exit(1)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verify schema
        cursor.execute("PRAGMA table_info(articles)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'pageviews' not in columns:
            print("[ERROR] Column 'pageviews' not found in articles table")
            print("Please run prepDb.py first to add required columns.")
            sys.exit(1)
        
        if 'lookup_title' not in columns:
            print("[ERROR] Column 'lookup_title' not found in articles table")
            print("Please run prepDb.py first to add required columns.")
            sys.exit(1)
        
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