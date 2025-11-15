#!/usr/bin/env python3
"""
Multithreaded BZ2 decompression utility
Decompresses all .bz2 files in a directory using parallel workers
"""
import bz2
import os
import sys
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- CONFIG ---
SOURCE_DIR = "data/pageviews"
MAX_WORKERS = 8  # Adjust based on CPU cores
CHUNK_SIZE = 16 * 1024 * 1024  # 16MB chunks
# ----------------

def decompress_file(filepath):
    """Decompress a single .bz2 file"""
    filename = os.path.basename(filepath)
    output_path = filepath[:-4]  # Remove .bz2 extension
    
    # Skip if already decompressed
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        return (filename, 'skipped', size, 0)
    
    start_time = time.time()
    
    try:
        compressed_size = os.path.getsize(filepath)
        decompressed_size = 0
        
        with bz2.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    decompressed_size += len(chunk)
        
        elapsed = time.time() - start_time
        rate = decompressed_size / elapsed if elapsed > 0 else 0
        
        return (filename, 'success', decompressed_size, elapsed)
        
    except Exception as e:
        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)
        return (filename, 'failed', 0, 0, str(e))

def format_bytes(bytes_val):
    """Format bytes into human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def main():
    print("Multithreaded BZ2 Decompressor")
    print("=" * 80)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Chunk size: {format_bytes(CHUNK_SIZE)}")
    print()
    
    # Find all .bz2 files
    file_pattern = os.path.join(SOURCE_DIR, "*.bz2")
    files_to_process = sorted(glob.glob(file_pattern))
    
    if not files_to_process:
        print(f"[ERROR] No .bz2 files found in {SOURCE_DIR}")
        sys.exit(1)
    
    # Calculate total compressed size
    total_compressed = sum(os.path.getsize(f) for f in files_to_process)
    
    print(f"Found {len(files_to_process)} .bz2 files")
    print(f"Total compressed size: {format_bytes(total_compressed)}")
    print()
    
    # Check for already decompressed files
    already_decompressed = []
    for f in files_to_process:
        output_path = f[:-4]
        if os.path.exists(output_path):
            already_decompressed.append(f)
    
    if already_decompressed:
        print(f"Found {len(already_decompressed)} already decompressed files")
        print("These will be skipped:")
        for f in already_decompressed[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(already_decompressed) > 5:
            print(f"  ... and {len(already_decompressed) - 5} more")
        print()
    
    # Confirm before proceeding
    files_to_decompress = len(files_to_process) - len(already_decompressed)
    if files_to_decompress > 0:
        print(f"Will decompress {files_to_decompress} files using {MAX_WORKERS} workers")
        print("This may take significant time and disk space.")
        print()
        print("Continue? (y/n): ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("Cancelled by user")
            sys.exit(0)
        print()
    
    # Process files in parallel
    print("Starting decompression...")
    print("-" * 80)
    
    start_time = time.time()
    completed = 0
    skipped = 0
    failed = 0
    total_decompressed_size = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files
        future_to_file = {
            executor.submit(decompress_file, filepath): filepath
            for filepath in files_to_process
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            result = future.result()
            
            filename = result[0]
            status = result[1]
            size = result[2]
            elapsed = result[3]
            
            completed += 1
            
            if status == 'skipped':
                skipped += 1
                print(f"  [{completed}/{len(files_to_process)}] ⊘ {filename} (already exists)")
            
            elif status == 'success':
                total_decompressed_size += size
                rate = size / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{len(files_to_process)}] ✓ {filename}")
                print(f"      Size: {format_bytes(size)} | Time: {elapsed:.1f}s | Rate: {format_bytes(rate)}/s")
            
            elif status == 'failed':
                failed += 1
                error = result[4] if len(result) > 4 else "Unknown error"
                print(f"  [{completed}/{len(files_to_process)}] ✗ {filename}: {error}")
    
    total_elapsed = time.time() - start_time
    
    print()
    print("=" * 80)
    print("Decompression Complete!")
    print("=" * 80)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Decompressed: {completed - skipped - failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total decompressed size: {format_bytes(total_decompressed_size)}")
    
    if total_elapsed > 0:
        avg_rate = total_decompressed_size / total_elapsed
        print(f"  Average rate: {format_bytes(avg_rate)}/s")
    
    print()
    
    if failed > 0:
        print(f"[WARNING] {failed} file(s) failed to decompress")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Decompression cancelled by user")
        sys.exit(1)