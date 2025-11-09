#!/usr/bin/env python3
"""
Ultra-fast parallel XML chunking for Wikipedia dumps
Uses multiprocessing for both parsing AND writing
"""

from lxml import etree as ET
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import os

def write_chunk(chunk_data):
    """Worker function to write a chunk to disk"""
    chunk_id, pages, output_dir = chunk_data
    
    output_file = Path(output_dir) / f"chunk_{chunk_id:04d}.xml"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">\n')
        
        for page_xml in pages:
            f.write(page_xml)
            f.write('\n')
        
        f.write('</mediawiki>\n')
    
    return chunk_id, len(pages)

def chunk_wikipedia_xml(
    input_file: str,
    output_dir: str,
    pages_per_chunk: int = 10000,
    num_workers: int = None
):
    """
    Extract Wikipedia XML into chunks with maximum parallelism
    
    Strategy:
    1. Single-threaded parse (must be for XML structure)
    2. Parallel writes (8-16 workers writing chunks simultaneously)
    3. Async job submission (don't wait for writes to finish)
    """
    
    if num_workers is None:
        num_workers = cpu_count()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
    
    print(f"========================================")
    print(f"Fast Parallel Wikipedia Chunking")
    print(f"========================================")
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print(f"Pages per chunk: {pages_per_chunk:,}")
    print(f"Write workers: {num_workers}")
    print(f"CPU cores: {cpu_count()}")
    print()
    
    # Check file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    file_size_gb = os.path.getsize(input_file) / (1024**3)
    print(f"File size: {file_size_gb:.1f} GB")
    print()
    
    with open(input_file, 'rb') as f:
        context = ET.iterparse(f, events=('end',), tag=f'{namespace}page')
        
        chunk_id = 0
        current_chunk = []
        total_pages = 0
        
        # Create worker pool for async writes
        pool = Pool(num_workers)
        write_jobs = []
        
        start_time = time.time()
        last_report = start_time
        
        print("Parsing and chunking...")
        print()
        
        for event, elem in context:
            # Convert element to string
            page_xml = ET.tostring(elem, encoding='unicode')
            current_chunk.append(page_xml)
            total_pages += 1
            
            # Progress reporting (every 10k pages or 5 seconds)
            current_time = time.time()
            if total_pages % 10000 == 0 or (current_time - last_report) > 5:
                elapsed = current_time - start_time
                rate = total_pages / elapsed if elapsed > 0 else 0
                eta_seconds = (7_000_000 - total_pages) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"  Parsed: {total_pages:>8,} pages | "
                      f"Rate: {rate:>6,.0f} pages/sec | "
                      f"ETA: {eta_minutes:>5.1f} min | "
                      f"Chunks queued: {len(write_jobs)}")
                
                last_report = current_time
            
            # When chunk is full, submit write job asynchronously
            if len(current_chunk) >= pages_per_chunk:
                # Submit job and don't wait for it
                job = pool.apply_async(
                    write_chunk,
                    ((chunk_id, current_chunk, output_dir),)
                )
                write_jobs.append(job)
                
                chunk_id += 1
                current_chunk = []
            
            # Critical: clear element to free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        
        # Write final chunk
        if current_chunk:
            job = pool.apply_async(
                write_chunk,
                ((chunk_id, current_chunk, output_dir),)
            )
            write_jobs.append(job)
            chunk_id += 1
        
        # Now wait for all writes to complete
        pool.close()
        
        parsing_time = time.time() - start_time
        
        print()
        print(f"Parsing complete in {parsing_time/60:.1f} minutes")
        print(f"Waiting for {len(write_jobs)} chunks to finish writing...")
        print()
        
        # Wait for writes with progress
        completed = 0
        for i, job in enumerate(write_jobs):
            job.get()  # Block until this write finishes
            completed += 1
            if completed % 10 == 0 or completed == len(write_jobs):
                print(f"  Written: {completed}/{len(write_jobs)} chunks ({completed*100/len(write_jobs):.0f}%)")
        
        pool.join()
        
        total_time = time.time() - start_time
        
        print()
        print(f"========================================")
        print(f"✓ Chunking Complete!")
        print(f"========================================")
        print(f"Total pages: {total_pages:,}")
        print(f"Total chunks: {chunk_id}")
        print(f"Parsing time: {parsing_time/60:.1f} minutes")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average rate: {total_pages/total_time:.0f} pages/sec")
        print(f"Output: {output_dir}")
        print()
        
        # Verify output
        chunk_files = list(output_path.glob("chunk_*.xml"))
        total_size_gb = sum(f.stat().st_size for f in chunk_files) / (1024**3)
        print(f"Total output size: {total_size_gb:.1f} GB ({len(chunk_files)} files)")
        print()

if __name__ == "__main__":
    chunk_wikipedia_xml(
        input_file="/mnt/data-large/wikipedia/raw/enwiki-20251101-pages-articles-multistream.xml",
        output_dir="/mnt/data-large/wikipedia/raw/xml_chunks/",
        pages_per_chunk=10000,
        num_workers=16  # Use 16 workers for max write parallelism
    )
