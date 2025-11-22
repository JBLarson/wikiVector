#!/usr/bin/env python3
"""
WIKIPEDIA LINK GRAPH EXTRACTOR (Gold Standard)

Extracts the complete source→target link graph from Wikipedia XML dumps.
Stores in a persistent 'pagelinks' table for PageRank and future analysis.

This is a ONE-TIME extraction that gives you the full graph forever.

Expected runtime: ~2-3 hours for 6.7M articles
Expected storage: ~500M-1B links, ~15GB database size
"""

import os
import sys
import sqlite3
import time
from glob import glob
from pathlib import Path
from multiprocessing import Pool, Manager, Queue, Process, cpu_count
from queue import Empty
from lxml import etree as ET
import re
from tqdm import tqdm
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

XML_CHUNK_DIR = "../../../data/raw/xml_chunks"
METADATA_DB = "../../../data/metadata.db"
TEMP_DB_DIR = Path("/tmp/wiki_linkgraph_temp")
CHECKPOINT_FILE = TEMP_DB_DIR / "completed_files.txt"

NUM_WORKERS = cpu_count()
BATCH_SIZE = 10000
QUEUE_TIMEOUT = 10

# ============================================================================
# CHECKPOINT LOGIC
# ============================================================================

def load_completed_files() -> set:
    """Load the set of already-processed XML files from checkpoint."""
    if not CHECKPOINT_FILE.exists():
        return set()
    
    with open(CHECKPOINT_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def mark_file_complete(filepath: str):
    """Atomically mark an XML file as processed."""
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(f"{filepath}\n")
        f.flush()
        os.fsync(f.fileno())

# ============================================================================
# TITLE NORMALIZATION (BULLETPROOF)
# ============================================================================

WIKILINK_REGEX = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]')

def normalize_title(title: str) -> str:
    """
    Normalize a wikilink title to match the 'lookup_title' column format.
    
    This MUST match the format in metadata.db for successful lookups.
    
    Transformations:
    1. Strip whitespace
    2. Remove fragments (#...)
    3. Replace spaces with underscores
    4. Convert to lowercase
    
    Examples:
        "United States" → "united_states"
        "Python (programming language)" → "python_(programming_language)"
        "World War II#Causes" → "world_war_ii"
        "computer science" → "computer_science"
    """
    if not title:
        return None
    
    title = title.strip()
    
    # Remove fragments
    if '#' in title:
        title = title.split('#')[0].strip()
    
    if not title:
        return None
    
    # Replace spaces with underscores
    title = title.replace(' ', '_')
    
    # Convert to lowercase (lookup_title format)
    title = title.lower()
    
    return title

def is_special_page(normalized_title: str) -> bool:
    """Filter out special pages that shouldn't be in the link graph."""
    if not normalized_title:
        return True
    
    special_prefixes = [
        'file:', 'image:', 'category:', 'template:',
        'wikipedia:', 'help:', 'portal:', 'user:',
        'talk:', 'mediawiki:', 'special:', 'draft:'
    ]
    
    return any(normalized_title.startswith(prefix) for prefix in special_prefixes)

# ============================================================================
# TITLE LOOKUP CACHE (CRITICAL FOR PERFORMANCE)
# ============================================================================

class TitleCache:
    """
    In-memory cache of lookup_title → article_id mappings.
    
    This is CRITICAL for performance. Without it, we'd do 500M+ database
    lookups which would take weeks instead of hours.
    
    Memory usage: ~500MB for 6.7M articles (acceptable)
    """
    
    def __init__(self, db_path: str):
        print("Building title lookup cache...")
        start = time.time()
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Load ALL lookup_title → article_id mappings
        cursor.execute("SELECT lookup_title, article_id FROM articles")
        
        self.cache = {}
        for lookup_title, article_id in cursor.fetchall():
            self.cache[lookup_title] = article_id
        
        conn.close()
        
        elapsed = time.time() - start
        print(f"✓ Loaded {len(self.cache):,} title mappings in {elapsed:.1f}s")
        print(f"  Memory usage: ~{len(self.cache) * 50 / 1024 / 1024:.0f}MB")
    
    def get_article_id(self, normalized_title: str) -> int:
        """Get article_id for a normalized title. Returns None if not found."""
        return self.cache.get(normalized_title)

# ============================================================================
# XML PARSER WORKER
# ============================================================================

def xml_parser_worker(
    worker_id: int,
    xml_file_path: str,
    queue: Queue,
    completed_files: set,
    title_cache: TitleCache
):
    """
    Producer: Parses XML chunks and extracts source→target link pairs.
    
    For each article:
    1. Extract article_id (source)
    2. Parse all wikilinks in article text
    3. Normalize each link title
    4. Lookup target article_id
    5. Queue (source_id, target_id) pairs
    """
    
    if xml_file_path in completed_files:
        return
    
    namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
    articles_processed = 0
    links_found = 0
    links_filtered = 0
    links_not_found = 0
    
    try:
        with open(xml_file_path, 'rb') as f:
            context = ET.iterparse(f, events=('end',), tag=f"{namespace}page")
            
            for event, elem in context:
                try:
                    # Get source article ID
                    id_elem = elem.find(f"{namespace}id")
                    if id_elem is None:
                        continue
                    source_id = int(id_elem.text)
                    
                    # Skip redirects
                    if elem.find(f"{namespace}redirect") is not None:
                        continue
                    
                    # Only main namespace
                    ns_elem = elem.find(f"{namespace}ns")
                    ns_value = int(ns_elem.text) if ns_elem is not None else 0
                    if ns_value != 0:
                        continue
                    
                    # Get article text
                    revision = elem.find(f"{namespace}revision")
                    if revision is None:
                        continue
                    
                    text_elem = revision.find(f"{namespace}text")
                    if text_elem is None or not text_elem.text:
                        continue
                    
                    raw_wikitext = text_elem.text
                    
                    # Extract all wikilinks
                    matches = WIKILINK_REGEX.findall(raw_wikitext)
                    
                    # Build link pairs for this article
                    link_pairs = []
                    
                    for target_title in matches:
                        normalized = normalize_title(target_title)
                        
                        if not normalized:
                            links_filtered += 1
                            continue
                        
                        # Filter special pages
                        if is_special_page(normalized):
                            links_filtered += 1
                            continue
                        
                        # Lookup target article_id
                        target_id = title_cache.get_article_id(normalized)
                        
                        if target_id is None:
                            links_not_found += 1
                            continue
                        
                        # Valid link!
                        link_pairs.append((source_id, target_id))
                        links_found += 1
                    
                    # Queue all links from this article in one batch
                    if link_pairs:
                        queue.put(link_pairs)
                    
                    articles_processed += 1
                    
                except Exception as e:
                    # Don't crash on individual article failures
                    pass
                    
                finally:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

    except Exception as e:
        print(f"ERROR: Worker {worker_id} failed on {xml_file_path}: {e}")
    finally:
        mark_file_complete(xml_file_path)
        print(f"Parser finished {os.path.basename(xml_file_path)}: "
              f"{articles_processed:,} articles, {links_found:,} links, "
              f"{links_not_found:,} not found, {links_filtered:,} filtered")

# ============================================================================
# DATABASE WRITER WORKER
# ============================================================================

def link_writer_worker(
    worker_id: int,
    queue: Queue,
    temp_db_path: Path
):
    """
    Consumer: Receives link pairs and writes to temporary database.
    
    Each worker maintains its own temporary database to avoid lock contention.
    These are merged at the end.
    """
    
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    # Create temporary links table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS link_pairs (
            source_id INTEGER,
            target_id INTEGER,
            PRIMARY KEY (source_id, target_id)
        )
    """)
    conn.commit()
    
    link_buffer = []
    items_processed = 0
    links_processed = 0
    
    pbar = tqdm(
        desc=f"Writer {worker_id}",
        position=worker_id + 1,
        dynamic_ncols=True,
        leave=False,
        unit=" batches"
    )
    
    while True:
        try:
            # Receive batch of link pairs from one article
            link_pairs = queue.get(timeout=QUEUE_TIMEOUT)
            
            link_buffer.extend(link_pairs)
            links_processed += len(link_pairs)
            items_processed += 1
            pbar.update(1)
            
            # Flush to disk periodically
            if len(link_buffer) >= BATCH_SIZE:
                cursor.executemany(
                    "INSERT OR IGNORE INTO link_pairs (source_id, target_id) VALUES (?, ?)",
                    link_buffer
                )
                conn.commit()
                link_buffer = []
                
        except Empty:
            # Final flush
            if link_buffer:
                cursor.executemany(
                    "INSERT OR IGNORE INTO link_pairs (source_id, target_id) VALUES (?, ?)",
                    link_buffer
                )
                conn.commit()
            break
        except Exception as e:
            print(f"Writer {worker_id} Error: {e}")
            break
    
    pbar.close()
    conn.close()
    print(f"Writer {worker_id} finished: {items_processed:,} batches, {links_processed:,} links")

# ============================================================================
# MERGE TEMPORARY DATABASES
# ============================================================================

def merge_temp_databases(temp_dbs: List[Path], final_db_path: str):
    """
    Merge all temporary databases into the main metadata.db.
    
    Creates the final 'pagelinks' table with the complete link graph.
    """
    
    print("\n" + "="*80)
    print("MERGING TEMPORARY DATABASES")
    print("="*80)
    
    main_conn = sqlite3.connect(final_db_path)
    main_cursor = main_conn.cursor()
    
    # Drop existing pagelinks table if it exists (fresh start)
    print("\nDropping old pagelinks table if exists...")
    main_cursor.execute("DROP TABLE IF EXISTS pagelinks")
    
    # Create new pagelinks table
    print("Creating pagelinks table...")
    main_cursor.execute("""
        CREATE TABLE pagelinks (
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            PRIMARY KEY (source_id, target_id)
        )
    """)
    main_conn.commit()
    
    # Merge all temp databases
    print("\nMerging temporary databases...")
    total_links = 0
    
    for i, temp_db_path in enumerate(temp_dbs):
        if not temp_db_path.exists():
            print(f"Warning: Temp DB not found: {temp_db_path}")
            continue
        
        print(f"\n  Processing temp DB {i+1}/{len(temp_dbs)}: {temp_db_path.name}")
        
        # Attach temp database
        main_cursor.execute(f"ATTACH DATABASE '{temp_db_path}' AS temp_db")
        
        # Count links in this temp DB
        main_cursor.execute("SELECT COUNT(*) FROM temp_db.link_pairs")
        db_links = main_cursor.fetchone()[0]
        print(f"    Links in this DB: {db_links:,}")
        
        # Copy links
        main_cursor.execute("""
            INSERT OR IGNORE INTO pagelinks (source_id, target_id)
            SELECT source_id, target_id FROM temp_db.link_pairs
        """)
        main_conn.commit()
        
        # Detach
        main_cursor.execute("DETACH DATABASE temp_db")
        
        total_links += db_links
        print(f"    Cumulative total: {total_links:,}")
    
    # Create indexes for fast lookups
    print("\nCreating indexes (this may take a few minutes)...")
    
    print("  Creating index on source_id...")
    main_cursor.execute("CREATE INDEX IF NOT EXISTS idx_pagelinks_source ON pagelinks(source_id)")
    
    print("  Creating index on target_id...")
    main_cursor.execute("CREATE INDEX IF NOT EXISTS idx_pagelinks_target ON pagelinks(target_id)")
    
    main_conn.commit()
    
    # Get final stats
    main_cursor.execute("SELECT COUNT(*) FROM pagelinks")
    final_count = main_cursor.fetchone()[0]
    
    main_cursor.execute("SELECT COUNT(DISTINCT source_id) FROM pagelinks")
    unique_sources = main_cursor.fetchone()[0]
    
    main_cursor.execute("SELECT COUNT(DISTINCT target_id) FROM pagelinks")
    unique_targets = main_cursor.fetchone()[0]
    
    main_conn.close()
    
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"Total links: {final_count:,}")
    print(f"Unique sources: {unique_sources:,}")
    print(f"Unique targets: {unique_targets:,}")
    print(f"Average links per source: {final_count/unique_sources:.1f}")
    
    # Cleanup temp databases
    print("\nCleaning up temporary databases...")
    for temp_db_path in temp_dbs:
        if temp_db_path.exists():
            os.remove(temp_db_path)
    
    print("✓ Cleanup complete")

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """Main orchestrator for link graph extraction"""
    
    print("="*80)
    print("WIKIPEDIA LINK GRAPH EXTRACTOR (Gold Standard)")
    print("="*80)
    print()
    print("This script will:")
    print("  1. Parse all XML chunks")
    print("  2. Extract every source→target link")
    print("  3. Store in a persistent 'pagelinks' table")
    print("  4. Create indexes for fast lookups")
    print()
    print("Expected: ~2-3 hours, ~500M links, ~15GB database growth")
    print()
    
    # Validate inputs
    if not os.path.exists(METADATA_DB):
        print(f"ERROR: metadata.db not found at {METADATA_DB}")
        sys.exit(1)
    
    xml_files = sorted(glob(f"{XML_CHUNK_DIR}/*.xml"))
    if not xml_files:
        print(f"ERROR: No XML chunks found in {XML_CHUNK_DIR}")
        sys.exit(1)
    
    print(f"Found {len(xml_files)} XML chunks")
    
    # Setup temp directory
    if TEMP_DB_DIR.exists():
        print(f"Cleaning temporary DB directory: {TEMP_DB_DIR}")
        for f in TEMP_DB_DIR.glob("worker_*.db"):
            os.remove(f)
    TEMP_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean checkpoint (fresh start)
    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
    
    # Build title cache (CRITICAL)
    title_cache = TitleCache(METADATA_DB)
    
    # Setup multiprocessing
    completed_files = load_completed_files()
    files_to_process = [f for f in xml_files if f not in completed_files]
    
    print(f"\nFiles to process: {len(files_to_process)}/{len(xml_files)}")
    
    if not files_to_process:
        print("All files already processed! Skipping to merge...")
    else:
        print("\n" + "="*80)
        print("PHASE 1: EXTRACTING LINK GRAPH")
        print("="*80)
        print(f"Workers: {NUM_WORKERS} parsers + {NUM_WORKERS} writers")
        print()
        
        manager = Manager()
        link_queue = manager.Queue(maxsize=500)
        temp_dbs = [TEMP_DB_DIR / f"worker_{i}.db" for i in range(NUM_WORKERS)]
        
        # Start writer processes
        writer_processes = []
        for i in range(NUM_WORKERS):
            p = Process(target=link_writer_worker, args=(i, link_queue, temp_dbs[i]))
            writer_processes.append(p)
            p.start()
        
        # Start parser pool
        parser_pool = Pool(NUM_WORKERS)
        parser_results = []
        
        for xml_file in files_to_process:
            result = parser_pool.apply_async(
                xml_parser_worker,
                args=(0, xml_file, link_queue, completed_files, title_cache)
            )
            parser_results.append(result)
        
        parser_pool.close()
        
        # Wait for parsers to finish
        for result in parser_results:
            result.get()
        
        print("\nAll XML parsing complete. Waiting for writers to finish...")
        
        # Wait for writers
        for p in writer_processes:
            p.join(timeout=QUEUE_TIMEOUT + 5)
            if p.is_alive():
                p.terminate()
                print("Warning: Writer terminated forcefully")
    
    # Merge all temporary databases
    merge_temp_databases(temp_dbs, METADATA_DB)
    
    # Cleanup checkpoint
    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
    
    print("\n" + "="*80)
    print("✓ LINK GRAPH EXTRACTION COMPLETE")
    print("="*80)
    print()
    print("Next step: Run calculate_pagerank.py to compute importance scores")
    print()

if __name__ == "__main__":
    main()