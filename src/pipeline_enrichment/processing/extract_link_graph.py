#!/usr/bin/env python3
"""
OPTIMIZED WIKIPEDIA LINK GRAPH EXTRACTOR

Key optimizations:
1. Memory-mapped title cache (shared across workers via fork)
2. No IPC queues - workers write directly to temp DBs
3. Batch XML parsing with aggressive memory cleanup
4. Optimized merge with bulk inserts

Expected: 2-3 hours for 6.7M articles on 16-core machine
"""

import os
import sys
import sqlite3
import time
import re
from glob import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from lxml import etree as ET
from tqdm import tqdm
from typing import Dict, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

XML_CHUNK_DIR = "../../../data/raw/xml_chunks"
METADATA_DB = "../../../data/metadata.db"
TEMP_DB_DIR = Path("/tmp/wiki_linkgraph_temp")

NUM_WORKERS = max(1, cpu_count() - 2)  # Leave 2 cores for OS
BATCH_SIZE = 50000  # Larger batches for fewer commits

# ============================================================================
# GLOBAL TITLE CACHE (Loaded once, inherited by workers via fork)
# ============================================================================

TITLE_TO_ID: Dict[str, int] = {}

def load_title_cache():
    """
    Load title->ID mapping into global memory.
    When using fork(), child processes inherit this via copy-on-write.
    """
    global TITLE_TO_ID
    
    print("\n" + "="*80)
    print("LOADING TITLE CACHE")
    print("="*80)
    
    start = time.time()
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    
    print("Fetching all lookup_title → article_id mappings...")
    cursor.execute("SELECT lookup_title, article_id FROM articles WHERE lookup_title IS NOT NULL")
    
    # Fetch all at once (faster than fetchmany for ~6M rows)
    rows = cursor.fetchall()
    TITLE_TO_ID = {lookup_title: article_id for lookup_title, article_id in rows}
    
    conn.close()
    
    elapsed = time.time() - start
    mem_mb = (len(TITLE_TO_ID) * 100) / (1024 * 1024)  # Rough estimate
    
    print(f"✓ Loaded {len(TITLE_TO_ID):,} titles in {elapsed:.1f}s")
    print(f"  Estimated memory: ~{mem_mb:.0f}MB")
    print()

# ============================================================================
# TITLE NORMALIZATION
# ============================================================================

WIKILINK_REGEX = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]')

def normalize_title(title: str) -> str:
    """Normalize wikilink title to match lookup_title format."""
    if not title:
        return None
    
    title = title.strip()
    
    # Remove fragment
    if '#' in title:
        title = title.split('#')[0].strip()
    
    if not title:
        return None
    
    # Convert to lookup_title format (underscores + lowercase)
    return title.replace(' ', '_').lower()

# Special page prefixes (lowercase)
SPECIAL_PREFIXES = (
    'file:', 'image:', 'category:', 'template:', 'wikipedia:',
    'help:', 'portal:', 'user:', 'talk:', 'mediawiki:', 'special:', 'draft:'
)

def is_special_page(normalized_title: str) -> bool:
    """Check if title is a special page."""
    return not normalized_title or normalized_title.startswith(SPECIAL_PREFIXES)

# ============================================================================
# WORKER PROCESS
# ============================================================================

def process_xml_files(args):
    """
    Worker process: Parse assigned XML files and extract links.
    Writes normalized title pairs to temp DB (lookup happens during merge).
    """
    worker_id, xml_files = args
    
    # Create worker-specific temp DB
    temp_db_path = TEMP_DB_DIR / f"worker_{worker_id:02d}.db"
    conn = sqlite3.connect(temp_db_path)
    
    # Performance settings
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
    
    # Create table - store source_id and target_title (lookup later)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS links (
            source_id INTEGER NOT NULL,
            target_title TEXT NOT NULL
        )
    """)
    
    cursor = conn.cursor()
    
    # Statistics
    articles_processed = 0
    links_found = 0
    links_filtered = 0
    files_processed = 0
    
    # Link buffer for batch inserts
    link_buffer = []
    
    namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
    
    for xml_file in xml_files:
        try:
            with open(xml_file, 'rb') as f:
                context = ET.iterparse(f, events=('end',), tag=f"{namespace}page")
                
                for event, elem in context:
                    try:
                        # Quick filters
                        ns_elem = elem.find(f"{namespace}ns")
                        if ns_elem is not None and ns_elem.text != '0':
                            continue
                        
                        if elem.find(f"{namespace}redirect") is not None:
                            continue
                        
                        # Get source ID
                        id_elem = elem.find(f"{namespace}id")
                        if id_elem is None:
                            continue
                        source_id = int(id_elem.text)
                        
                        # Get text
                        revision = elem.find(f"{namespace}revision")
                        if revision is None:
                            continue
                        
                        text_elem = revision.find(f"{namespace}text")
                        if text_elem is None or not text_elem.text:
                            continue
                        
                        # Extract wikilinks
                        raw_wikitext = text_elem.text
                        matches = WIKILINK_REGEX.findall(raw_wikitext)
                        
                        # Process each link
                        for target_title in matches:
                            normalized = normalize_title(target_title)
                            
                            if not normalized or is_special_page(normalized):
                                links_filtered += 1
                                continue
                            
                            # Store normalized title (lookup during merge)
                            link_buffer.append((source_id, normalized))
                            links_found += 1
                        
                        articles_processed += 1
                        
                        # Batch insert
                        if len(link_buffer) >= BATCH_SIZE:
                            cursor.executemany(
                                "INSERT INTO links (source_id, target_title) VALUES (?, ?)",
                                link_buffer
                            )
                            conn.commit()
                            link_buffer = []
                        
                    except Exception:
                        pass
                    finally:
                        # CRITICAL: Memory cleanup
                        elem.clear()
                        while elem.getprevious() is not None:
                            del elem.getparent()[0]
            
            files_processed += 1
            
        except Exception as e:
            print(f"Worker {worker_id}: Error in {xml_file}: {e}")
    
    # Final flush
    if link_buffer:
        cursor.executemany(
            "INSERT INTO links (source_id, target_title) VALUES (?, ?)",
            link_buffer
        )
        conn.commit()
    
    conn.close()
    
    return {
        'worker_id': worker_id,
        'files': files_processed,
        'articles': articles_processed,
        'links_found': links_found,
        'links_filtered': links_filtered
    }

# ============================================================================
# MERGE TEMP DATABASES
# ============================================================================

def merge_databases():
    """Merge all temp databases into main metadata.db with title lookup."""
    
    print("\n" + "="*80)
    print("MERGING TEMPORARY DATABASES")
    print("="*80)
    
    main_conn = sqlite3.connect(METADATA_DB)
    
    # Performance settings
    main_conn.execute("PRAGMA synchronous = OFF")
    main_conn.execute("PRAGMA journal_mode = WAL")
    main_conn.execute("PRAGMA cache_size = -256000")  # 256MB cache
    
    cursor = main_conn.cursor()
    
    # Drop old table
    print("\nDropping old pagelinks table...")
    cursor.execute("DROP TABLE IF EXISTS pagelinks")
    
    # Create new table
    print("Creating new pagelinks table...")
    cursor.execute("""
        CREATE TABLE pagelinks (
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            PRIMARY KEY (source_id, target_id)
        )
    """)
    main_conn.commit()
    
    # Find all temp databases
    temp_dbs = sorted(TEMP_DB_DIR.glob("worker_*.db"))
    
    print(f"\nMerging {len(temp_dbs)} temporary databases...")
    print("(Performing title->ID lookups during merge)")
    
    total_links_raw = 0
    total_links_valid = 0
    links_not_found = 0
    
    # Process each temp DB
    for i, temp_db_path in enumerate(tqdm(temp_dbs, desc="Merging")):
        temp_conn = sqlite3.connect(temp_db_path)
        temp_cursor = temp_conn.cursor()
        
        # Get all links from this worker
        temp_cursor.execute("SELECT source_id, target_title FROM links")
        
        link_batch = []
        batch_size = 10000
        
        for source_id, target_title in temp_cursor:
            total_links_raw += 1
            
            # Lookup target ID
            target_id = TITLE_TO_ID.get(target_title)
            
            if target_id is None:
                links_not_found += 1
                continue
            
            link_batch.append((source_id, target_id))
            total_links_valid += 1
            
            # Insert batch
            if len(link_batch) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO pagelinks (source_id, target_id) VALUES (?, ?)",
                    link_batch
                )
                main_conn.commit()
                link_batch = []
        
        # Final flush for this DB
        if link_batch:
            cursor.executemany(
                "INSERT OR IGNORE INTO pagelinks (source_id, target_id) VALUES (?, ?)",
                link_batch
            )
            main_conn.commit()
        
        temp_conn.close()
        
        # Delete temp DB to save space
        os.remove(temp_db_path)
    
    # Create indexes
    print("\nCreating indexes (this will take several minutes)...")
    
    print("  Index on source_id...")
    cursor.execute("CREATE INDEX idx_pagelinks_source ON pagelinks(source_id)")
    
    print("  Index on target_id...")
    cursor.execute("CREATE INDEX idx_pagelinks_target ON pagelinks(target_id)")
    
    main_conn.commit()
    
    # Final statistics
    cursor.execute("SELECT COUNT(*) FROM pagelinks")
    final_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT source_id) FROM pagelinks")
    unique_sources = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT target_id) FROM pagelinks")
    unique_targets = cursor.fetchone()[0]
    
    main_conn.close()
    
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"  Raw links extracted: {total_links_raw:,}")
    print(f"  Valid links (found in DB): {total_links_valid:,}")
    print(f"  Links not found: {links_not_found:,}")
    print(f"  Unique links (after dedup): {final_count:,}")
    print(f"  Unique sources: {unique_sources:,}")
    print(f"  Unique targets: {unique_targets:,}")
    if unique_sources > 0:
        print(f"  Avg links per source: {final_count/unique_sources:.1f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("OPTIMIZED WIKIPEDIA LINK GRAPH EXTRACTOR")
    print("="*80)
    print(f"\nWorkers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print()
    
    start_time = time.time()
    
    # Setup
    if not os.path.exists(METADATA_DB):
        print(f"ERROR: metadata.db not found at {METADATA_DB}")
        sys.exit(1)
    
    xml_files = sorted(glob(f"{XML_CHUNK_DIR}/*.xml"))
    if not xml_files:
        print(f"ERROR: No XML files found in {XML_CHUNK_DIR}")
        sys.exit(1)
    
    print(f"Found {len(xml_files)} XML chunks")
    
    # Create temp directory
    TEMP_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean old temp DBs
    for old_db in TEMP_DB_DIR.glob("worker_*.db"):
        os.remove(old_db)
    
    # Phase 1: Load title cache (happens BEFORE forking)
    load_title_cache()
    
    # Phase 2: Distribute work to workers
    print("="*80)
    print("PARSING XML AND EXTRACTING LINKS")
    print("="*80)
    print()
    
    # Distribute files evenly
    files_per_worker = len(xml_files) // NUM_WORKERS + 1
    work_chunks = []
    
    for i in range(NUM_WORKERS):
        start_idx = i * files_per_worker
        end_idx = min(start_idx + files_per_worker, len(xml_files))
        if start_idx < len(xml_files):
            work_chunks.append((i, xml_files[start_idx:end_idx]))
    
    print(f"Starting {len(work_chunks)} workers...")
    print(f"Files per worker: ~{files_per_worker}")
    print()
    
    # Use Pool to process in parallel
    with Pool(NUM_WORKERS) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(process_xml_files, work_chunks),
            total=len(work_chunks),
            desc="Workers completed"
        ):
            results.append(result)
    
    # Print worker statistics
    print("\n" + "="*80)
    print("PARSING COMPLETE")
    print("="*80)
    
    total_articles = sum(r['articles'] for r in results)
    total_links = sum(r['links_found'] for r in results)
    total_filtered = sum(r['links_filtered'] for r in results)
    
    print(f"\nArticles processed: {total_articles:,}")
    print(f"Links found: {total_links:,}")
    print(f"Links filtered: {total_filtered:,}")
    print(f"Valid link rate: {100*total_links/(total_links+total_filtered):.1f}%")
    
    # Phase 3: Merge databases
    merge_databases()
    
    # Final summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("✓ LINK GRAPH EXTRACTION COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print()
    print("Next step: Run calculate_pagerank.py")
    print()

if __name__ == "__main__":
    main()