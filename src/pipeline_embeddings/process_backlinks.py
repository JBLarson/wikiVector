#!/usr/bin/env python3
"""
BACKLINK COUNTER (Inbound Links) - CORRECTED VERSION

Processes XML chunks to count how many times each article is LINKED TO (backlinks).
Uses the CORRECT normalization strategy based on diagnostic results.

Key Fix: Wikilinks use spaces + uppercase, database uses underscores + lowercase.
Normalization: [[United States]] → "united_states" → matches lookup_title
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
from typing import List, Tuple, Dict
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

XML_CHUNK_DIR = "../../data/raw/xml_chunks"
OUTPUT_DIR = "../../data"
TEMP_DB_DIR = Path("/tmp/wiki_backlinks_temp")
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
# CORE LOGIC - CORRECTED NORMALIZATION
# ============================================================================

WIKILINK_REGEX = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]')

def normalize_title(title: str) -> str:
    """
    Normalize a wikilink title to match the 'lookup_title' column format.
    
    Based on diagnostic results:
    - Wikilinks have spaces: "United States" 
    - Database lookup_title has underscores + lowercase: "united_states"
    
    Transformation:
    1. Strip whitespace
    2. Remove fragments (#...)
    3. Replace spaces with underscores
    4. Convert to lowercase
    
    Examples:
        "United States" → "united_states"
        "Python (programming language)" → "python_(programming_language)"
        "World War II#Causes" → "world_war_ii"
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
    """
    Filter out special pages that shouldn't count as backlinks.
    
    These are meta-pages, not actual articles.
    """
    if not normalized_title:
        return True
    
    # Common special page prefixes (already lowercase from normalization)
    special_prefixes = [
        'file:', 'image:', 'category:', 'template:',
        'wikipedia:', 'help:', 'portal:', 'user:',
        'talk:', 'mediawiki:', 'special:'
    ]
    
    return any(normalized_title.startswith(prefix) for prefix in special_prefixes)

def xml_parser_worker(
    worker_id: int,
    xml_file_path: str, 
    queue: Queue,
    completed_files: set
):
    """
    Producer: Parses XML chunks and extracts ALL wikilinks.
    Batches all links from an article into a single queue operation.
    """
    
    if xml_file_path in completed_files:
        return
    
    namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
    articles_processed = 0
    links_found = 0
    links_filtered = 0
    
    try:
        with open(xml_file_path, 'rb') as f:
            context = ET.iterparse(f, events=('end',), tag=f"{namespace}page")
            
            for event, elem in context:
                try:
                    id_elem = elem.find(f"{namespace}id")
                    if id_elem is None: 
                        continue
                    
                    if elem.find(f"{namespace}redirect") is not None: 
                        continue
                    
                    ns_elem = elem.find(f"{namespace}ns")
                    ns_value = int(ns_elem.text) if ns_elem is not None else 0
                    if ns_value != 0: 
                        continue
                    
                    revision = elem.find(f"{namespace}revision")
                    if revision is None: 
                        continue
                    
                    text_elem = revision.find(f"{namespace}text")
                    if text_elem is None or not text_elem.text:
                        continue
                    
                    raw_wikitext = text_elem.text
                    
                    # Extract all wikilink targets for this article
                    matches = WIKILINK_REGEX.findall(raw_wikitext)
                    
                    # CORRECTED: Normalize using the proven strategy
                    link_batch = []
                    for target_title in matches:
                        normalized = normalize_title(target_title)
                        
                        if not normalized:
                            links_filtered += 1
                            continue
                        
                        # Filter out special pages
                        if is_special_page(normalized):
                            links_filtered += 1
                            continue
                        
                        link_batch.append(normalized)
                    
                    # Single queue operation for all links from this article
                    if link_batch:
                        queue.put(link_batch)
                        links_found += len(link_batch)
                    
                    articles_processed += 1
                    
                except Exception:
                    pass
                    
                finally:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

    except Exception as e:
        print(f"ERROR: Worker {worker_id} failed on {xml_file_path}: {e}")
    finally:
        mark_file_complete(xml_file_path)
        print(f"Parser finished {os.path.basename(xml_file_path)}: {articles_processed:,} articles, {links_found:,} links ({links_filtered:,} filtered)")

def backlink_counter_worker(
    worker_id: int, 
    queue: Queue, 
    temp_db_path: Path
):
    """
    Consumer: Reads batches of links and aggregates backlink counts.
    """
    
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backlink_counts (
            lookup_title TEXT PRIMARY KEY,
            backlink_count INTEGER
        )
    """)
    conn.commit()

    backlink_map = defaultdict(int)
    items_processed = 0
    links_processed = 0
    
    pbar = tqdm(
        desc=f"Worker {worker_id}", 
        position=worker_id+1, 
        dynamic_ncols=True, 
        leave=False,
        unit=" batches"
    )
    
    while True:
        try:
            # Receive a BATCH of links (all from one article)
            link_batch = queue.get(timeout=QUEUE_TIMEOUT)
            
            # Count each link in the batch
            for target_title in link_batch:
                backlink_map[target_title] += 1
                links_processed += 1
            
            items_processed += 1
            pbar.update(1)
            
            # Flush to DB periodically
            if items_processed % 1000 == 0:
                flush_to_db(cursor, backlink_map)
                conn.commit()
                backlink_map.clear()
                
        except Empty:
            # Final flush
            if backlink_map:
                flush_to_db(cursor, backlink_map)
                conn.commit()
            break
        except Exception as e:
            print(f"Worker {worker_id} Error: {e}")
            break

    pbar.close()
    conn.close()
    print(f"Counter {worker_id} finished: {items_processed:,} batches, {links_processed:,} links")

def flush_to_db(cursor, backlink_map: Dict[str, int]):
    """Write backlink counts to database"""
    cursor.executemany(
        """
        INSERT INTO backlink_counts (lookup_title, backlink_count) 
        VALUES (?, ?)
        ON CONFLICT(lookup_title) DO UPDATE SET 
            backlink_count = backlink_count + excluded.backlink_count
        """,
        list(backlink_map.items())
    )

def merge_and_update_main_db(temp_dbs: List[Path], final_db_path: Path):
    """
    Merges backlink counts from all temp DBs into the main metadata.db.
    CORRECTED: Uses lookup_title for matching.
    """
    print("\n=========================================")
    print("PHASE 2: Merging backlink counts...")
    print("=========================================")
    
    main_conn = sqlite3.connect(final_db_path)
    main_cursor = main_conn.cursor()
    
    # Check if backlinks column exists
    main_cursor.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in main_cursor.fetchall()]
    
    if 'backlinks' not in columns:
        print(" - Adding 'backlinks' column to articles table")
        main_cursor.execute("ALTER TABLE articles ADD COLUMN backlinks INTEGER DEFAULT 0")
        main_conn.commit()
    
    # Reset all backlinks to 0
    print(" - Resetting all backlink counts to 0...")
    main_cursor.execute("UPDATE articles SET backlinks = 0")
    main_conn.commit()
    
    # Aggregate counts from all temp DBs
    print(" - Aggregating counts from temp databases...")
    global_backlink_map = defaultdict(int)
    
    for i, temp_db_path in enumerate(temp_dbs):
        if not temp_db_path.exists():
            print(f"Warning: Temp DB not found: {temp_db_path}")
            continue

        temp_conn = sqlite3.connect(temp_db_path)
        temp_cursor = temp_conn.cursor()
        
        temp_cursor.execute("SELECT lookup_title, backlink_count FROM backlink_counts")
        
        for lookup_title, count in temp_cursor.fetchall():
            global_backlink_map[lookup_title] += count
        
        temp_conn.close()
        print(f"   Loaded temp DB {i+1}/{len(temp_dbs)} ({len(global_backlink_map):,} unique titles so far)")
    
    print(f" - Total unique titles with backlinks: {len(global_backlink_map):,}")
    
    # Update the main DB - CORRECTED VERSION
    print(" - Updating main database...")
    
    updates = list(global_backlink_map.items())
    
    batch_size = 5000
    updated_count = 0
    
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i+batch_size]
        
        # CRITICAL FIX: Match on lookup_title (which is already lowercase with underscores)
        # Our normalized titles are already in the correct format
        update_data = [(count, lookup_title) for lookup_title, count in batch]
        
        main_cursor.executemany(
            """
            UPDATE articles 
            SET backlinks = ? 
            WHERE lookup_title = ?
            """,
            update_data
        )
        
        updated_count += len(batch)
        
        if updated_count % 50000 == 0:
            print(f"   Updated {updated_count:,} / {len(updates):,} articles...")
    
    main_conn.commit()
    
    # Get stats
    main_cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
    articles_with_backlinks = main_cursor.fetchone()[0]
    
    main_cursor.execute("SELECT SUM(backlinks) FROM articles")
    total_backlinks = main_cursor.fetchone()[0] or 0
    
    main_cursor.execute("SELECT AVG(backlinks), MAX(backlinks) FROM articles WHERE backlinks > 0")
    avg_bl, max_bl = main_cursor.fetchone()
    
    print(f"\n✓ Success!")
    print(f"  Articles with backlinks: {articles_with_backlinks:,}")
    print(f"  Total backlink references: {total_backlinks:,}")
    print(f"  Average backlinks: {avg_bl:.1f}")
    print(f"  Max backlinks: {max_bl:,}")
    
    main_conn.close()
    
    # Cleanup temp DBs
    for temp_db_path in temp_dbs:
        if temp_db_path.exists():
            os.remove(temp_db_path)

def main():
    """Main Orchestrator"""
    
    print("=" * 80)
    print("WIKIPEDIA BACKLINK COUNTER (CORRECTED VERSION)")
    print("=" * 80)

    final_db_path = Path(OUTPUT_DIR) / "metadata.db"
    if not final_db_path.exists():
        print(f"ERROR: Main database not found at {final_db_path}. Run pipeline first.")
        sys.exit(1)
        
    xml_files = sorted(glob(f"{XML_CHUNK_DIR}/*.xml"))
    if not xml_files:
        print(f"ERROR: No XML chunks found in {XML_CHUNK_DIR}.")
        sys.exit(1)
    
    print(f"Found {len(xml_files)} XML chunks to process")
    
    if TEMP_DB_DIR.exists():
        print(f"Cleaning temporary DB directory: {TEMP_DB_DIR}")
        for f in TEMP_DB_DIR.glob("worker_*.db"):
            os.remove(f)
    TEMP_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean up old checkpoint
    if CHECKPOINT_FILE.exists():
        print(f"Removing old checkpoint file...")
        os.remove(CHECKPOINT_FILE)
    
    completed_files = load_completed_files()
    files_to_process = [f for f in xml_files if f not in completed_files]
    
    print(f"\nCheckpoint status:")
    print(f"  Total XML files: {len(xml_files)}")
    print(f"  Remaining: {len(files_to_process)}")
    
    if not files_to_process:
        print("\n✓ All files already processed! Skipping to merge...")
    
    manager = Manager()
    link_queue = manager.Queue(maxsize=500)
    temp_dbs = [TEMP_DB_DIR / f"worker_{i}.db" for i in range(NUM_WORKERS)]
    
    if files_to_process:
        print("\n=========================================")
        print(f"PHASE 1: Extracting backlinks from {len(files_to_process)} XML chunks...")
        print("=========================================")
        print(f"Using {NUM_WORKERS} parser workers and {NUM_WORKERS} counter workers")
        print("CORRECTED: Wikilinks normalized to lowercase with underscores")
        print()
        
        producer_results = []
        producer_pool = Pool(NUM_WORKERS)
        
        for f in files_to_process:
            result = producer_pool.apply_async(xml_parser_worker, args=(0, f, link_queue, completed_files))
            producer_results.append(result)
            
        producer_pool.close() 
        
        consumer_processes = []
        for i in range(NUM_WORKERS):
            p = Process(target=backlink_counter_worker, args=(i, link_queue, temp_dbs[i]))
            consumer_processes.append(p)
            p.start()

        for res in producer_results:
            res.get()

        print("\nAll XML parsing complete. Signaling consumers to finish...")
        
        for p in consumer_processes:
            p.join(timeout=QUEUE_TIMEOUT + 5) 
            if p.is_alive():
                p.terminate()
                print(f"Warning: Consumer terminated forcefully.")
    
    merge_and_update_main_db(temp_dbs, final_db_path)
    
    if CHECKPOINT_FILE.exists():
        print(f"\n✓ Cleaning up checkpoint file: {CHECKPOINT_FILE}")
        os.remove(CHECKPOINT_FILE)
    
    # Show validation stats
    print("\n=========================================")
    print("VALIDATION")
    print("=========================================")
    conn = sqlite3.connect(final_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
    total_with_backlinks = cursor.fetchone()[0]
    print(f"Articles with backlinks: {total_with_backlinks:,}")
    
    cursor.execute("SELECT AVG(backlinks), MAX(backlinks) FROM articles WHERE backlinks > 0")
    avg_bl, max_bl = cursor.fetchone()
    print(f"Average backlinks: {avg_bl:.1f}")
    print(f"Max backlinks: {max_bl:,}")
    
    print("\nTop 20 by backlinks:")
    cursor.execute("""
        SELECT title, backlinks, char_count 
        FROM articles 
        WHERE backlinks > 0 
        ORDER BY backlinks DESC 
        LIMIT 20
    """)
    for i, (title, bl, cc) in enumerate(cursor.fetchall(), 1):
        print(f"  {i:2d}. {bl:>8,} backlinks | {cc:>8,} chars | {title}")
    
    print("\nSample important articles:")
    for title in ["United_States", "World_War_II", "Python_(programming_language)", "Education"]:
        cursor.execute("SELECT backlinks FROM articles WHERE title = ?", (title,))
        result = cursor.fetchone()
        if result:
            print(f"  {title:40s}: {result[0]:>6,} backlinks")
    
    conn.close()

if __name__ == "__main__":
    main()