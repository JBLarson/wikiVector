#!/usr/bin/env python3
"""
WIKIPEDIA METADATA UPDATE SCRIPT (Offline Char Count Recalculation + Feature Extraction)
WITH CHECKPOINTING, CRASH RECOVERY, AND AGGRESSIVE FILTERING

Processes XML chunks and calculates the true, uncapped character count along with
additional low-cost, high-value metadata features. Results are stored in temporary
SQLite databases per worker and merged into the main metadata.db at the end.

NEW: Supports resume from checkpoint. If the script crashes, just re-run it and
it will skip already-processed XML files.

NEW: Aggressive filtering to exclude list/index/glossary articles.
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
from typing import List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

XML_CHUNK_DIR = "../../data/raw/xml_chunks"
OUTPUT_DIR = "../../data"
TEMP_DB_DIR = Path("/tmp/wiki_char_count_temp")
CHECKPOINT_FILE = TEMP_DB_DIR / "completed_files.txt"

NUM_WORKERS = cpu_count()
BATCH_SIZE = 5000 
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
# FILTERING LOGIC
# ============================================================================

def should_skip_article(title: str) -> bool:
    """
    Returns True if this article should be skipped.
    Aggressive filtering for list/index/meta articles.
    """
    title_lower = title.lower()
    
    # Skip common meta-article prefixes
    skip_prefixes = [
        "list_of_", "index_of_", "glossary_of_", "outline_of_",
        "timeline_of_", "rosters_of_", "history_of_",
        "bibliography_of_", "discography_of_"
    ]
    
    for prefix in skip_prefixes:
        if title_lower.startswith(prefix):
            return True
    
    # Skip year-based competition/event lists (e.g., "2025_NSWRL_Feeder_Competitions")
    if re.match(r'^\d{4}[-_]', title):
        return True
    
    # Skip statute/legal compilations
    if "revision_act" in title_lower or "_act_" in title_lower:
        if any(word in title_lower for word in ["statute", "law", "revision"]):
            return True
    
    # Skip honours/awards lists
    if "honours" in title_lower or "awards" in title_lower:
        if any(word in title_lower for word in ["birthday", "new_year", "queens", "kings"]):
            return True
    
    # Skip "in YEAR" articles (often compilation lists)
    if re.search(r'_in_\d{4}$', title_lower):
        return True
    
    # Skip commemorative coin lists
    if "commemorative" in title_lower and "coin" in title_lower:
        return True
    
    # Skip disambiguation pages
    if "(disambiguation)" in title_lower:
        return True
    
    return False

# ============================================================================
# CORE LOGIC
# ============================================================================

WIKILINK_REGEX = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]')

def _clean_wikitext(text: str) -> str:
    """The exact same static cleaner used in the embeddings pipeline."""
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*\/>', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[http[^\]]*\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def xml_parser_worker(
    worker_id: int,
    xml_file_path: str, 
    queue: Queue,
    completed_files: set
):
    """Producer: Parses XML chunks and queues the article ID and calculated features."""
    
    if xml_file_path in completed_files:
        return
    
    namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
    
    try:
        with open(xml_file_path, 'rb') as f:
            context = ET.iterparse(f, events=('end',), tag=f"{namespace}page")
            
            for event, elem in context:
                try:
                    id_elem = elem.find(f"{namespace}id")
                    if id_elem is None: 
                        continue
                    page_id = int(id_elem.text)
                    
                    if elem.find(f"{namespace}redirect") is not None: 
                        continue
                    
                    title_elem = elem.find(f"{namespace}title")
                    if title_elem is None or not title_elem.text: 
                        continue
                    title = title_elem.text.strip().replace(" ", "_")
                    
                    ns_elem = elem.find(f"{namespace}ns")
                    ns_value = int(ns_elem.text) if ns_elem is not None else 0
                    
                    # Basic namespace filter
                    if ns_value != 0: 
                        continue
                    
                    # AGGRESSIVE FILTERING - Skip meta articles
                    if should_skip_article(title):
                        continue

                    revision = elem.find(f"{namespace}revision")
                    if revision is None: 
                        continue

                    timestamp_elem = revision.find(f"{namespace}timestamp")
                    last_revision_timestamp = timestamp_elem.text if timestamp_elem is not None else None
                    
                    text_elem = revision.find(f"{namespace}text")
                    raw_wikitext = text_elem.text
                    if raw_wikitext is None: 
                        continue
                    
                    cleaned_text = _clean_wikitext(raw_wikitext)
                    true_char_count = len(cleaned_text)
                    if true_char_count < 100: 
                        continue
                    
                    wikilink_count = len(WIKILINK_REGEX.findall(raw_wikitext))
                    
                    title_words = title.replace('_', ' ').split()
                    title_word_count = len(title_words)
                    
                    queue.put((page_id, true_char_count, wikilink_count, last_revision_timestamp, title_word_count))
                    
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

def db_writer_worker(
    worker_id: int, 
    queue: Queue, 
    temp_db_path: Path,
    total_articles: int
):
    """Consumer: Reads data from queue and writes to its own temporary SQLite DB."""
    
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS article_features (
            article_id INTEGER PRIMARY KEY,
            char_count INTEGER,
            wikilink_count INTEGER,
            last_revision_timestamp TEXT,
            title_word_count INTEGER
        )
    """)
    conn.commit()

    batch: List[Tuple] = []
    
    # Fixed progress bar - no misleading total
    pbar = tqdm(
        desc=f"Worker {worker_id}", 
        position=worker_id+1, 
        dynamic_ncols=True, 
        leave=False,
        unit=" articles"
    )
    
    while True:
        try:
            item = queue.get(timeout=QUEUE_TIMEOUT) 
            
            batch.append(item)
            pbar.update(1)
            
            if len(batch) >= BATCH_SIZE:
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO article_features 
                    (article_id, char_count, wikilink_count, last_revision_timestamp, title_word_count) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    batch
                )
                conn.commit()
                batch = []
                
        except Empty:
            if batch:
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO article_features 
                    (article_id, char_count, wikilink_count, last_revision_timestamp, title_word_count) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    batch
                )
                conn.commit()
            break
        except Exception as e:
            print(f"Worker {worker_id} Error: {e}")
            break

    pbar.close()
    conn.close()
    print(f"Worker {worker_id} finished. Saved {pbar.n:,} records to {temp_db_path.name}")

def get_existing_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Helper to get existing columns in a SQLite table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def merge_and_update_main_db(temp_dbs: List[Path], final_db_path: Path):
    """Merges data from all temporary DBs into the main metadata.db."""
    print("\n=========================================")
    print("PHASE 2: Merging temporary databases...")
    print("=========================================")
    
    main_conn = sqlite3.connect(final_db_path)
    main_cursor = main_conn.cursor()
    
    required_cols = {
        "wikilink_count": "INTEGER",
        "last_revision_timestamp": "TEXT",
        "title_word_count": "INTEGER"
    }
    existing_cols = get_existing_columns(main_conn, 'articles')
    
    for col, col_type in required_cols.items():
        if col not in existing_cols:
            print(f" - Adding missing column: {col}")
            try:
                main_cursor.execute(f"ALTER TABLE articles ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError as e:
                print(f"   WARNING: Could not add column {col}. Error: {e}")
                
    main_conn.commit()
    
    total_updates = 0
    
    for i, temp_db_path in enumerate(temp_dbs):
        if not temp_db_path.exists():
            print(f"Warning: Temp DB not found: {temp_db_path}")
            continue

        temp_conn = sqlite3.connect(temp_db_path)
        temp_cursor = temp_conn.cursor()
        
        temp_cursor.execute("SELECT article_id, char_count, wikilink_count, last_revision_timestamp, title_word_count FROM article_features")
        updates = temp_cursor.fetchall()
        
        if updates:
            print(f" - Merging {len(updates):,} records from temp DB {i+1}/{len(temp_dbs)}")
            
            updates_formatted = [
                (u[1], u[2], u[3], u[4], u[0]) for u in updates
            ]
            
            main_cursor.executemany(
                """
                UPDATE articles 
                SET 
                    char_count = ?, 
                    wikilink_count = ?, 
                    last_revision_timestamp = ?, 
                    title_word_count = ?
                WHERE article_id = ?
                """,
                updates_formatted
            )
            total_updates += len(updates)
            
        temp_conn.close()
        os.remove(temp_db_path)
        
    main_conn.commit()
    main_conn.close()
    
    print(f"✓ Success! Updated {total_updates:,} article features in {final_db_path.name}")

def main():
    """Main Orchestrator"""
    
    print("=" * 80)
    print("WIKIPEDIA METADATA UPDATE: TRUE CHAR COUNT + FEATURES")
    print("WITH AGGRESSIVE FILTERING")
    print("=" * 80)

    final_db_path = Path(OUTPUT_DIR) / "metadata.db"
    if not final_db_path.exists():
        print(f"ERROR: Main database not found at {final_db_path}. Run pipeline first.")
        sys.exit(1)
        
    xml_files = sorted(glob(f"{XML_CHUNK_DIR}/*.xml"))
    if not xml_files:
        print(f"ERROR: No XML chunks found in {XML_CHUNK_DIR}. Run chunk.py first.")
        sys.exit(1)
    
    if TEMP_DB_DIR.exists():
        print(f"Cleaning temporary DB directory: {TEMP_DB_DIR}")
        for f in TEMP_DB_DIR.glob("worker_*.db"):
            os.remove(f)
    TEMP_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean up old checkpoint - we're starting fresh
    if CHECKPOINT_FILE.exists():
        print(f"Removing old checkpoint file...")
        os.remove(CHECKPOINT_FILE)
    
    completed_files = load_completed_files()
    files_to_process = [f for f in xml_files if f not in completed_files]
    
    print(f"\nCheckpoint status:")
    print(f"  Total XML files: {len(xml_files)}")
    print(f"  Already processed: {len(completed_files)}")
    print(f"  Remaining: {len(files_to_process)}")
    
    if not files_to_process:
        print("\n✓ All files already processed! Skipping to merge...")
    
    manager = Manager()
    article_queue = manager.Queue()
    temp_dbs = [TEMP_DB_DIR / f"worker_{i}.db" for i in range(NUM_WORKERS)]
    
    try:
        conn = sqlite3.connect(final_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles_estimate = cursor.fetchone()[0]
        conn.close()
        print(f"Estimated total articles: {total_articles_estimate:,}")
    except:
        total_articles_estimate = 7_000_000
    
    if files_to_process:
        print("\n=========================================")
        print(f"PHASE 1: Processing {len(files_to_process)} XML chunks with {NUM_WORKERS} workers...")
        print("=========================================")
        
        producer_results = []
        producer_pool = Pool(NUM_WORKERS)
        
        for f in files_to_process:
            result = producer_pool.apply_async(xml_parser_worker, args=(0, f, article_queue, completed_files))
            producer_results.append(result)
            
        producer_pool.close() 
        
        consumer_processes = []
        for i in range(NUM_WORKERS):
            p = Process(target=db_writer_worker, args=(i, article_queue, temp_dbs[i], total_articles_estimate))
            consumer_processes.append(p)
            p.start()

        for res in producer_results:
            res.get()

        print("\nAll XML parsing complete. Signaling consumers to finish...")
        
        for p in consumer_processes:
            p.join(timeout=QUEUE_TIMEOUT + 5) 
            if p.is_alive():
                p.terminate()
                print(f"Warning: Consumer {p} terminated forcefully.")
    
    merge_and_update_main_db(temp_dbs, final_db_path)
    
    if CHECKPOINT_FILE.exists():
        print(f"\n✓ Cleaning up checkpoint file: {CHECKPOINT_FILE}")
        os.remove(CHECKPOINT_FILE)
    
    # Show some stats
    print("\n=========================================")
    print("VALIDATION")
    print("=========================================")
    conn = sqlite3.connect(final_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE char_count > 0")
    total_with_data = cursor.fetchone()[0]
    print(f"Articles with metadata: {total_with_data:,}")
    
    print("\nTop 5 by char_count:")
    cursor.execute("SELECT title, char_count, wikilink_count FROM articles ORDER BY char_count DESC LIMIT 5")
    for title, cc, wc in cursor.fetchall():
        print(f"  {title}: {cc:,} chars, {wc:,} links")
    
    print("\nTop 5 by wikilink_count:")
    cursor.execute("SELECT title, char_count, wikilink_count FROM articles ORDER BY wikilink_count DESC LIMIT 5")
    for title, cc, wc in cursor.fetchall():
        print(f"  {title}: {cc:,} chars, {wc:,} links")
    
    conn.close()

if __name__ == "__main__":
    main()