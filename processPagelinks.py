import sqlite3
import re
import time
import sys
import os

# --- CONFIG ---
PAGELINKS_SQL_PATH = "data/enwiki-latest-pagelinks.sql"
TEMP_DB_PATH = "data/temp_wiki.db"
# ----------------

def parse_sql_inserts(file_path, value_processor, batch_size=10000):
    """Parse SQL INSERT statements and yield batches of values"""
    insert_pattern = re.compile(r'INSERT INTO `\w+` VALUES (.+);', re.DOTALL)
    
    batch = []
    lines_processed = 0
    last_update = time.time()
    
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        buffer = ''
        bytes_read = 0
        
        for line in f:
            bytes_read += len(line.encode('utf-8'))
            lines_processed += 1
            
            current_time = time.time()
            if current_time - last_update >= 1.0:
                percent = (bytes_read / file_size) * 100
                print(f"  Parsing: {percent:.1f}% ({lines_processed:,} lines)", end='\r')
                last_update = current_time
            
            if line.startswith('INSERT INTO'):
                buffer = line
            elif buffer:
                buffer += line
                
            if buffer and buffer.rstrip().endswith(';'):
                match = insert_pattern.search(buffer)
                if match:
                    values_str = match.group(1)
                    rows = parse_value_tuples(values_str)
                    
                    for row in rows:
                        processed = value_processor(row)
                        if processed:
                            batch.append(processed)
                            
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                
                buffer = ''
        
        if batch:
            yield batch

def parse_value_tuples(values_str):
    """Parse comma-separated tuples from SQL VALUES clause"""
    rows = []
    current_row = []
    current_value = ''
    in_quotes = False
    escape_next = False
    paren_depth = 0
    
    for char in values_str:
        if escape_next:
            current_value += char
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            current_value += char
            continue
            
        if char == "'" and not escape_next:
            in_quotes = not in_quotes
            current_value += char
            continue
            
        if not in_quotes:
            if char == '(':
                if paren_depth > 0:
                    current_value += char
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    if current_value:
                        current_row.append(current_value.strip())
                    if current_row:
                        rows.append(current_row)
                    current_row = []
                    current_value = ''
                else:
                    current_value += char
            elif char == ',' and paren_depth == 1:
                current_row.append(current_value.strip())
                current_value = ''
            else:
                current_value += char
        else:
            current_value += char
    
    return rows

def process_pagelinks_row(row):
    """Extract (pl_from, pl_from_namespace, pl_target_id) from row"""
    if len(row) < 3:
        return None
    
    try:
        pl_from = int(row[0])
        pl_from_namespace = int(row[1])
        pl_target_id = int(row[2])
        
        # Import all links - we'll filter during JOIN
        return (pl_from, pl_from_namespace, pl_target_id)
    except (ValueError, IndexError):
        pass
    
    return None

def add_pagelinks_table():
    """Add pagelinks table to existing database"""
    if not os.path.exists(TEMP_DB_PATH):
        print(f"Error: {TEMP_DB_PATH} does not exist!")
        print("Please run processPage.py first.")
        sys.exit(1)
    
    conn = sqlite3.connect(TEMP_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pagelinks'")
    if cursor.fetchone():
        print("Warning: pagelinks table already exists. Dropping and recreating...")
        cursor.execute("DROP TABLE pagelinks")
    
    # Match the actual Wikipedia schema
    cursor.execute("""
        CREATE TABLE pagelinks (
            pl_from INTEGER,
            pl_from_namespace INTEGER,
            pl_target_id INTEGER
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Created pagelinks table in {TEMP_DB_PATH}")

def process_pagelinks():
    print("Starting pagelinks processor...")
    print(f"File: {PAGELINKS_SQL_PATH}")
    
    file_size = os.path.getsize(PAGELINKS_SQL_PATH)
    file_size_gb = file_size / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    total_start_time = time.time()
    
    conn = None
    
    try:
        add_pagelinks_table()
        
        conn = sqlite3.connect(TEMP_DB_PATH)
        cursor = conn.cursor()
        
        print(f"\nPhase 1: Parsing and importing pagelinks...")
        print("This will take 2-4 hours...")
        start_time = time.time()
        total_links = 0
        batch_count = 0
        
        for batch in parse_sql_inserts(PAGELINKS_SQL_PATH, process_pagelinks_row):
            cursor.executemany("INSERT INTO pagelinks VALUES (?, ?, ?)", batch)
            total_links += len(batch)
            batch_count += 1
            
            if batch_count % 100 == 0:
                conn.commit()
            
            current_time = time.time()
            elapsed = current_time - start_time
            rate = total_links / elapsed if elapsed > 0 else 0
            
            if total_links > 100000:
                estimated_total = 550_000_000
                eta_seconds = (estimated_total - total_links) / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600
                print(f"  Imported {total_links:,} links @ {rate:,.0f}/sec (ETA: {eta_hours:.1f}h)    ", end='\r')
            else:
                print(f"  Imported {total_links:,} links @ {rate:,.0f}/sec    ", end='\r')
        
        conn.commit()
        elapsed = time.time() - start_time
        print(f"\n\nPhase 1 complete: Imported {total_links:,} links in {elapsed/60:.1f} minutes")
        
        print("\nPhase 2: Creating index...")
        print("This will take 10-20 minutes...")
        start_time = time.time()
        # Index on target_id since we'll be joining on page_id
        cursor.execute("CREATE INDEX idx_pagelinks_target ON pagelinks(pl_target_id)")
        conn.commit()
        elapsed = time.time() - start_time
        print(f"Index created in {elapsed/60:.1f} minutes")
        
        cursor.execute("SELECT COUNT(*) FROM pagelinks")
        total_count = cursor.fetchone()[0]
        print(f"\nTotal pagelinks imported: {total_count:,}")
        
        total_elapsed = time.time() - total_start_time
        print(f"Total processing time: {total_elapsed/3600:.2f} hours")
        print(f"Output: {TEMP_DB_PATH}")

    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    process_pagelinks()