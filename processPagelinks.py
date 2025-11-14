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
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        buffer = ''
        
        for line in f:
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
    """Extract (pl_from, pl_namespace, pl_title, pl_from_namespace) from row"""
    if len(row) < 4:
        return None
    
    try:
        pl_from = int(row[0])
        namespace = int(row[1])
        title = row[2].strip("'")
        pl_from_namespace = int(row[3])
        
        if namespace == 0:  # Only links to main namespace
            return (pl_from, namespace, title, pl_from_namespace)
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
    
    # Check if pagelinks table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pagelinks'")
    if cursor.fetchone():
        print("Warning: pagelinks table already exists. Dropping and recreating...")
        cursor.execute("DROP TABLE pagelinks")
    
    cursor.execute("""
        CREATE TABLE pagelinks (
            pl_from INTEGER,
            pl_namespace INTEGER,
            pl_title TEXT,
            pl_from_namespace INTEGER
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Added pagelinks table to {TEMP_DB_PATH}")

def process_pagelinks():
    print("Starting pagelinks processor...")
    total_start_time = time.time()
    
    conn = None
    
    try:
        # Add pagelinks table to existing database
        add_pagelinks_table()
        
        conn = sqlite3.connect(TEMP_DB_PATH)
        cursor = conn.cursor()
        
        # Import pagelinks data
        print(f"Importing {PAGELINKS_SQL_PATH}...")
        print("This will take several hours...")
        start_time = time.time()
        total_links = 0
        
        for batch in parse_sql_inserts(PAGELINKS_SQL_PATH, process_pagelinks_row):
            cursor.executemany("INSERT INTO pagelinks VALUES (?, ?, ?, ?)", batch)
            total_links += len(batch)
            elapsed = time.time() - start_time
            rate = total_links / elapsed if elapsed > 0 else 0
            print(f"  Imported {total_links:,} links ({rate:,.0f}/sec)...", end='\r')
        
        conn.commit()
        print(f"\nImported {total_links:,} links in {time.time() - start_time:.2f}s")
        
        # Create index
        print("Creating index on pagelinks table...")
        start_time = time.time()
        cursor.execute("CREATE INDEX idx_pagelinks_target ON pagelinks(pl_namespace, pl_title)")
        conn.commit()
        print(f"Index created in {time.time() - start_time:.2f}s")
        
        # Show stats
        cursor.execute("SELECT COUNT(*) FROM pagelinks WHERE pl_namespace = 0")
        count = cursor.fetchone()[0]
        print(f"\nTotal links to main namespace: {count:,}")
        
        print(f"\nPagelinks processing complete in {time.time() - total_start_time:.2f}s")
        print(f"Output: {TEMP_DB_PATH}")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    process_pagelinks()