import sqlite3
import re
import time
import sys
import os

# --- CONFIG ---
PAGE_SQL_PATH = "data/enwiki-latest-page.sql"
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

def process_page_row(row):
    """Extract (page_id, page_namespace, page_title) from row"""
    if len(row) < 3:
        return None
    
    try:
        page_id = int(row[0])
        namespace = int(row[1])
        title = row[2].strip("'")
        
        if namespace == 0:  # Only main namespace articles
            return (page_id, namespace, title)
    except (ValueError, IndexError):
        pass
    
    return None

def create_page_table():
    """Create temporary database with page table"""
    if os.path.exists(TEMP_DB_PATH):
        print(f"Warning: {TEMP_DB_PATH} already exists. Deleting...")
        os.remove(TEMP_DB_PATH)
    
    conn = sqlite3.connect(TEMP_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE page (
            page_id INTEGER PRIMARY KEY,
            page_namespace INTEGER,
            page_title TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Created {TEMP_DB_PATH} with page table")

def process_page():
    print("Starting page processor...")
    total_start_time = time.time()
    
    conn = None
    
    try:
        # Create database
        create_page_table()
        
        conn = sqlite3.connect(TEMP_DB_PATH)
        cursor = conn.cursor()
        
        # Import page data
        print(f"Importing {PAGE_SQL_PATH}...")
        start_time = time.time()
        total_rows = 0
        
        for batch in parse_sql_inserts(PAGE_SQL_PATH, process_page_row):
            cursor.executemany("INSERT INTO page VALUES (?, ?, ?)", batch)
            total_rows += len(batch)
            print(f"  Imported {total_rows:,} articles...", end='\r')
        
        conn.commit()
        print(f"\nImported {total_rows:,} articles in {time.time() - start_time:.2f}s")
        
        # Create index
        print("Creating index on page table...")
        start_time = time.time()
        cursor.execute("CREATE INDEX idx_page_lookup ON page(page_namespace, page_title)")
        conn.commit()
        print(f"Index created in {time.time() - start_time:.2f}s")
        
        # Show stats
        cursor.execute("SELECT COUNT(*) FROM page WHERE page_namespace = 0")
        count = cursor.fetchone()[0]
        print(f"\nTotal articles in main namespace: {count:,}")
        
        print(f"\nPage processing complete in {time.time() - total_start_time:.2f}s")
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
    process_page()