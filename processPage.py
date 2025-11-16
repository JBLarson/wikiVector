#!/usr/bin/env python3
import sqlite3
import time
import sys
import os

# --- CONFIG ---
PAGE_SQL_PATH = "data/enwiki-latest-page.sql"
TEMP_DB_PATH = "data/temp_wiki.db"
BATCH_SIZE = 50000
# ----------------

def create_page_table():
    if os.path.exists(TEMP_DB_PATH):
        print(f"Removing existing {TEMP_DB_PATH}...")
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
    print(f"Created {TEMP_DB_PATH}")

def process_page():
    print("=" * 80)
    print("PAGE PROCESSOR")
    print("=" * 80)
    print()
    
    if not os.path.exists(PAGE_SQL_PATH):
        print(f"ERROR: {PAGE_SQL_PATH} not found")
        sys.exit(1)
    
    file_size = os.path.getsize(PAGE_SQL_PATH)
    print(f"File: {PAGE_SQL_PATH} ({file_size / (1024**3):.2f} GB)")
    print()
    
    create_page_table()
    
    conn = sqlite3.connect(TEMP_DB_PATH)
    cursor = conn.cursor()
    
    total_start = time.time()
    batch = []
    total_imported = 0
    
    print("Processing SQL file...")
    print()
    
    with open(PAGE_SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('INSERT INTO'):
                continue
            
            # Extract the VALUES portion
            if 'VALUES ' not in line:
                continue
            
            values_portion = line.split('VALUES ', 1)[1].rstrip().rstrip(';')
            
            # Split by ),( to get individual tuples
            # Handle the outer parentheses
            values_portion = values_portion.strip()
            if values_portion.startswith('('):
                values_portion = values_portion[1:]
            if values_portion.endswith(')'):
                values_portion = values_portion[:-1]
            
            # Now split by ),(
            tuples = values_portion.split('),(')
            
            for tuple_str in tuples:
                # Split by comma, but we need to handle quoted strings
                fields = []
                current = ""
                in_quotes = False
                paren_depth = 0
                
                i = 0
                while i < len(tuple_str):
                    char = tuple_str[i]
                    
                    if char == "'" and (i == 0 or tuple_str[i-1] != '\\'):
                        in_quotes = not in_quotes
                        current += char
                    elif char == '(' and not in_quotes:
                        paren_depth += 1
                        current += char
                    elif char == ')' and not in_quotes:
                        paren_depth -= 1
                        current += char
                    elif char == ',' and not in_quotes and paren_depth == 0:
                        fields.append(current)
                        current = ""
                    else:
                        current += char
                    
                    i += 1
                
                if current:
                    fields.append(current)
                
                # Now parse the fields
                if len(fields) >= 3:
                    try:
                        page_id = int(fields[0].strip())
                        namespace = int(fields[1].strip())
                        title = fields[2].strip()
                        
                        # Remove quotes
                        if title.startswith("'") and title.endswith("'"):
                            title = title[1:-1]
                        
                        # Unescape
                        title = title.replace("\\'", "'").replace("\\\\", "\\")
                        
                        # Only main namespace
                        if namespace == 0:
                            batch.append((page_id, namespace, title))
                            
                            if len(batch) >= BATCH_SIZE:
                                cursor.executemany("INSERT INTO page VALUES (?, ?, ?)", batch)
                                total_imported += len(batch)
                                batch = []
                                
                                if total_imported % 500000 == 0:
                                    conn.commit()
                                    print(f"  Imported: {total_imported:,} articles", end='\r')
                    except (ValueError, IndexError) as e:
                        pass
    
    # Insert remaining
    if batch:
        cursor.executemany("INSERT INTO page VALUES (?, ?, ?)", batch)
        total_imported += len(batch)
        conn.commit()
    
    print(f"\n\nImported {total_imported:,} articles")
    print()
    
    # Create index
    print("Creating index...")
    start = time.time()
    cursor.execute("CREATE INDEX idx_page_lookup ON page(page_namespace, page_title)")
    conn.commit()
    print(f"Index created in {time.time() - start:.1f}s")
    print()
    
    # Stats
    cursor.execute("SELECT COUNT(*) FROM page")
    final_count = cursor.fetchone()[0]
    
    print("=" * 80)
    print(f"COMPLETE")
    print("=" * 80)
    print(f"Total articles: {final_count:,}")
    print(f"Total time: {(time.time() - total_start) / 60:.1f} minutes")
    
    conn.close()

if __name__ == "__main__":
    process_page()