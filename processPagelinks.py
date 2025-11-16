#!/usr/bin/env python3
import sqlite3
import time
import sys
import os

# --- CONFIG ---
PAGELINKS_SQL_PATH = "data/enwiki-latest-pagelinks.sql"
TEMP_DB_PATH = "data/temp_wiki.db"
BATCH_SIZE = 100000
# ----------------

def reset_pagelinks_table():
    conn = sqlite3.connect(TEMP_DB_PATH)
    cursor = conn.cursor()
    
    print("Dropping existing pagelinks table...")
    cursor.execute("DROP TABLE IF EXISTS pagelinks")
    
    cursor.execute("""
        CREATE TABLE pagelinks (
            pl_from INTEGER,
            pl_from_namespace INTEGER,
            pl_target_id INTEGER
        )
    """)
    
    conn.commit()
    conn.close()
    print("Created fresh pagelinks table")

def process_pagelinks():
    print("=" * 80)
    print("PAGELINKS PROCESSOR (FIXED)")
    print("=" * 80)
    print()
    
    if not os.path.exists(PAGELINKS_SQL_PATH):
        print(f"ERROR: {PAGELINKS_SQL_PATH} not found")
        sys.exit(1)
    
    file_size = os.path.getsize(PAGELINKS_SQL_PATH)
    print(f"File: {PAGELINKS_SQL_PATH} ({file_size / (1024**3):.2f} GB)")
    print(f"Expected: ~2 billion records")
    print()
    
    reset_pagelinks_table()
    
    conn = sqlite3.connect(TEMP_DB_PATH)
    cursor = conn.cursor()
    
    total_start = time.time()
    batch = []
    total_imported = 0
    insert_statements = 0
    errors = 0
    
    print("Phase 1: Importing pagelinks...")
    print("This will take 2-4 hours - DO NOT INTERRUPT")
    print()
    
    try:
        with open(PAGELINKS_SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if not line.startswith('INSERT INTO'):
                    continue
                
                insert_statements += 1
                
                if 'VALUES ' not in line:
                    continue
                
                try:
                    values_portion = line.split('VALUES ', 1)[1].rstrip().rstrip(';')
                    
                    # Remove outer parens
                    values_portion = values_portion.strip()
                    if values_portion.startswith('('):
                        values_portion = values_portion[1:]
                    if values_portion.endswith(')'):
                        values_portion = values_portion[:-1]
                    
                    # Split by ),(
                    tuples = values_portion.split('),(')
                    
                    for tuple_str in tuples:
                        fields = tuple_str.split(',')
                        
                        if len(fields) >= 3:
                            try:
                                pl_from = int(fields[0].strip())
                                pl_from_namespace = int(fields[1].strip())
                                pl_target_id = int(fields[2].strip())
                                
                                batch.append((pl_from, pl_from_namespace, pl_target_id))
                                
                                if len(batch) >= BATCH_SIZE:
                                    cursor.executemany("INSERT INTO pagelinks VALUES (?, ?, ?)", batch)
                                    total_imported += len(batch)
                                    batch = []
                                    
                                    if total_imported % 50000000 == 0:
                                        conn.commit()
                                        elapsed = time.time() - total_start
                                        rate = total_imported / elapsed
                                        eta_seconds = (2068000000 - total_imported) / rate if rate > 0 else 0
                                        eta_hours = eta_seconds / 3600
                                        pct = 100 * total_imported / 2068000000
                                        
                                        print(f"  Progress: {pct:5.1f}% | {total_imported:,} records | {rate:,.0f}/sec | ETA: {eta_hours:.1f}h")
                            except (ValueError, IndexError) as e:
                                errors += 1
                                if errors < 10:
                                    print(f"  Parse error in tuple: {e}")
                
                except Exception as e:
                    errors += 1
                    if errors < 10:
                        print(f"  Error processing INSERT {insert_statements}: {e}")
                    if errors > 1000:
                        print(f"  Too many errors ({errors}), stopping")
                        raise
        
        # Insert remaining
        if batch:
            cursor.executemany("INSERT INTO pagelinks VALUES (?, ?, ?)", batch)
            total_imported += len(batch)
            conn.commit()
        
        elapsed = time.time() - total_start
        print(f"\n\nPhase 1 complete:")
        print(f"  INSERT statements: {insert_statements:,}")
        print(f"  Records imported: {total_imported:,}")
        print(f"  Errors: {errors:,}")
        print(f"  Time: {elapsed/60:.1f} minutes")
        
        expected = 2068000000
        pct = 100 * total_imported / expected
        
        if pct < 95:
            print(f"\n  ⚠ WARNING: Only {pct:.1f}% of expected records imported!")
        else:
            print(f"\n  ✓ Import appears complete ({pct:.1f}%)")
        
        print()
        
        # Create index
        print("Phase 2: Creating index (10-20 minutes)...")
        start = time.time()
        cursor.execute("CREATE INDEX idx_pagelinks_target ON pagelinks(pl_target_id)")
        conn.commit()
        print(f"Index created in {(time.time() - start)/60:.1f} minutes")
        print()
        
        # Stats
        cursor.execute("SELECT COUNT(*) FROM pagelinks")
        final_count = cursor.fetchone()[0]
        
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)
        print(f"Total pagelinks: {final_count:,}")
        print(f"Total time: {(time.time() - total_start)/3600:.2f} hours")
        
    except KeyboardInterrupt:
        print("\n\n!!! INTERRUPTED BY USER !!!")
        print(f"Imported {total_imported:,} records before interruption")
        conn.rollback()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    process_pagelinks()