#!/usr/bin/env python3
import sqlite3

TEMP_DB = "data/temp_wiki.db"
PAGELINKS_SQL = "data/enwiki-latest-pagelinks.sql"
import os

print("=" * 80)
print("PAGELINKS PROCESSING VERIFICATION")
print("=" * 80)
print()

# Check SQL file
sql_size = os.path.getsize(PAGELINKS_SQL)
print(f"Pagelinks SQL file size: {sql_size / (1024**3):.2f} GB")
print()

# Check what we imported
temp_conn = sqlite3.connect(TEMP_DB)
temp_cursor = temp_conn.cursor()

temp_cursor.execute("SELECT COUNT(*) FROM pagelinks")
imported = temp_cursor.fetchone()[0]
print(f"Records imported: {imported:,}")
print()

# Estimate expected records from file
# Quick scan of file to count INSERT statements
print("Scanning SQL file to estimate expected records...")
print("(this will take a few minutes)")
print()

insert_count = 0
with open(PAGELINKS_SQL, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('INSERT INTO'):
            insert_count += 1
            if insert_count % 100 == 0:
                print(f"  Found {insert_count} INSERT statements...", end='\r')

print(f"\nTotal INSERT statements in SQL file: {insert_count:,}")
print()

# Each INSERT has many tuples - estimate from first INSERT
print("Checking first INSERT to estimate tuples per statement...")
with open(PAGELINKS_SQL, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('INSERT INTO'):
            tuple_count = line.count('),(') + 1
            print(f"First INSERT has ~{tuple_count:,} tuples")
            estimated_total = insert_count * tuple_count
            print(f"Estimated total records: {estimated_total:,}")
            print()
            break

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if imported < estimated_total * 0.9:
    print(f"⚠ WARNING: Only imported {100*imported/estimated_total:.1f}% of expected records")
    print("processPagelinks.py may have failed partway through")
else:
    print("✓ Import appears complete")

# Check specific high-profile articles
print()
print("Checking specific articles:")

test_articles = [
    ('United_States', 3434750),
    ('Python_(programming_language)', 23862),
    ('World_War_II', 32927)
]

for title, expected_id in test_articles:
    temp_cursor.execute("SELECT page_id FROM page WHERE page_title = ? AND page_namespace = 0", (title,))
    result = temp_cursor.fetchone()
    if result:
        page_id = result[0]
        temp_cursor.execute("SELECT COUNT(*) FROM pagelinks WHERE pl_target_id = ?", (page_id,))
        count = temp_cursor.fetchone()[0]
        print(f"  {title}: {count:,} backlinks")

temp_conn.close()