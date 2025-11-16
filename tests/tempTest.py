#!/usr/bin/env python3
import sqlite3
import os

TEMP_DB = "../data/temp_wiki.db"

if not os.path.exists(TEMP_DB):
    print(f"temp_wiki.db does NOT exist at {TEMP_DB}")
    exit(1)

conn = sqlite3.connect(TEMP_DB)
cursor = conn.cursor()

print("=" * 80)
print("TEMP_WIKI.DB CHECK")
print("=" * 80)
print()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables in temp_wiki.db: {tables}")
print()

# Check page table
if 'page' in tables:
    cursor.execute("SELECT COUNT(*) FROM page")
    page_count = cursor.fetchone()[0]
    print(f"Page table records: {page_count:,}")
    
    if page_count > 0:
        cursor.execute("SELECT page_id, page_title FROM page LIMIT 5")
        print("Sample page records:")
        for pid, title in cursor.fetchall():
            print(f"  {pid}: {title}")
    print()
else:
    print("NO PAGE TABLE FOUND")
    print()

# Check pagelinks table  
if 'pagelinks' in tables:
    cursor.execute("SELECT COUNT(*) FROM pagelinks")
    link_count = cursor.fetchone()[0]
    print(f"Pagelinks table records: {link_count:,}")
    
    if link_count > 0:
        cursor.execute("SELECT * FROM pagelinks LIMIT 5")
        print("Sample pagelink records:")
        for row in cursor.fetchall():
            print(f"  {row}")
    print()
else:
    print("NO PAGELINKS TABLE FOUND")
    print()

conn.close()

print("=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
if 'page' not in tables or 'pagelinks' not in tables:
    print("temp_wiki.db is missing required tables")
    print("processPage.py and/or processPagelinks.py did NOT run successfully")
elif page_count < 1000000:
    print(f"Page table has only {page_count:,} records (need ~6.9M)")
    print("processPage.py did NOT complete successfully")
elif link_count < 1000000:
    print(f"Pagelinks table has only {link_count:,} records (need ~500M)")
    print("processPagelinks.py did NOT complete successfully")
else:
    print("temp_wiki.db looks good - problem is with mergeTemp.py")