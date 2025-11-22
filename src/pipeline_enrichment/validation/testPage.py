#!/usr/bin/env python3
import sqlite3

TEMP_DB_PATH = "data/temp_wiki.db"
MAIN_DB_PATH = "data/metadata.db"

print("=" * 80)
print("VALIDATE PAGE DATA")
print("=" * 80)
print()

# Check temp_wiki.db
conn = sqlite3.connect(TEMP_DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM page")
total_pages = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM page WHERE page_namespace = 0")
namespace_0 = cursor.fetchone()[0]
conn.close()

print(f"temp_wiki.db:")
print(f"  Total page records: {total_pages:,}")
print(f"  Namespace 0 (main articles): {namespace_0:,}")
print()

if namespace_0 < 6000000:
    print(f"⚠ Expected ~6.9M namespace 0 articles, found {namespace_0:,}")
else:
    print(f"✓ Page data looks good")

print()
print("=" * 80)
print("NEXT STEP")
print("=" * 80)
print()
print("Run processPagelinks.py now (2-4 hours)")