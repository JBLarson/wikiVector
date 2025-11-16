#!/usr/bin/env python3
import sqlite3

TEMP_DB = "data/temp_wiki.db"

print("=" * 80)
print("PAGELINKS TABLE INVESTIGATION")
print("=" * 80)
print()

temp_conn = sqlite3.connect(TEMP_DB)
temp_cursor = temp_conn.cursor()

# Check what United_States page_id should have
temp_cursor.execute("SELECT page_id FROM page WHERE page_title = 'United_States' AND page_namespace = 0")
result = temp_cursor.fetchone()
if result:
    us_page_id = result[0]
    print(f"United_States page_id: {us_page_id}")
    print()
    
    # Check if ANYTHING links to it
    temp_cursor.execute("SELECT COUNT(*) FROM pagelinks WHERE pl_target_id = ?", (us_page_id,))
    total_links = temp_cursor.fetchone()[0]
    print(f"Total links TO United_States (any namespace): {total_links:,}")
    
    temp_cursor.execute("SELECT COUNT(*) FROM pagelinks WHERE pl_target_id = ? AND pl_from_namespace = 0", (us_page_id,))
    ns0_links = temp_cursor.fetchone()[0]
    print(f"Links FROM namespace 0 TO United_States: {ns0_links:,}")
    print()
    
    # Sample what DOES link to United_States
    temp_cursor.execute("""
        SELECT pl_from, pl_from_namespace, p.page_title
        FROM pagelinks pl
        LEFT JOIN page p ON pl.pl_from = p.page_id
        WHERE pl.pl_target_id = ?
        LIMIT 20
    """, (us_page_id,))
    
    print("Sample links TO United_States:")
    for pl_from, pl_ns, title in temp_cursor.fetchall():
        print(f"  from page_id={pl_from:>8} ns={pl_ns} | {title}")
    print()

# Check overall pagelinks structure
print("=" * 80)
print("PAGELINKS TABLE STRUCTURE")
print("=" * 80)
print()

temp_cursor.execute("SELECT COUNT(*) FROM pagelinks")
total = temp_cursor.fetchone()[0]
print(f"Total pagelinks records: {total:,}")

temp_cursor.execute("SELECT COUNT(DISTINCT pl_target_id) FROM pagelinks")
unique_targets = temp_cursor.fetchone()[0]
print(f"Unique target IDs: {unique_targets:,}")

temp_cursor.execute("SELECT COUNT(*) FROM pagelinks WHERE pl_from_namespace = 0")
ns0_total = temp_cursor.fetchone()[0]
print(f"Links FROM namespace 0: {ns0_total:,}")

temp_cursor.execute("SELECT COUNT(DISTINCT pl_target_id) FROM pagelinks WHERE pl_from_namespace = 0")
ns0_unique_targets = temp_cursor.fetchone()[0]
print(f"Unique targets FROM namespace 0: {ns0_unique_targets:,}")
print()

# Check namespace distribution
print("NAMESPACE DISTRIBUTION (source pages)")
print("-" * 80)
temp_cursor.execute("""
    SELECT pl_from_namespace, COUNT(*) 
    FROM pagelinks 
    GROUP BY pl_from_namespace 
    ORDER BY COUNT(*) DESC 
    LIMIT 10
""")
for ns, count in temp_cursor.fetchall():
    print(f"  Namespace {ns:>3}: {count:,} links")
print()

# What ARE the most linked pages?
print("=" * 80)
print("MOST LINKED PAGES (all namespaces)")
print("=" * 80)
temp_cursor.execute("""
    SELECT pl_target_id, COUNT(*) as cnt
    FROM pagelinks
    GROUP BY pl_target_id
    ORDER BY cnt DESC
    LIMIT 20
""")

for target_id, count in temp_cursor.fetchall():
    temp_cursor.execute("SELECT page_title, page_namespace FROM page WHERE page_id = ?", (target_id,))
    result = temp_cursor.fetchone()
    if result:
        title, ns = result
        print(f"  {count:>10,} links -> page_id={target_id:>8} ns={ns} | {title}")
    else:
        print(f"  {count:>10,} links -> page_id={target_id:>8} (NOT IN PAGE TABLE)")

temp_conn.close()