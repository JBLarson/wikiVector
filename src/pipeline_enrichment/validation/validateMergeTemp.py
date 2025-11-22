#!/usr/bin/env python3
import sqlite3

DB_PATH = "data/metadata.db"
TEMP_DB_PATH = "data/temp_wiki.db"

print("=" * 80)
print("METADATA.DB VALIDATION - STAGE 2 DATA QUALITY")
print("=" * 80)
print()

# Check metadata.db
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("METADATA.DB STATISTICS")
print("-" * 80)

# Basic counts
cursor.execute("SELECT COUNT(*) FROM articles")
total = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 0")
with_pageviews = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
with_backlinks = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 0 AND backlinks > 0")
with_both = cursor.fetchone()[0]

print(f"Total articles: {total:,}")
print(f"With pageviews: {with_pageviews:,} ({100*with_pageviews/total:.1f}%)")
print(f"With backlinks: {with_backlinks:,} ({100*with_backlinks/total:.1f}%)")
print(f"With BOTH signals: {with_both:,} ({100*with_both/total:.1f}%)")
print()

# Pageview stats
cursor.execute("SELECT MIN(pageviews), MAX(pageviews), AVG(pageviews), SUM(pageviews) FROM articles WHERE pageviews > 0")
min_pv, max_pv, avg_pv, total_pv = cursor.fetchone()

print("PAGEVIEW STATISTICS")
print("-" * 80)
print(f"Min pageviews: {min_pv:,}")
print(f"Max pageviews: {max_pv:,}")
print(f"Avg pageviews: {avg_pv:,.1f}")
print(f"Total pageviews: {total_pv:,}")
print()

# Backlink stats
cursor.execute("SELECT MIN(backlinks), MAX(backlinks), AVG(backlinks), SUM(backlinks) FROM articles WHERE backlinks > 0")
min_bl, max_bl, avg_bl, total_bl = cursor.fetchone()

print("BACKLINK STATISTICS")
print("-" * 80)
print(f"Min backlinks: {min_bl:,}")
print(f"Max backlinks: {max_bl:,}")
print(f"Avg backlinks: {avg_bl:,.1f}")
print(f"Total backlinks: {total_bl:,}")
print()

# Top articles by backlinks
print("TOP 15 BY BACKLINKS")
print("-" * 80)
cursor.execute("SELECT title, backlinks, pageviews FROM articles WHERE backlinks > 0 ORDER BY backlinks DESC LIMIT 15")
for i, (title, bl, pv) in enumerate(cursor.fetchall(), 1):
    print(f"{i:2d}. {bl:>10,} backlinks | {pv:>10,} views | {title}")
print()

# Top articles by pageviews
print("TOP 15 BY PAGEVIEWS")
print("-" * 80)
cursor.execute("SELECT title, pageviews, backlinks FROM articles WHERE pageviews > 0 ORDER BY pageviews DESC LIMIT 15")
for i, (title, pv, bl) in enumerate(cursor.fetchall(), 1):
    print(f"{i:2d}. {pv:>10,} views | {bl:>10,} backlinks | {title}")
print()

# Distribution
cursor.execute("""
    SELECT 
        SUM(CASE WHEN backlinks = 0 AND pageviews = 0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN backlinks > 0 AND pageviews = 0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN backlinks = 0 AND pageviews > 0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN backlinks > 0 AND pageviews > 0 THEN 1 ELSE 0 END)
    FROM articles
""")
none, bl_only, pv_only, both = cursor.fetchone()

print("SIGNAL DISTRIBUTION")
print("-" * 80)
print(f"No signals: {none:,} ({100*none/total:.1f}%)")
print(f"Backlinks only: {bl_only:,} ({100*bl_only/total:.1f}%)")
print(f"Pageviews only: {pv_only:,} ({100*pv_only/total:.1f}%)")
print(f"Both signals: {both:,} ({100*both/total:.1f}%)")
print()

conn.close()

# Check temp_wiki.db for comparison
print("TEMP_WIKI.DB REFERENCE")
print("-" * 80)

temp_conn = sqlite3.connect(TEMP_DB_PATH)
temp_cursor = temp_conn.cursor()

temp_cursor.execute("SELECT COUNT(*) FROM page")
page_count = temp_cursor.fetchone()[0]

temp_cursor.execute("SELECT COUNT(*) FROM page WHERE page_namespace = 0")
namespace_0 = temp_cursor.fetchone()[0]

temp_cursor.execute("SELECT COUNT(*) FROM pagelinks")
link_count = temp_cursor.fetchone()[0]

temp_cursor.execute("SELECT COUNT(*) FROM pagelinks WHERE pl_from_namespace = 0")
namespace_0_links = temp_cursor.fetchone()[0]

print(f"Total page records: {page_count:,}")
print(f"Namespace 0 pages: {namespace_0:,}")
print(f"Total pagelink records: {link_count:,}")
print(f"Namespace 0 pagelinks: {namespace_0_links:,}")

temp_conn.close()
print()

# Quality assessment
print("=" * 80)
print("QUALITY ASSESSMENT")
print("=" * 80)

coverage = 100 * (with_backlinks + with_pageviews - with_both) / total

print(f"Overall data coverage: {coverage:.1f}%")
print()

if with_backlinks < total * 0.5:
    print("❌ FAIL: Less than 50% of articles have backlinks")
    print("   Expected: 60-80%")
    print("   Actual: {:.1f}%".format(100*with_backlinks/total))
elif with_backlinks < total * 0.6:
    print("⚠  WARN: Backlink coverage is low")
    print("   Expected: 60-80%")
    print("   Actual: {:.1f}%".format(100*with_backlinks/total))
else:
    print("✓ PASS: Backlink coverage is good ({:.1f}%)".format(100*with_backlinks/total))

if max_bl < 100000:
    print("❌ FAIL: Max backlinks is suspiciously low")
    print("   Expected: >100,000 for popular articles")
    print(f"   Actual: {max_bl:,}")
else:
    print(f"✓ PASS: Max backlinks looks reasonable ({max_bl:,})")

if with_pageviews < total * 0.4:
    print("⚠  WARN: Pageview coverage seems low")
    print("   Actual: {:.1f}%".format(100*with_pageviews/total))
else:
    print("✓ PASS: Pageview coverage is acceptable ({:.1f}%)".format(100*with_pageviews/total))

print()

if coverage >= 70:
    print("=" * 80)
    print("✓ OVERALL: STAGE 2 DATA QUALITY IS GOOD")
    print("=" * 80)
elif coverage >= 50:
    print("=" * 80)
    print("⚠ OVERALL: STAGE 2 DATA QUALITY IS ACCEPTABLE")
    print("=" * 80)
else:
    print("=" * 80)
    print("❌ OVERALL: STAGE 2 DATA QUALITY NEEDS IMPROVEMENT")
    print("=" * 80)