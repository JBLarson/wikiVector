#!/usr/bin/env python3
import sqlite3

DB_PATH = "../data/metadata.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("METADATA.DB VALIDATION")
print("=" * 80)
print()

# Query 1: Total articles
cursor.execute("SELECT COUNT(*) FROM articles")
total = cursor.fetchone()[0]
print(f"Total articles: {total:,}")
print()

# Query 2: Articles with backlinks
cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
with_backlinks = cursor.fetchone()[0]
print(f"Articles with backlinks > 0: {with_backlinks:,} ({100*with_backlinks/total:.1f}%)")
print()

# Query 3: Backlink stats
cursor.execute("SELECT MIN(backlinks), MAX(backlinks), AVG(backlinks) FROM articles WHERE backlinks > 0")
min_bl, max_bl, avg_bl = cursor.fetchone()
print(f"Backlink stats (for articles with backlinks > 0):")
print(f"  Min: {min_bl}")
print(f"  Max: {max_bl:,}")
print(f"  Avg: {avg_bl:.1f}")
print()

# Query 4: Top 10 by backlinks
print("Top 10 articles by backlinks:")
cursor.execute("SELECT title, backlinks FROM articles WHERE backlinks > 0 ORDER BY backlinks DESC LIMIT 10")
for title, bl in cursor.fetchall():
    print(f"  {bl:>8,} - {title}")
print()

# Query 5: Articles with pageviews
cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 0")
with_pageviews = cursor.fetchone()[0]
print(f"Articles with pageviews > 0: {with_pageviews:,} ({100*with_pageviews/total:.1f}%)")
print()

# Query 6: Both signals
cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0 AND pageviews > 0")
both = cursor.fetchone()[0]
print(f"Articles with BOTH backlinks and pageviews: {both:,} ({100*both/total:.1f}%)")
print()

# Query 7: Distribution
cursor.execute("""
    SELECT 
        SUM(CASE WHEN backlinks = 0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN backlinks BETWEEN 1 AND 10 THEN 1 ELSE 0 END),
        SUM(CASE WHEN backlinks BETWEEN 11 AND 100 THEN 1 ELSE 0 END),
        SUM(CASE WHEN backlinks BETWEEN 101 AND 1000 THEN 1 ELSE 0 END),
        SUM(CASE WHEN backlinks > 1000 THEN 1 ELSE 0 END)
    FROM articles
""")
zero, low, med, high, very_high = cursor.fetchone()
print("Backlink distribution:")
print(f"  0 backlinks: {zero:,} ({100*zero/total:.1f}%)")
print(f"  1-10: {low:,} ({100*low/total:.1f}%)")
print(f"  11-100: {med:,} ({100*med/total:.1f}%)")
print(f"  101-1000: {high:,} ({100*high/total:.1f}%)")
print(f"  >1000: {very_high:,} ({100*very_high/total:.1f}%)")
print()

conn.close()

print("=" * 80)
print("DONE")
print("=" * 80)