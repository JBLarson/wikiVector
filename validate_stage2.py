import sqlite3
import os
import sys
from datetime import datetime

# --- CONFIG ---
DB_PATH = "data/metadata.db"
TEMP_DB_PATH = "data/temp_wiki.db"
# ----------------

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"  {text}")

def validate_file_existence():
    """Check that required database files exist."""
    print_header("FILE EXISTENCE VALIDATION")
    
    all_exist = True
    
    if os.path.exists(DB_PATH):
        size_mb = os.path.getsize(DB_PATH) / (1024**2)
        print_success(f"Main database found: {DB_PATH} ({size_mb:.1f} MB)")
    else:
        print_error(f"Main database missing: {DB_PATH}")
        all_exist = False
    
    if os.path.exists(TEMP_DB_PATH):
        size_mb = os.path.getsize(TEMP_DB_PATH) / (1024**2)
        print_success(f"Temp database found: {TEMP_DB_PATH} ({size_mb:.1f} MB)")
    else:
        print_warning(f"Temp database missing: {TEMP_DB_PATH}")
        print_info("This is expected if you've already run computeBacklinks and deleted it")
    
    return all_exist

def validate_main_db_schema(cursor):
    """Validate that main database has required schema."""
    print_header("MAIN DATABASE SCHEMA VALIDATION")
    
    # Check articles table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
    if not cursor.fetchone():
        print_error("Articles table does not exist!")
        return False
    print_success("Articles table exists")
    
    # Check required columns
    cursor.execute("PRAGMA table_info(articles)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    
    required_columns = {
        'article_id': 'INTEGER',
        'title': 'TEXT',
        'lookup_title': 'TEXT',
        'pageviews': 'INTEGER',
        'backlinks': 'INTEGER'
    }
    
    schema_valid = True
    for col_name, col_type in required_columns.items():
        if col_name in columns:
            print_success(f"Column '{col_name}' exists ({columns[col_name]})")
        else:
            print_error(f"Column '{col_name}' is missing!")
            schema_valid = False
    
    # Check indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='articles'")
    indexes = [row[0] for row in cursor.fetchall()]
    
    print_info(f"\nIndexes found: {len(indexes)}")
    for idx in indexes:
        print_info(f"  - {idx}")
    
    return schema_valid

def validate_temp_db_schema(cursor):
    """Validate that temp database has required tables."""
    print_header("TEMP DATABASE SCHEMA VALIDATION")
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    if 'page' in tables:
        print_success("Page table exists")
    else:
        print_error("Page table missing!")
        return False
    
    if 'pagelinks' in tables:
        print_success("Pagelinks table exists")
    else:
        print_error("Pagelinks table missing!")
        return False
    
    # Check pagelinks schema
    cursor.execute("PRAGMA table_info(pagelinks)")
    pl_columns = {row[1] for row in cursor.fetchall()}
    
    if 'pl_target_id' in pl_columns:
        print_success("Pagelinks uses new schema (pl_target_id)")
    elif 'pl_title' in pl_columns:
        print_warning("Pagelinks uses old schema (pl_title) - may need migration")
    else:
        print_error("Pagelinks schema is invalid!")
        return False
    
    return True

def validate_page_processing(main_cursor, temp_cursor):
    """Validate page dump processing."""
    print_header("PAGE DUMP VALIDATION")
    
    # Count articles in temp database
    temp_cursor.execute("SELECT COUNT(*) FROM page WHERE page_namespace = 0")
    temp_count = temp_cursor.fetchone()[0]
    print_info(f"Articles in temp DB (namespace 0): {temp_count:,}")
    
    # Count articles in main database
    main_cursor.execute("SELECT COUNT(*) FROM articles")
    main_count = main_cursor.fetchone()[0]
    print_info(f"Articles in main DB: {main_count:,}")
    
    # Sample check: verify some articles exist in both
    main_cursor.execute("SELECT article_id, title FROM articles ORDER BY RANDOM() LIMIT 10")
    sample_articles = main_cursor.fetchall()
    
    matches = 0
    for article_id, title in sample_articles:
        temp_cursor.execute("SELECT 1 FROM page WHERE page_id = ? AND page_namespace = 0", (article_id,))
        if temp_cursor.fetchone():
            matches += 1
    
    match_rate = (matches / 10) * 100
    print_info(f"\nSample validation: {matches}/10 articles matched ({match_rate:.0f}%)")
    
    if matches >= 8:
        print_success("Page processing appears valid")
        return True
    elif matches >= 5:
        print_warning("Page processing may have issues - some articles missing")
        return True
    else:
        print_error("Page processing validation failed - too many mismatches")
        return False

def validate_pagelinks_processing(main_cursor, temp_cursor):
    """Validate pagelinks dump processing."""
    print_header("PAGELINKS DUMP VALIDATION")
    
    # Count total pagelinks
    temp_cursor.execute("SELECT COUNT(*) FROM pagelinks")
    total_links = temp_cursor.fetchone()[0]
    print_info(f"Total pagelinks imported: {total_links:,}")
    
    if total_links == 0:
        print_error("No pagelinks found in temp database!")
        return False
    
    print_success(f"Pagelinks table has {total_links:,} records")
    
    # Check if links reference valid pages
    temp_cursor.execute("""
        SELECT COUNT(DISTINCT pl_target_id) 
        FROM pagelinks
    """)
    unique_targets = temp_cursor.fetchone()[0]
    print_info(f"Unique target pages: {unique_targets:,}")
    
    # Sample check: verify some links exist
    temp_cursor.execute("""
        SELECT pl_from, pl_target_id 
        FROM pagelinks 
        LIMIT 5
    """)
    sample_links = temp_cursor.fetchall()
    
    print_info("\nSample pagelinks:")
    for pl_from, pl_target in sample_links:
        temp_cursor.execute("SELECT page_title FROM page WHERE page_id = ?", (pl_target,))
        target_title = temp_cursor.fetchone()
        if target_title:
            print_info(f"  Page {pl_from} → {pl_target} ({target_title[0]})")
        else:
            print_warning(f"  Page {pl_from} → {pl_target} (target not in page table)")
    
    return True

def validate_backlinks_application(cursor):
    """Validate backlinks have been applied to main database."""
    print_header("BACKLINKS APPLICATION VALIDATION")
    
    # Count articles with backlinks
    cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
    articles_with_backlinks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_articles = cursor.fetchone()[0]
    
    coverage = (articles_with_backlinks / total_articles * 100) if total_articles > 0 else 0
    
    print_info(f"Total articles: {total_articles:,}")
    print_info(f"Articles with backlinks: {articles_with_backlinks:,} ({coverage:.1f}%)")
    
    if articles_with_backlinks == 0:
        print_error("No backlinks found! Run computeBacklinks.py")
        return False
    
    if coverage < 10:
        print_warning(f"Only {coverage:.1f}% of articles have backlinks - this seems low")
        return False
    
    print_success("Backlinks appear to be applied")
    
    # Statistics
    cursor.execute("""
        SELECT AVG(backlinks), MAX(backlinks), MIN(backlinks)
        FROM articles 
        WHERE backlinks > 0
    """)
    avg, max_bl, min_bl = cursor.fetchone()
    
    print_info(f"\nBacklink statistics:")
    print_info(f"  Average: {avg:,.1f}")
    print_info(f"  Maximum: {max_bl:,}")
    print_info(f"  Minimum: {min_bl:,}")
    
    # Show top articles by backlinks
    cursor.execute("""
        SELECT title, backlinks 
        FROM articles 
        WHERE backlinks > 0 
        ORDER BY backlinks DESC 
        LIMIT 5
    """)
    
    print_info("\nTop 5 most linked articles:")
    for title, backlinks in cursor.fetchall():
        print_info(f"  {backlinks:>8,} - {title}")
    
    return True

def validate_pageviews_application(cursor):
    """Validate pageviews have been applied to main database."""
    print_header("PAGEVIEWS APPLICATION VALIDATION")
    
    # Count articles with pageviews
    cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 0")
    articles_with_views = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_articles = cursor.fetchone()[0]
    
    coverage = (articles_with_views / total_articles * 100) if total_articles > 0 else 0
    
    print_info(f"Total articles: {total_articles:,}")
    print_info(f"Articles with pageviews: {articles_with_views:,} ({coverage:.1f}%)")
    
    if articles_with_views == 0:
        print_error("No pageviews found! Run processPageviews.py")
        return False
    
    if coverage < 1:
        print_warning(f"Only {coverage:.1f}% of articles have pageviews - processing may be incomplete")
        return False
    
    print_success("Pageviews appear to be applied")
    
    # Statistics
    cursor.execute("""
        SELECT SUM(pageviews), AVG(pageviews), MAX(pageviews), MIN(pageviews)
        FROM articles 
        WHERE pageviews > 0
    """)
    total, avg, max_pv, min_pv = cursor.fetchone()
    
    print_info(f"\nPageview statistics:")
    print_info(f"  Total views: {total:,}")
    print_info(f"  Average: {avg:,.1f}")
    print_info(f"  Maximum: {max_pv:,}")
    print_info(f"  Minimum: {min_pv:,}")
    
    # Show top articles by pageviews
    cursor.execute("""
        SELECT title, pageviews 
        FROM articles 
        WHERE pageviews > 0 
        ORDER BY pageviews DESC 
        LIMIT 5
    """)
    
    print_info("\nTop 5 most viewed articles:")
    for title, pageviews in cursor.fetchall():
        print_info(f"  {pageviews:>12,} - {title}")
    
    return True

def validate_lookup_title_normalization(cursor):
    """Validate that lookup_title normalization is complete."""
    print_header("LOOKUP_TITLE NORMALIZATION VALIDATION")
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE lookup_title IS NULL")
    null_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM articles")
    total = cursor.fetchone()[0]
    
    if null_count == 0:
        print_success("All articles have normalized lookup_title")
        return True
    else:
        coverage = ((total - null_count) / total * 100) if total > 0 else 0
        print_warning(f"{null_count:,} articles missing lookup_title ({coverage:.1f}% complete)")
        print_info("Run prepareMetadataDB.py to complete normalization")
        return False

def validate_data_quality(cursor):
    """Additional data quality checks."""
    print_header("DATA QUALITY CHECKS")
    
    issues_found = 0
    
    # Check for articles with null titles
    cursor.execute("SELECT COUNT(*) FROM articles WHERE title IS NULL")
    null_titles = cursor.fetchone()[0]
    if null_titles > 0:
        print_warning(f"{null_titles:,} articles have NULL titles")
        issues_found += 1
    else:
        print_success("No NULL titles found")
    
    # Check for duplicate article_ids
    cursor.execute("""
        SELECT article_id, COUNT(*) 
        FROM articles 
        GROUP BY article_id 
        HAVING COUNT(*) > 1
    """)
    duplicates = cursor.fetchall()
    if duplicates:
        print_warning(f"{len(duplicates)} duplicate article_ids found")
        issues_found += 1
    else:
        print_success("No duplicate article_ids")
    
    # Check for unreasonably high values
    cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 1000000")
    extreme_backlinks = cursor.fetchone()[0]
    if extreme_backlinks > 10:
        print_warning(f"{extreme_backlinks} articles have >1M backlinks (may indicate data issues)")
        issues_found += 1
    else:
        print_success("Backlink values appear reasonable")
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 100000000")
    extreme_pageviews = cursor.fetchone()[0]
    if extreme_pageviews > 10:
        print_warning(f"{extreme_pageviews} articles have >100M pageviews (may indicate data issues)")
        issues_found += 1
    else:
        print_success("Pageview values appear reasonable")
    
    return issues_found == 0

def generate_summary_report(cursor):
    """Generate final summary report."""
    print_header("SUMMARY REPORT")
    
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_articles = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
    with_backlinks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 0")
    with_pageviews = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0 AND pageviews > 0")
    with_both = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM articles WHERE lookup_title IS NOT NULL")
    normalized = cursor.fetchone()[0]
    
    print(f"Total Articles:           {total_articles:>12,}")
    print(f"With Backlinks:           {with_backlinks:>12,} ({with_backlinks/total_articles*100:>5.1f}%)")
    print(f"With Pageviews:           {with_pageviews:>12,} ({with_pageviews/total_articles*100:>5.1f}%)")
    print(f"With Both Signals:        {with_both:>12,} ({with_both/total_articles*100:>5.1f}%)")
    print(f"Normalized Titles:        {normalized:>12,} ({normalized/total_articles*100:>5.1f}%)")
    
    # Calculate completeness score
    backlink_score = (with_backlinks / total_articles) * 100
    pageview_score = (with_pageviews / total_articles) * 100
    normalized_score = (normalized / total_articles) * 100
    
    overall_score = (backlink_score + pageview_score + normalized_score) / 3
    
    print(f"\n{'Overall Completeness:':.<30} {overall_score:>5.1f}%")
    
    if overall_score >= 90:
        print_success("\nData processing is COMPLETE and valid!")
    elif overall_score >= 70:
        print_warning("\nData processing is MOSTLY complete - some issues remain")
    else:
        print_error("\nData processing is INCOMPLETE - significant work needed")

def main():
    print_header(f"WIKIPEDIA DUMP PROCESSING VALIDATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Check file existence
    if not validate_file_existence():
        print_error("\nCritical files missing - cannot continue validation")
        sys.exit(1)
    
    main_conn = None
    temp_conn = None
    
    try:
        # Connect to main database
        main_conn = sqlite3.connect(DB_PATH)
        main_cursor = main_conn.cursor()
        
        # Validate main database schema
        if not validate_main_db_schema(main_cursor):
            print_error("\nMain database schema validation failed")
            sys.exit(1)
        
        # Validate lookup_title normalization
        validate_lookup_title_normalization(main_cursor)
        
        # Validate pageviews application
        validate_pageviews_application(main_cursor)
        
        # Validate backlinks application
        validate_backlinks_application(main_cursor)
        
        # If temp database exists, do additional validation
        if os.path.exists(TEMP_DB_PATH):
            temp_conn = sqlite3.connect(TEMP_DB_PATH)
            temp_cursor = temp_conn.cursor()
            
            # Validate temp database schema
            if validate_temp_db_schema(temp_cursor):
                # Validate page processing
                validate_page_processing(main_cursor, temp_cursor)
                
                # Validate pagelinks processing
                validate_pagelinks_processing(main_cursor, temp_cursor)
        else:
            print_warning("\nSkipping temp database validation (file not found)")
        
        # Data quality checks
        validate_data_quality(main_cursor)
        
        # Generate summary
        generate_summary_report(main_cursor)
        
    except Exception as e:
        print_error(f"\nValidation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        if main_conn:
            main_conn.close()
        if temp_conn:
            temp_conn.close()

if __name__ == "__main__":
    main()