import sqlite3
import time
import sys

DB_PATH = "data/embeddings/metadata.db"

def add_column(cursor, table, column, col_type, default=None):
    """Add a column to a table if it doesn't already exist."""
    try:
        col_def = f"{column} {col_type}"
        if default is not None:
            col_def += f" DEFAULT {default}"
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
        print(f"  ✓ Added '{column}' column to '{table}' table.")
        return True
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print(f"  • Column '{column}' already exists in '{table}'.")
            return False
        else:
            raise e

def create_index(cursor, index_name, table, column):
    """Create an index if it doesn't already exist."""
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})")
    print(f"  ✓ Index '{index_name}' created or already exists.")

def normalize_titles(cursor):
    """Populate lookup_title column with normalized titles."""
    print("\nNormalizing article titles...")
    
    # Check if normalization is needed
    cursor.execute("SELECT COUNT(*) FROM articles WHERE lookup_title IS NULL")
    count_null = cursor.fetchone()[0]
    
    if count_null == 0:
        print("  • All titles already normalized. Skipping.")
        return
    
    print(f"  Found {count_null:,} articles to normalize...")
    start_time = time.time()
    
    # Fetch titles that need normalization
    cursor.execute("SELECT article_id, title FROM articles WHERE lookup_title IS NULL")
    rows = cursor.fetchall()
    
    # Prepare updates
    updates = [
        (title.lower(), article_id)
        for article_id, title in rows
        if title  # Ensure title is not None
    ]
    
    # Bulk update
    cursor.executemany("UPDATE articles SET lookup_title = ? WHERE article_id = ?", updates)
    
    elapsed = time.time() - start_time
    print(f"  ✓ Normalized {len(updates):,} articles in {elapsed:.2f}s")

def prepare_metadata_db():
    """Prepare metadata database with all required columns and indexes."""
    print(f"Starting metadata database preparation...")
    print(f"Database: {DB_PATH}\n")
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # --- Step 1: Add new columns ---
        print("Step 1: Adding required columns...")
        add_column(cursor, "articles", "lookup_title", "TEXT")
        add_column(cursor, "articles", "pageviews", "INTEGER", 0)
        add_column(cursor, "articles", "backlinks", "INTEGER", 0)
        
        # --- Step 2: Normalize titles ---
        normalize_titles(cursor)
        
        # --- Step 3: Create indexes ---
        print("\nStep 2: Creating indexes for performance...")
        create_index(cursor, "idx_article_id", "articles", "article_id")
        create_index(cursor, "idx_lookup_title", "articles", "lookup_title")
        
        # Commit all changes
        conn.commit()
        
        # --- Step 4: Show summary ---
        print("\n" + "="*60)
        print("Database preparation complete!")
        print("="*60)
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles WHERE lookup_title IS NOT NULL")
        normalized_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 0")
        pageviews_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
        backlinks_count = cursor.fetchone()[0]
        
        print(f"\nDatabase statistics:")
        print(f"  Total articles: {total_articles:,}")
        print(f"  Normalized titles: {normalized_count:,}")
        print(f"  Articles with pageviews: {pageviews_count:,}")
        print(f"  Articles with backlinks: {backlinks_count:,}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    prepare_metadata_db()