import sqlite3
import sys

DB_PATH = "data/embeddings/metadata.db"

def add_column(cursor, table, column, col_type):
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        print(f"Added '{column}' column to '{table}' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print(f"Column '{column}' already exists in '{table}'.")
        else:
            raise e

def main():
    print(f"Connecting to database at {DB_PATH}...")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Add columns for our new signals
        add_column(cursor, "articles", "pageviews", "INTEGER DEFAULT 0")
        add_column(cursor, "articles", "backlinks", "INTEGER DEFAULT 0")
        
        # Create an index on 'article_id' for fast joins later
        print("Creating index on 'article_id' for backlink processing...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_id ON articles(article_id)")
        
        conn.commit()
        print("Database is prepared for new signals.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()