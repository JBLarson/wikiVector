#!/usr/bin/env python3
"""
Diagnostic Script for char_count Data Quality

This script validates that 'char_count' (article length) is a
stable and reliable proxy for "importance", especially when
compared to the volatile 'pageviews' data.
"""

import sqlite3
import os
import sys
import math

# --- CONFIG ---
# This path assumes you are running from the `wikiVector` root
METADATA_DB_PATH = "data/metadata.db"
# --------------

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
    print(f"{Colors.GREEN}  ✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}  ✗ {text}{Colors.END}")

def print_info(text):
    print(f"    {text}")

def format_num(n):
    return f"{n:,}"

def main():
    print_header(f"Analyzing char_count Data Quality in metadata.db")
    
    if not os.path.exists(METADATA_DB_PATH):
        print_error(f"Database not found at: {METADATA_DB_PATH}")
        sys.exit(1)
        
    conn = None
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        
        print_info(f"Connected to {METADATA_DB_PATH}")
        
        # --- 1. Schema Check ---
        print_header("Checking Database Schema")
        cursor.execute("PRAGMA table_info(articles)")
        columns = {row[1] for row in cursor.fetchall()}
        
        if 'char_count' in columns:
            print_success("Found 'char_count' column.")
        else:
            print_error("'char_count' column is missing!")
            print_info("This is unexpected. The Stage 1 script should have created this.")
            return

        # --- 2. Coverage & Stats ---
        print_header("Validating Data Coverage")
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles WHERE char_count > 0")
        with_char_count = cursor.fetchone()[0]
        
        coverage_pct = (with_char_count / total_articles) * 100
        print_success(f"{format_num(with_char_count)} / {format_num(total_articles)} articles ({coverage_pct:.1f}%) have char_count data.")

        if coverage_pct < 99:
            print_warning("Coverage is less than 99%. This might be fine, but is unexpected.")
        
        # Get stats
        cursor.execute("SELECT AVG(char_count), MAX(char_count), MIN(char_count) FROM articles WHERE char_count > 0")
        avg_cc, max_cc, min_cc = cursor.fetchone()
        
        print_info("\nStatistics (for articles with > 0 chars):")
        print_info(f"  Avg Length:   {avg_cc:,.1f} chars")
        print_info(f"  Max Length:   {format_num(max_cc)} chars")
        print_info(f"  Min Length:   {format_num(min_cc)} chars") # Should be 100

        # --- 3. Top 30 Articles by Length ---
        print_header("Top 30 Longest Articles (The 'Stable Importance' List)")
        print_info("This list should contain comprehensive, foundational topics.")
        print_info("-" * 70)
        
        cursor.execute("""
            SELECT title, char_count 
            FROM articles 
            WHERE char_count > 0 
            ORDER BY char_count DESC 
            LIMIT 30
        """)
        top_articles = cursor.fetchall()
        for i, (title, char_count) in enumerate(top_articles, 1):
            print_info(f"  {i:2d}. {char_count:>12,} - {title}")
            
        # --- 4. Benchmark Article Check ---
        print_header("Benchmark Article Rank Check")
        
        # Get ranks for our benchmark articles
        benchmark_titles = (
            'United_States', 'World_War_II', 'Ed_Gein', 'Mabel_McKay'
        )
        
        print_info("Finding ranks for stable vs. spiky articles:")
        print_info("-" * 70)
        
        for title in benchmark_titles:
            cursor.execute("""
                WITH RankedArticles AS (
                    SELECT title, char_count,
                           ROW_NUMBER() OVER (ORDER BY char_count DESC) as rank
                    FROM articles
                )
                SELECT rank, char_count FROM RankedArticles WHERE title = ?
            """, (title,))
            
            result = cursor.fetchone()
            if result:
                rank, char_count = result
                print_info(f"  {title:<20} | Rank: {format_num(rank):<10} | Length: {format_num(char_count)}")
            else:
                print_info(f"  {title:<20} | Not Found")
                
        # --- 5. Diagnosis ---
        print_header("Diagnosis")
        
        try:
            # Get the rank for 'United_States'
            cursor.execute("""
                WITH RankedArticles AS (
                    SELECT title, ROW_NUMBER() OVER (ORDER BY char_count DESC) as rank
                    FROM articles
                )
                SELECT rank FROM RankedArticles WHERE title = 'United_States'
            """)
            us_rank = cursor.fetchone()[0]
            
            # Get the rank for 'Ed_Gein'
            cursor.execute("""
                WITH RankedArticles AS (
                    SELECT title, ROW_NUMBER() OVER (ORDER BY char_count DESC) as rank
                    FROM articles
                )
                SELECT rank FROM RankedArticles WHERE title = 'Ed_Gein'
            """)
            spike_rank = cursor.fetchone()[0]
            
            if us_rank < spike_rank:
                print_success("Diagnosis: 'char_count' is a GOOD signal of importance.")
                print_info(f"  'United_States' (Rank #{format_num(us_rank)}) is ranked higher")
                print_info(f"  (longer) than the spiky article 'Ed_Gein' (Rank #{format_num(spike_rank)}).")
                print_success("We can use 'char_count' as our new importance signal.")
            else:
                print_error("Diagnosis: 'char_count' is a POOR signal.")
                print_info(f"  'Ed_Gein' (Rank #{format_num(spike_rank)}) is ranked higher")
                print_info(f"  (longer) than 'United_States' (Rank #{format_num(us_rank)}).")

        except Exception as e:
            print_error(f"Could not complete diagnosis: {e}")

    except Exception as e:
        print_error(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print_info("\nDatabase connection closed.")

if __name__ == "__main__":
    main()