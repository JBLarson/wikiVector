#!/usr/bin/env python3
"""
Diagnostic Script for pagelinks.sql

This script finds the first 'INSERT INTO pagelinks' line and
prints the raw data for the first few tuples.
"""

import sys
import os
import re

# --- CONFIG ---
PAGELINKS_SQL_PATH = "data/dumps_sql/enwiki-latest-pagelinks.sql"
# ----------------

# A simple regex to find things that look like tuples
TUPLE_FINDER = re.compile(r"\((.*?)\)")

def main():
    print(f"Diagnosing {PAGELINKS_SQL_PATH}...")
    
    if not os.path.exists(PAGELINKS_SQL_PATH):
        print(f"✗ ERROR: File not found at {PAGELINKS_SQL_PATH}")
        sys.exit(1)
        
    try:
        with open(PAGELINKS_SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Use a looser check to find the line
                if 'INSERT INTO' in line and 'pagelinks' in line:
                    print(f"\n✓ Found an INSERT line at line {line_num}:")
                    print("-" * 80)
                    print(f"{line[:200]}...")
                    print("-" * 80)
                    
                    print("\nAttempting to find first 5 tuples in this line:")
                    
                    # Find all things that look like (...)
                    matches = TUPLE_FINDER.finditer(line)
                    
                    for i, match in enumerate(matches):
                        if i >= 5:
                            break
                        print(f"\nTuple {i+1} (Raw):")
                        print(match.group(0)) # Print the full tuple with parens
                        
                    if i == 0 and not match:
                        print("✗ ERROR: Found INSERT line, but no tuples like (...) were found.")
                        
                    sys.exit(0) # We only need the first line
                    
            print("✗ ERROR: Reached end of file without finding any 'INSERT INTO ... pagelinks' lines.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()