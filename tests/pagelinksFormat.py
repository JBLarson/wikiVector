#!/usr/bin/env python3

PAGELINKS_SQL_PATH = "data/enwiki-latest-pagelinks.sql"

print("Searching for INSERT statements in pagelinks dump...")
print()

with open(PAGELINKS_SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f, 1):
        if 'INSERT INTO' in line and 'pagelinks' in line.lower():
            print(f"Found INSERT at line {i}")
            print()
            
            # Show the INSERT line
            print(f"{i}: {line.rstrip()}")
            
            # Extract a sample tuple to see the structure
            if 'VALUES ' in line:
                values_portion = line.split('VALUES ', 1)[1]
                # Get just the first few tuples
                sample = values_portion[:2000]  # First 2000 chars
                print()
                print("Sample data:")
                print(sample)
            
            print()
            print("Full line length:", len(line))
            break
    else:
        print("No INSERT statements found!")