#!/bin/bash

# Define the path to your database
DB_PATH="/mnt/data-large/wikipedia/embeddings/wikipedia_metadata.db"

echo "========================================="
echo "Query 1: Get Exact Article Count"
echo "========================================="
sqlite3 $DB_PATH "SELECT COUNT(*) FROM articles;"

echo "
=========================================
Query 2: Check Text Length Distribution
=========================================
"
sqlite3 $DB_PATH "
SELECT 
    MIN(char_count) AS min_length, 
    MAX(char_count) AS max_length, 
    AVG(char_count) AS avg_length 
FROM articles;
"

echo "
=========================================
Query 3: Verify All Articles are in Main Namespace
=========================================
"
sqlite3 $DB_PATH "SELECT DISTINCT namespace FROM articles;"

echo "
=========================================
Query 4: Show a Random Sample of 5 Articles
=========================================
"
sqlite3 $DB_PATH "
SELECT title, char_count 
FROM articles 
ORDER BY RANDOM() 
LIMIT 5;
"

echo "
=========================================
Query 5: Check Processing Time Elapsed
=========================================
"
sqlite3 $DB_PATH "
SELECT 
    MIN(embedding_timestamp) AS first_article_time, 
    MAX(embedding_timestamp) AS latest_article_time,
    MAX(embedding_timestamp) - MIN(embedding_timestamp) AS seconds_elapsed
FROM articles;
"

echo "
=========================================
Test Complete
=========================================
"
