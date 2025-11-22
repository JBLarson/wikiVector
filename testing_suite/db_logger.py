#!/usr/bin/env python3
"""
Test Suite Database Logger

This script manages the SQLite database for historical test results.
It handles:
- Database initialization
- Creating new "TestRun" entries
- Logging API, Benchmark, and Quality results against a run_id
"""

import sqlite3
import time
import os
import sys
import json
import subprocess
import argparse
from typing import List, Dict, Any

RESULTS_DB = "results.db"

def connect_db():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(RESULTS_DB)

def init_db():
    """Initializes the database schema if it doesn't exist."""
    print(f"Initializing database at {RESULTS_DB}...")
    conn = connect_db()
    cursor = conn.cursor()
    
    # TestRun Table: A master record for each test execution
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TestRun (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp INTEGER NOT NULL,
        git_commit_hash TEXT,
        notes TEXT
    )
    """)
    
    # ApiTestResult Table: Stores results from test_wiki_api.py
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ApiTestResult (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        category TEXT,
        query TEXT,
        success BOOLEAN,
        recall REAL,
        latency_ms REAL,
        top_result TEXT,
        FOREIGN KEY(run_id) REFERENCES TestRun(run_id)
    )
    """)
    
    # BenchmarkResult Table: Stores results from benchmark_wiki.py
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS BenchmarkResult (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        name TEXT,
        num_queries INTEGER,
        total_time REAL,
        qps REAL,
        avg_latency REAL,
        p50_latency REAL,
        p95_latency REAL,
        p99_latency REAL,
        success_rate REAL,
        FOREIGN KEY(run_id) REFERENCES TestRun(run_id)
    )
    """)
    
    # QualityResult Table: Stores results from analyze_quality.py
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS QualityResult (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        test_type TEXT,  -- e.g., 'is_a', 'clustering', 'multi_hop'
        test_name TEXT,  -- e.g., 'Python_(programming_language)', 'Programming Languages'
        success BOOLEAN,
        value REAL,      -- For scores like 'connectivity'
        details TEXT,    -- JSON blob for 'found', 'expected', etc.
        FOREIGN KEY(run_id) REFERENCES TestRun(run_id)
    )
    """)
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully.")

def get_git_commit_hash() -> str:
    """Fetches the current git commit hash."""
    try:
        hash_str = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return hash_str
    except Exception:
        return "N/A"

def create_run(notes: str) -> int:
    """Creates a new test run record and returns the run_id."""
    conn = connect_db()
    cursor = conn.cursor()
    
    timestamp = int(time.time())
    commit_hash = get_git_commit_hash()
    
    cursor.execute(
        "INSERT INTO TestRun (run_timestamp, git_commit_hash, notes) VALUES (?, ?, ?)",
        (timestamp, commit_hash, notes)
    )
    run_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    print(f"Created new test run. run_id: {run_id}, commit: {commit_hash}, notes: '{notes}'")
    return run_id

def log_api_test_results(run_id: int, results: List[Any]):
    """Logs a batch of API test results."""
    conn = connect_db()
    cursor = conn.cursor()
    
    insert_data = [
        (
            run_id,
            r.category,
            r.query,
            r.success,
            r.recall,
            r.latency_ms,
            r.top_result
        ) for r in results
    ]
    
    cursor.executemany(
        """
        INSERT INTO ApiTestResult 
        (run_id, category, query, success, recall, latency_ms, top_result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        insert_data
    )
    
    conn.commit()
    conn.close()
    print(f"✓ Logged {len(insert_data)} API test results to run_id {run_id}.")

def log_benchmark_results(run_id: int, results: List[Any]):
    """Logs a batch of benchmark results."""
    conn = connect_db()
    cursor = conn.cursor()
    
    insert_data = [
        (
            run_id,
            r.name,
            r.num_queries,
            r.total_time,
            r.qps,
            r.avg_latency,
            r.p50_latency,
            r.p95_latency,
            r.p99_latency,
            r.successes / r.num_queries if r.num_queries > 0 else 0
        ) for r in results
    ]
    
    cursor.executemany(
        """
        INSERT INTO BenchmarkResult 
        (run_id, name, num_queries, total_time, qps, avg_latency, p50_latency, p95_latency, p99_latency, success_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_data
    )
    
    conn.commit()
    conn.close()
    print(f"✓ Logged {len(insert_data)} benchmark results to run_id {run_id}.")

def log_quality_results(run_id: int, results: Dict[str, Any]):
    """Parses and logs the complex quality results dictionary."""
    conn = connect_db()
    cursor = conn.cursor()
    insert_data = []

    try:
        # 1. 'is_a' and 'part_of' tests
        for test_type in ['is_a', 'part_of']:
            for item in results.get(test_type, []):
                insert_data.append((
                    run_id, test_type, item.get('relationship', test_type), 
                    item['success'], None, json.dumps(item)
                ))
        
        # 2. 'clustering' tests
        for domain_name, data in results.get('clustering', {}).items():
            insert_data.append((
                run_id, 'clustering', domain_name,
                data['connectivity'] > 0.7,  # Simple success metric
                data['connectivity'], json.dumps(data)
            ))
            
        # 3. 'analogies' tests
        for item in results.get('analogies', []):
            insert_data.append((
                run_id, 'analogy', f"{item.get('a','?')}:{item.get('b','?')}::{item.get('c','?')}:?",
                item['success'], None, json.dumps(item)
            ))
            
        # 4. 'multi_hop' tests
        for item in results.get('multi_hop', []):
            # --- UPDATED: Make the test name more descriptive ---
            test_name = f"{item.get('start', 'N/A')} -> {item.get('target', 'N/A')}"
            insert_data.append((
                run_id, 'multi_hop', test_name,
                item['success'], item.get('hops'), json.dumps(item)
            ))
            # ----------------------------------------------------
            
        # 5. 'disambiguation' tests
        for item in results.get('disambiguation', []):
            insert_data.append((
                run_id, 'disambiguation', item['term'],
                item['success_rate'] > 0.8, # Simple success metric
                item['success_rate'], json.dumps(item)
            ))
        
        cursor.executemany(
            """
            INSERT INTO QualityResult
            (run_id, test_type, test_name, success, value, details)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            insert_data
        )
        
        conn.commit()
        print(f"✓ Logged {len(insert_data)} quality results to run_id {run_id}.")

    except Exception as e:
        print(f"✗ FAILED to log quality results: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_previous_run_id(current_run_id: int) -> int:
    """Finds the run_id that came immediately before the current one."""
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT run_id FROM TestRun WHERE run_id < ? ORDER BY run_id DESC LIMIT 1",
        (current_run_id,)
    )
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Suite DB Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 'init' command
    init_parser = subparsers.add_parser("init", help="Initialize the database schema")
    
    # 'create_run' command
    create_parser = subparsers.add_parser("create_run", help="Create a new test run")
    create_parser.add_argument("--notes", type=str, default="", help="Optional notes for this run")
    
    # 'get_previous_run_id' command
    get_prev_parser = subparsers.add_parser("get_previous_run_id", help="Get the run_id before the current one")
    get_prev_parser.add_argument("--current", type=int, required=True, help="The current run_id")

    args = parser.parse_args()
    
    if args.command == "init":
        init_db()
    elif args.command == "create_run":
        run_id = create_run(args.notes)
        print(run_id) # Print the run_id to stdout for quickstart.sh to capture
    elif args.command == "get_previous_run_id":
        prev_id = get_previous_run_id(args.current)
        if prev_id:
            print(prev_id) # Print the ID to stdout