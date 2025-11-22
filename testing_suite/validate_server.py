#!/usr/bin/env python3
"""
Quick verification of the optimized server.
Tests natural language queries to confirm they now work.
"""

import requests
import json

API_BASE = "http://localhost:5001"

print("="*80)
print("VERIFYING OPTIMIZED SERVER")
print("="*80)

# Test queries that previously failed
test_queries = [
    ("programming language created by Guido van Rossum", "Python"),
    ("capital of France", "Paris"),
    ("theory of evolution by natural selection", "Darwin"),
    ("subatomic particle with negative charge", "Electron"),
    ("moon landing 1969 Armstrong", "Apollo"),
    ("Beatles rock band Liverpool", "Beatles"),
    ("machine learning", "Machine learning"),
]

print("\nTesting natural language queries...")
print("-" * 80)

passed = 0
failed = 0

for query, expected_keyword in test_queries:
    try:
        response = requests.get(f"{API_BASE}/api/related/{query}", timeout=10)
        
        if response.status_code == 200:
            results = response.json()
            
            if results:
                # Check if expected keyword appears in any result
                found = any(expected_keyword.lower() in r['title'].lower() for r in results)
                
                status = "✓" if found else "~"
                top = results[0]['title']
                score = results[0]['score']
                
                print(f"{status} '{query[:50]}'")
                print(f"   → {top} (score: {score})")
                
                if found:
                    passed += 1
                else:
                    failed += 1
            else:
                print(f"✗ '{query[:50]}'")
                print(f"   → No results")
                failed += 1
        else:
            print(f"✗ '{query[:50]}'")
            print(f"   → HTTP {response.status_code}")
            failed += 1
            
    except Exception as e:
        print(f"✗ '{query[:50]}'")
        print(f"   → Error: {e}")
        failed += 1

print("\n" + "="*80)
print(f"Results: {passed} passed, {failed} failed")

if passed > 5:
    print("✓ Server optimization working! Natural language queries are functioning.")
elif passed > 2:
    print("~ Partial improvement. Some queries work, but tuning may be needed.")
else:
    print("✗ Server may not be updated or there's an issue.")

# Test health endpoint
print("\n" + "="*80)
print("Server Configuration:")
try:
    response = requests.get(f"{API_BASE}/api/health", timeout=5)
    if response.status_code == 200:
        config = response.json()
        print(json.dumps(config, indent=2))
except:
    print("Could not fetch health status")

print("\n" + "="*80)
