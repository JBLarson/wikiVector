#!/bin/bash
# Wikipedia Embeddings Testing - Quick Start Script

echo "=================================================="
echo "Wikipedia Embeddings Testing Suite"
echo "Quick Start Guide"
echo "=================================================="
echo ""

# Check if server is running
echo "Checking if Flask server is running..."
if ! curl -s http://localhost:5001/api/related/test > /dev/null 2>&1; then
    echo "❌ Flask server is not running!"
    echo ""
    echo "Please start it first:"
    echo "  python3 api.py"
    echo ""
    exit 1
fi

echo "✓ Server is running"
echo ""

# Menu
echo "What would you like to do?"
echo ""
echo "1) Run comprehensive test suite (40+ tests, ~2 minutes)"
echo "2) Run performance benchmark (stress test, ~3 minutes)"
echo "3) Run semantic quality analysis (~5 minutes)"
echo "4) Open interactive explorer"
echo "5) Run ALL tests and generate dashboard (~10 minutes)"
echo "6) Generate dashboard from existing results"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Running comprehensive test suite..."
        python3 test_wiki_api.py
        ;;
    2)
        echo ""
        echo "Running performance benchmarks..."
        python3 benchmark_wiki.py
        ;;
    3)
        echo ""
        echo "Running semantic quality analysis..."
        python3 analyze_quality.py
        ;;
    4)
        echo ""
        echo "Starting interactive explorer..."
        echo "(Type 'quit' to exit)"
        echo ""
        python3 explore_wiki.py
        ;;
    5)
        echo ""
        echo "Running all tests (this will take ~10 minutes)..."
        echo ""
        
        echo "1/3 Running test suite..."
        python3 test_wiki_api.py
        
        echo ""
        echo "2/3 Running benchmarks..."
        python3 benchmark_wiki.py
        
        echo ""
        echo "3/3 Running quality analysis..."
        python3 analyze_quality.py
        
        echo ""
        echo "Generating dashboard..."
        python3 generate_dashboard.py
        
        echo ""
        echo "=================================================="
        echo "✓ ALL TESTS COMPLETE!"
        echo "=================================================="
        echo ""
        echo "Results:"
        echo "  - wiki_test_results.json"
        echo "  - benchmark_results.json"
        echo "  - semantic_quality_report.json"
        echo "  - wiki_dashboard.html"
        echo ""
        ;;
    6)
        echo ""
        echo "Generating dashboard from existing results..."
        python3 generate_dashboard.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done!"
