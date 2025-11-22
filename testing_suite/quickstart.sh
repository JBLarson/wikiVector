#!/bin/bash
# Wikipedia Embeddings Testing - Quick Start Script

RESULTS_DB="results.db"

echo "=================================================="
echo "Wikipedia Embeddings Testing Suite"
echo "=================================================="
echo "Using results database: $RESULTS_DB"
echo ""

# --- 1. Initialize DB if it doesn't exist ---
if [ ! -f "$RESULTS_DB" ]; then
    echo "Database not found. Initializing..."
    python3 db_logger.py init
    echo ""
fi

# --- 2. Check if server is running ---
echo "Checking if Flask server is running..."
if ! curl -s http://localhost:5001/api/related/test > /dev/null 2>&1; then
    echo "✗ Flask server is not running!"
    echo ""
    echo "Please start it first:"
    echo "  python3 server.py"
    echo ""
    exit 1
fi
echo "✓ Server is running"
echo ""

# --- 3. Menu ---
echo "What would you like to do?"
echo ""
echo "1) Run comprehensive test suite (40+ tests, ~2 minutes)"
echo "2) Run performance benchmark (stress test, ~3 minutes)"
echo "3) Run semantic quality analysis (~5 minutes)"
echo "4) Open interactive explorer"
echo "5) Run ALL tests and generate comparison dashboard (~10 minutes)"
echo ""
read -p "Enter choice (1-5): " choice
echo ""

# --- 4. Create a new TestRun ---
read -p "Enter notes for this run (e.g., 'tuned nprobe=64') [optional]: " notes

# --- THIS IS THE FIX ---
# We pipe the output to 'tail -n 1' to get ONLY the last line (the run_id)
RUN_ID=$(python3 db_logger.py create_run --notes "$notes" | tail -n 1)
# ---------------------

if [ -z "$RUN_ID" ]; then
    echo "✗ FAILED to create a new run in the database. Exiting."
    exit 1
fi
echo ""
echo "Logging all results to run_id: $RUN_ID"
echo ""

case $choice in
    1)
        echo "Running comprehensive test suite..."
        python3 test_wiki_api.py --run_id $RUN_ID
        echo ""
        echo "✓ Test results saved to database."
        ;;
    2)
        echo "Running performance benchmarks..."
        python3 benchmark_wiki.py --run_id $RUN_ID
        echo ""
        echo "✓ Benchmark results saved to database."
        ;;
    3)
        echo "Running semantic quality analysis..."
        python3 analyze_quality.py --run_id $RUN_ID
        echo ""
        echo "✓ Quality report saved to database."
        ;;
    4)
        # This one doesn't produce reports, we can ignore the run_id
        echo "Starting interactive explorer..."
        echo "(Type 'quit' to exit)"
        echo ""
        python3 explore_wiki.py
        ;;
    5)
        echo "Running all tests (this will take ~10 minutes)..."
        echo ""
        
        echo "--- 1/4 Running test suite... ---"
        python3 test_wiki_api.py --run_id $RUN_ID
        
        echo ""
        echo "--- 2/4 Running benchmarks... ---"
        python3 benchmark_wiki.py --run_id $RUN_ID
        
        echo ""
        echo "--- 3/4 Running quality analysis... ---"
        python3 analyze_quality.py --run_id $RUN_ID
        
        echo ""
        echo "--- 4/4 Generating dashboard... ---"
        
        # Try to find the *previous* run for comparison
        PREV_RUN_ID=$(python3 db_logger.py get_previous_run_id --current $RUN_ID | tail -n 1)
        
        if [ -n "$PREV_RUN_ID" ]; then
            echo "Found previous run ($PREV_RUN_ID). Generating comparison report..."
            mkdir -p results # Ensure 'results' dir exists
            OUTPUT_FILE="results/compare_${PREV_RUN_ID}_vs_${RUN_ID}.html"
            python3 generate_dashboard.py --compare $PREV_RUN_ID --with $RUN_ID --output_file $OUTPUT_FILE
        else
            echo "No previous run found. Generating single run report..."
            mkdir -p results # Ensure 'results' dir exists
            OUTPUT_FILE="results/run_${RUN_ID}.html"
            python3 generate_dashboard.py --run_id $RUN_ID --output_file $OUTPUT_FILE
        fi
        
        echo ""
        echo "=================================================="
        echo "✓ ALL TESTS COMPLETE!"
        echo "=================================================="
        echo ""
        echo "All results saved to database (run_id: $RUN_ID)"
        echo "Dashboard generated: $OUTPUT_FILE"
        echo ""
        ;;
    *)
        echo "Invalid choice. The created run_id $RUN_ID will be empty."
        exit 1
        ;;
esac

echo ""
echo "Done!"