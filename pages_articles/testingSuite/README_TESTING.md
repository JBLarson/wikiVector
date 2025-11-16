# Wikipedia Embeddings Testing & Experimentation Suite

Comprehensive tools for testing, benchmarking, and exploring your Wikipedia embeddings vector database.

## Overview

After processing ~7M Wikipedia articles into embeddings, these tools help you:
- **Test semantic search quality** across various query types
- **Benchmark performance** under different load conditions
- **Explore semantic relationships** interactively
- **Analyze embedding quality** for conceptual understanding

## Tools

### 1. API Test Suite (`test_wiki_api.py`)
Comprehensive automated testing of your Wikipedia API.

**Features:**
- 40+ predefined test queries across multiple categories
- Natural language, exact matches, multi-hop reasoning
- Performance metrics (latency, recall, success rate)
- Concurrent load testing
- Cache effectiveness analysis

**Usage:**
```bash
python test_wiki_api.py
```

**Output:**
- Console summary with pass/fail for each test
- Detailed JSON report (`wiki_test_results.json`)
- Latency distribution analysis
- Category-wise performance breakdown

**Test Categories:**
- Exact title matches
- Natural language queries
- Scientific & technical terms
- Historical events
- Geography
- Arts & culture
- Technology
- Edge cases (special characters, acronyms)
- Ambiguous queries

### 2. Interactive Explorer (`explore_wiki.py`)
Interactive CLI for exploring semantic relationships.

**Features:**
- Query any topic and see related articles
- Compare two queries side-by-side
- Explore semantic neighborhoods (multi-hop)
- Test query variations
- Analyze score distributions
- Query history tracking

**Usage:**
```bash
# Interactive mode
python explore_wiki.py

# Direct query
python explore_wiki.py "machine learning neural networks"
```

**Commands:**
```
query <text>           - Search for related articles
compare <q1> | <q2>    - Compare two queries
explore <article>      - Explore semantic neighborhood
variations <query>     - Test query variations
analyze <query>        - Analyze score distribution
history                - Show query history
quit                   - Exit
```

**Example Session:**
```
> query Python programming language

Results:
1. Python_(programming_language)
   Score: 95/100  [████████████████████████████████████████]

2. JavaScript
   Score: 87/100  [██████████████████████████████████░░░░░░]

> compare Python | Java

Overlap: 4/7 articles
Shared results:
  • Programming_language
    Q1 score: 82/100, Q2 score: 85/100

> explore Python_(programming_language)

Level 1:
Python_(programming_language) →
  • Guido_van_Rossum (92)
  • Programming_language (88)
  • Object-oriented_programming (84)
```

### 3. Performance Benchmark (`benchmark_wiki.py`)
Stress testing and performance analysis.

**Features:**
- Sequential vs concurrent request testing
- Latency percentile analysis (P50, P95, P99)
- Warmup effect detection
- Score consistency validation
- Query type comparison
- Sustained load (stress) testing

**Usage:**
```bash
python benchmark_wiki.py
```

**Tests Performed:**
1. **Warmup Test** - Detects caching effects
2. **Score Consistency** - Validates deterministic results
3. **Query Type Benchmark** - Compares different query patterns
4. **Concurrent Load** - Tests 5 and 10 worker scenarios
5. **Stress Test** - 30-second sustained load at 20 QPS

**Output:**
```
Benchmark: Concurrent 10 workers
============================================================
Queries:        50
Success rate:   50/50 (100.0%)
Total time:     2.34s
Throughput:     21.4 queries/sec

Latency:
  Mean:         145.2ms
  Median (P50): 142.1ms
  P95:          178.3ms
  P99:          195.7ms
  Min:          98.4ms
  Max:          203.1ms
```

### 4. Semantic Quality Analyzer (`analyze_quality.py`)
Deep analysis of semantic understanding quality.

**Features:**
- Taxonomy relationships (IS-A, PART-OF)
- Domain clustering analysis
- Analogical reasoning tests
- Multi-hop relationship detection
- Disambiguation quality
- Conceptual coherence metrics

**Usage:**
```bash
python analyze_quality.py
```

**Tests:**
- **IS-A Relationships:** "Python is a programming language"
- **PART-OF Relationships:** "Heart is part of circulatory system"
- **Domain Clustering:** Do related articles cluster together?
- **Analogies:** Paris:France :: London:?
- **Multi-hop:** Can we reach B from A in N hops?
- **Disambiguation:** "Mercury" (planet vs element vs mythology)

**Output:**
```
Domain Clustering:
  Programming Languages:  78.5% connectivity
  Renaissance Artists:    91.2% connectivity
  Physics Concepts:       65.3% connectivity

Multi-hop test: Python → Guido van Rossum
  Hop 1: 23 new articles
  Hop 2: ✓ Found 'Guido_van_Rossum'!

Disambiguation: 'Mercury'
  ✓ 'planet' → Mercury_(planet)
  ✓ 'element' → Mercury_(element)
  ✓ 'mythology' → Mercury_(mythology)
```

## Quick Start

1. **Start your Flask server:**
   ```bash
   python server.py
   ```

2. **Run basic tests:**
   ```bash
   python test_wiki_api.py
   ```

3. **Explore interactively:**
   ```bash
   python explore_wiki.py
   ```

4. **Run comprehensive analysis:**
   ```bash
   python benchmark_wiki.py
   python analyze_quality.py
   ```

## Understanding Your Results

### Latency Benchmarks
- **< 100ms:** Excellent - Near real-time
- **100-200ms:** Good - Acceptable for production
- **200-500ms:** Moderate - May need optimization
- **> 500ms:** Slow - Investigate bottlenecks

### Semantic Quality Metrics
- **Recall > 0.7:** High-quality embeddings
- **Recall 0.4-0.7:** Moderate quality
- **Recall < 0.4:** May need retraining or better queries

### Score Distributions
- Scores near 100: Very high semantic similarity
- Scores 70-90: Strong relationship
- Scores 50-70: Moderate relationship
- Scores < 50: Weak relationship

## Customization

### Adding Custom Tests

**Add to test suite** (`test_wiki_api.py`):
```python
QueryTest(
    query="your custom query",
    expected_results=["Expected_Article_1", "Expected_Article_2"],
    category="Your Category",
    description="What this tests"
)
```

**Add domain clustering** (`analyze_quality.py`):
```python
domains = {
    'Your Domain': [
        'Article_1',
        'Article_2',
        'Article_3',
    ]
}
```

### Adjusting Thresholds

Edit configuration at top of each script:
```python
API_BASE = "http://localhost:5001"  # Your API endpoint
TOP_K = 7                           # Number of results
TIMEOUT = 30                        # Request timeout
```

## Interpreting Results

### High-Quality Indicators
✓ Consistent scores across runs
✓ High domain connectivity (>70%)
✓ Good disambiguation accuracy
✓ Multi-hop paths found within 2-3 hops
✓ Analogies resolve correctly

### Warning Signs
✗ Inconsistent scores between runs
✗ Low domain connectivity (<50%)
✗ Poor disambiguation
✗ No multi-hop paths found
✗ High latency variance

## Performance Optimization Tips

Based on benchmark results:

1. **High Latency:**
   - Check `nprobe` setting in FAISS index
   - Consider index optimization (IVF+PQ)
   - Enable GPU if available

2. **Low Throughput:**
   - Increase batch size
   - Enable concurrent request handling
   - Add caching layer

3. **Poor Semantic Quality:**
   - Verify embedding model is appropriate
   - Check text preprocessing quality
   - Validate normalization settings

## Troubleshooting

**Connection Refused:**
```bash
# Make sure server is running
python server.py

# Check if port 5001 is available
lsof -i :5001
```

**Slow Queries:**
```python
# In server.py, check index settings
try:
    ivf_index = faiss.downcast_index(index.index)
    ivf_index.nprobe = 16  # Adjust this (higher = slower but more accurate)
except:
    pass
```

**Inconsistent Results:**
- Verify index has `make_direct_map` enabled
- Check if using correct similarity metric (IP vs L2)
- Ensure embeddings are normalized

## Next Steps

After running these tests:

1. **Identify bottlenecks** from benchmark results
2. **Tune scoring weights** based on quality analysis
3. **Optimize index** if latency is high
4. **Add domain-specific tests** for your use case
5. **Deploy with confidence** knowing your system's capabilities

## Files Generated

- `wiki_test_results.json` - Detailed test results
- `benchmark_results.json` - Performance metrics
- `semantic_quality_report.json` - Quality analysis

## Requirements

All tools use the same dependencies as your main system:
- requests
- Standard library (json, time, statistics, etc.)

No additional packages needed!

## Contributing

Found an edge case? Add it to the test suite!
Want to test a specific domain? Extend the quality analyzer!

---

Built to thoroughly test your 7M article Wikipedia embeddings system. 🚀
