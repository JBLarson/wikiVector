#!/usr/bin/env python3
"""
COMPREHENSIVE WIKIPEDIA EMBEDDINGS API TEST SUITE

Tests semantic search quality, performance, edge cases, and API reliability.
Provides detailed analytics on search behavior and quality metrics.
"""

import requests
import time
import json
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys

# Configuration
API_BASE = "http://localhost:5001"
TIMEOUT = 30

@dataclass
class QueryTest:
    """Represents a single test query"""
    query: str
    expected_results: List[str]
    category: str
    description: str = ""

@dataclass
class TestResult:
    """Results from a single query test"""
    query: str
    category: str
    success: bool
    found_results: List[str]
    expected_results: List[str]
    latency_ms: float
    recall: float
    top_result: str = ""
    scores: List[int] = None
    
    def __post_init__(self):
        if self.scores is None:
            self.scores = []

class WikiAPITester:
    """Main testing orchestrator"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
    def test_query(self, test: QueryTest) -> TestResult:
        """Execute a single test query"""
        url = f"{self.base_url}/api/related/{test.query}"
        
        start = time.time()
        try:
            response = requests.get(url, timeout=TIMEOUT)
            latency = (time.time() - start) * 1000
            
            if response.status_code != 200:
                return TestResult(
                    query=test.query,
                    category=test.category,
                    success=False,
                    found_results=[],
                    expected_results=test.expected_results,
                    latency_ms=latency,
                    recall=0.0
                )
            
            data = response.json()
            found_titles = [item['title'] for item in data]
            scores = [item['score'] for item in data]
            
            # Calculate recall: how many expected results were found?
            hits = sum(1 for expected in test.expected_results 
                      if any(expected.lower() in found.lower() for found in found_titles))
            recall = hits / len(test.expected_results) if test.expected_results else 1.0
            
            return TestResult(
                query=test.query,
                category=test.category,
                success=recall > 0,
                found_results=found_titles,
                expected_results=test.expected_results,
                latency_ms=latency,
                recall=recall,
                top_result=found_titles[0] if found_titles else "",
                scores=scores
            )
            
        except requests.exceptions.Timeout:
            return TestResult(
                query=test.query,
                category=test.category,
                success=False,
                found_results=[],
                expected_results=test.expected_results,
                latency_ms=(time.time() - start) * 1000,
                recall=0.0
            )
        except Exception as e:
            print(f"Error testing '{test.query}': {e}")
            return TestResult(
                query=test.query,
                category=test.category,
                success=False,
                found_results=[],
                expected_results=test.expected_results,
                latency_ms=0.0,
                recall=0.0
            )
    
    def run_test_suite(self, tests: List[QueryTest]):
        """Run all tests and collect results"""
        print("=" * 80)
        print("WIKIPEDIA EMBEDDINGS API TEST SUITE")
        print("=" * 80)
        print(f"\nRunning {len(tests)} test queries...\n")
        
        for i, test in enumerate(tests, 1):
            print(f"[{i}/{len(tests)}] Testing: {test.query}")
            result = self.test_query(test)
            self.results.append(result)
            
            # Print immediate feedback
            status = "✓" if result.success else "✗"
            print(f"  {status} Recall: {result.recall:.2f} | Latency: {result.latency_ms:.0f}ms")
            if result.top_result:
                print(f"  Top: {result.top_result}")
            print()
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print comprehensive test summary"""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        # Overall metrics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        avg_recall = statistics.mean(r.recall for r in self.results)
        avg_latency = statistics.mean(r.latency_ms for r in self.results)
        
        print(f"\nOverall Performance:")
        print(f"  Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"  Average recall: {avg_recall:.3f}")
        print(f"  Average latency: {avg_latency:.1f}ms")
        
        # Latency percentiles
        latencies = sorted(r.latency_ms for r in self.results)
        print(f"\nLatency Distribution:")
        print(f"  Min: {min(latencies):.1f}ms")
        print(f"  P50: {latencies[len(latencies)//2]:.1f}ms")
        print(f"  P95: {latencies[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  Max: {max(latencies):.1f}ms")
        
        # Category breakdown
        by_category = defaultdict(list)
        for result in self.results:
            by_category[result.category].append(result)
        
        print(f"\nPerformance by Category:")
        for category, results in sorted(by_category.items()):
            cat_passed = sum(1 for r in results if r.success)
            cat_recall = statistics.mean(r.recall for r in results)
            print(f"  {category:20s}: {cat_passed}/{len(results)} passed, "
                  f"recall={cat_recall:.3f}")
        
        # Failed queries
        failures = [r for r in self.results if not r.success]
        if failures:
            print(f"\nFailed Queries ({len(failures)}):")
            for fail in failures[:10]:  # Show first 10
                print(f"  ✗ {fail.query}")
                print(f"    Expected: {fail.expected_results}")
                print(f"    Got: {fail.found_results[:3]}")
        
        print("\n" + "=" * 80)
    
    def export_results(self, filename: str = "test_results.json"):
        """Export detailed results to JSON"""
        output = {
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results if r.success),
                'avg_recall': statistics.mean(r.recall for r in self.results),
                'avg_latency_ms': statistics.mean(r.latency_ms for r in self.results)
            },
            'tests': [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results exported to {filename}")


# ============================================================================
# TEST DEFINITIONS
# ============================================================================

def get_test_suite() -> List[QueryTest]:
    """Comprehensive test suite covering various query types"""
    
    return [
        # === EXACT TITLE MATCHES ===
        QueryTest(
            query="Python_(programming_language)",
            expected_results=["Python", "Programming", "Guido"],
            category="Exact Match",
            description="Direct Wikipedia title"
        ),
        QueryTest(
            query="Machine_learning",
            expected_results=["Machine learning", "Artificial intelligence", "Deep learning"],
            category="Exact Match"
        ),
        
        # === NATURAL LANGUAGE QUERIES ===
        QueryTest(
            query="programming language created by Guido van Rossum",
            expected_results=["Python"],
            category="Natural Language",
            description="Should find Python"
        ),
        QueryTest(
            query="ancient wonder great pyramid egypt",
            expected_results=["Great Pyramid", "Giza", "Egypt"],
            category="Natural Language"
        ),
        QueryTest(
            query="playwright who wrote Hamlet and Romeo and Juliet",
            expected_results=["Shakespeare", "William Shakespeare"],
            category="Natural Language"
        ),
        QueryTest(
            query="theory of evolution by natural selection",
            expected_results=["Darwin", "Evolution", "Natural selection"],
            category="Natural Language"
        ),
        
        # === SCIENTIFIC & TECHNICAL ===
        QueryTest(
            query="double helix structure of genetic material",
            expected_results=["DNA", "Watson", "Crick"],
            category="Science"
        ),
        QueryTest(
            query="Einstein's theory about space and time",
            expected_results=["Relativity", "Einstein", "General relativity"],
            category="Science"
        ),
        QueryTest(
            query="subatomic particle with negative charge",
            expected_results=["Electron"],
            category="Science"
        ),
        QueryTest(
            query="greenhouse gas causing climate change",
            expected_results=["Carbon dioxide", "CO2", "Climate"],
            category="Science"
        ),
        
        # === HISTORICAL EVENTS ===
        QueryTest(
            query="moon landing 1969 Armstrong",
            expected_results=["Apollo 11", "Neil Armstrong", "Moon landing"],
            category="History"
        ),
        QueryTest(
            query="French Revolution guillotine",
            expected_results=["French Revolution", "Reign of Terror"],
            category="History"
        ),
        QueryTest(
            query="World War II atomic bomb Japan",
            expected_results=["Hiroshima", "Nagasaki", "Atomic bomb"],
            category="History"
        ),
        
        # === GEOGRAPHY ===
        QueryTest(
            query="capital of France",
            expected_results=["Paris"],
            category="Geography"
        ),
        QueryTest(
            query="largest desert in the world",
            expected_results=["Sahara", "Antarctic", "Desert"],
            category="Geography"
        ),
        QueryTest(
            query="mountain range between Europe and Asia",
            expected_results=["Ural", "Mountains"],
            category="Geography"
        ),
        
        # === ARTS & CULTURE ===
        QueryTest(
            query="Mona Lisa painter Renaissance Italy",
            expected_results=["Leonardo da Vinci", "Mona Lisa"],
            category="Arts"
        ),
        QueryTest(
            query="Beatles rock band Liverpool",
            expected_results=["Beatles", "Liverpool", "John Lennon"],
            category="Arts"
        ),
        QueryTest(
            query="jazz trumpet player Louis Armstrong",
            expected_results=["Louis Armstrong", "Jazz"],
            category="Arts"
        ),
        
        # === TECHNOLOGY ===
        QueryTest(
            query="invented World Wide Web CERN",
            expected_results=["Tim Berners-Lee", "World Wide Web"],
            category="Technology"
        ),
        QueryTest(
            query="open source Linux kernel operating system",
            expected_results=["Linux", "Linus Torvalds"],
            category="Technology"
        ),
        QueryTest(
            query="iPhone creator company Cupertino",
            expected_results=["Apple", "Steve Jobs"],
            category="Technology"
        ),
        
        # === MEDICINE & HEALTH ===
        QueryTest(
            query="heart medication from foxglove plant",
            expected_results=["Digitalis", "Digoxin"],
            category="Medicine"
        ),
        QueryTest(
            query="penicillin antibiotic Fleming discovery",
            expected_results=["Penicillin", "Alexander Fleming"],
            category="Medicine"
        ),
        
        # === EDGE CASES ===
        QueryTest(
            query="xkcd web comic",
            expected_results=["xkcd", "Randall Munroe"],
            category="Edge Cases",
            description="Lowercase acronym"
        ),
        QueryTest(
            query="C++ programming language",
            expected_results=["C++", "Bjarne Stroustrup"],
            category="Edge Cases",
            description="Special characters"
        ),
        QueryTest(
            query="e=mc^2 mass energy equivalence",
            expected_results=["Mass-energy", "Einstein"],
            category="Edge Cases",
            description="Mathematical notation"
        ),
        
        # === MULTI-HOP REASONING ===
        QueryTest(
            query="author of 1984 and Animal Farm",
            expected_results=["George Orwell", "1984", "Animal Farm"],
            category="Multi-hop"
        ),
        QueryTest(
            query="company founded by Bill Gates and Paul Allen",
            expected_results=["Microsoft", "Bill Gates"],
            category="Multi-hop"
        ),
        
        # === AMBIGUOUS QUERIES ===
        QueryTest(
            query="Mercury",
            expected_results=["Mercury", "Planet", "Element"],
            category="Ambiguous",
            description="Could be planet, element, or deity"
        ),
        QueryTest(
            query="Jordan",
            expected_results=["Jordan", "Michael Jordan", "Country"],
            category="Ambiguous",
            description="Country or person"
        ),
    ]


# ============================================================================
# SPECIALIZED TESTS
# ============================================================================

class PerformanceTester:
    """Tests API performance under load"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
    
    def concurrent_load_test(self, query: str, num_requests: int = 50):
        """Test concurrent request handling"""
        print(f"\n=== Concurrent Load Test ===")
        print(f"Query: {query}")
        print(f"Requests: {num_requests}")
        
        import concurrent.futures
        
        def single_request():
            start = time.time()
            try:
                resp = requests.get(f"{self.base_url}/api/related/{query}", timeout=TIMEOUT)
                return (time.time() - start) * 1000, resp.status_code == 200
            except:
                return (time.time() - start) * 1000, False
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: single_request(), range(num_requests)))
        
        total_time = time.time() - start_time
        latencies = [r[0] for r in results]
        successes = sum(1 for r in results if r[1])
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests/sec: {num_requests/total_time:.1f}")
        print(f"  Success rate: {successes}/{num_requests} ({successes/num_requests*100:.1f}%)")
        print(f"  Latency (avg): {statistics.mean(latencies):.1f}ms")
        print(f"  Latency (p95): {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  Latency (max): {max(latencies):.1f}ms")
    
    def cache_effectiveness_test(self, query: str, iterations: int = 10):
        """Test if repeated queries are cached"""
        print(f"\n=== Cache Effectiveness Test ===")
        print(f"Query: {query}")
        print(f"Iterations: {iterations}")
        
        latencies = []
        for i in range(iterations):
            start = time.time()
            requests.get(f"{self.base_url}/api/related/{query}", timeout=TIMEOUT)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"  Run {i+1}: {latency:.1f}ms")
        
        print(f"\nFirst request: {latencies[0]:.1f}ms")
        print(f"Subsequent avg: {statistics.mean(latencies[1:]):.1f}ms")
        print(f"Speedup: {latencies[0]/statistics.mean(latencies[1:]):.2f}x")


class SemanticQualityTester:
    """Tests semantic understanding quality"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
    
    def synonym_test(self):
        """Test if synonyms return similar results"""
        print(f"\n=== Synonym Test ===")
        
        synonym_pairs = [
            ("automobile", "car"),
            ("doctor", "physician"),
            ("computer", "computing machine"),
            ("happy", "joyful"),
        ]
        
        for term1, term2 in synonym_pairs:
            results1 = self._get_results(term1)
            results2 = self._get_results(term2)
            
            # Calculate overlap
            overlap = len(set(results1) & set(results2))
            print(f"\n'{term1}' vs '{term2}':")
            print(f"  Overlap: {overlap}/{min(len(results1), len(results2))}")
            print(f"  {term1}: {results1[:3]}")
            print(f"  {term2}: {results2[:3]}")
    
    def negation_test(self):
        """Test understanding of negation"""
        print(f"\n=== Negation Test ===")
        
        queries = [
            "hot beverage",
            "cold beverage",
        ]
        
        for query in queries:
            results = self._get_results(query)
            print(f"\n'{query}': {results[:5]}")
    
    def _get_results(self, query: str) -> List[str]:
        """Helper to get result titles"""
        try:
            resp = requests.get(f"{self.base_url}/api/related/{query}", timeout=TIMEOUT)
            if resp.status_code == 200:
                return [item['title'] for item in resp.json()]
        except:
            pass
        return []


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests"""
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/api/related/test", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to API at {API_BASE}")
        print("Make sure the Flask server is running:")
        print("  python server.py")
        sys.exit(1)
    
    # Run main test suite
    tester = WikiAPITester(API_BASE)
    tests = get_test_suite()
    tester.run_test_suite(tests)
    tester.export_results("wiki_test_results.json")
    
    # Run performance tests
    perf = PerformanceTester(API_BASE)
    perf.concurrent_load_test("machine learning", num_requests=50)
    perf.cache_effectiveness_test("Python_(programming_language)", iterations=10)
    
    # Run semantic quality tests
    quality = SemanticQualityTester(API_BASE)
    quality.synonym_test()
    quality.negation_test()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
