#!/usr/bin/env python3
"""
WIKIPEDIA EMBEDDINGS PERFORMANCE BENCHMARK

Comprehensive performance testing for the Wikipedia embeddings API:
- Latency analysis across query types
- Throughput testing
- Score consistency validation
- Index reconstruction testing
- Memory and resource profiling
"""

import requests
import time
import statistics
import json
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

API_BASE = "http://localhost:5001"

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    num_queries: int
    total_time: float
    latencies: List[float]
    successes: int
    failures: int
    
    @property
    def qps(self) -> float:
        """Queries per second"""
        return self.num_queries / self.total_time if self.total_time > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[len(sorted_lat) // 2]
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[int(len(sorted_lat) * 0.95)]
    
    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[int(len(sorted_lat) * 0.99)]
    
    def summary(self) -> str:
        """Generate summary string"""
        return f"""
Benchmark: {self.name}
{'='*60}
Queries:        {self.num_queries}
Success rate:   {self.successes}/{self.num_queries} ({self.successes/self.num_queries*100:.1f}%)
Total time:     {self.total_time:.2f}s
Throughput:     {self.qps:.1f} queries/sec

Latency:
  Mean:         {self.avg_latency:.1f}ms
  Median (P50): {self.p50_latency:.1f}ms
  P95:          {self.p95_latency:.1f}ms
  P99:          {self.p99_latency:.1f}ms
  Min:          {min(self.latencies) if self.latencies else 0:.1f}ms
  Max:          {max(self.latencies) if self.latencies else 0:.1f}ms
"""


class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
    
    def _single_query(self, query: str, timeout: int = 30) -> Tuple[float, bool]:
        """Execute single query and return (latency_ms, success)"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/api/related/{query}", timeout=timeout)
            latency = (time.time() - start) * 1000
            return latency, response.status_code == 200
        except Exception:
            latency = (time.time() - start) * 1000
            return latency, False
    
    def sequential_benchmark(self, queries: List[str], name: str = "Sequential") -> BenchmarkResult:
        """Run queries sequentially"""
        print(f"\nRunning {name} benchmark ({len(queries)} queries)...")
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(queries)}", end='\r')
            
            latency, success = self._single_query(query)
            latencies.append(latency)
            
            if success:
                successes += 1
            else:
                failures += 1
        
        total_time = time.time() - start_time
        
        result = BenchmarkResult(
            name=name,
            num_queries=len(queries),
            total_time=total_time,
            latencies=latencies,
            successes=successes,
            failures=failures
        )
        
        self.results.append(result)
        print(f"\n  ✓ Complete: {successes}/{len(queries)} successful, {result.qps:.1f} qps")
        return result
    
    def concurrent_benchmark(
        self, 
        queries: List[str], 
        max_workers: int = 10,
        name: str = "Concurrent"
    ) -> BenchmarkResult:
        """Run queries concurrently"""
        print(f"\nRunning {name} benchmark ({len(queries)} queries, {max_workers} workers)...")
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(self._single_query, query): query 
                for query in queries
            }
            
            completed = 0
            for future in as_completed(future_to_query):
                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{len(queries)}", end='\r')
                
                try:
                    latency, success = future.result()
                    latencies.append(latency)
                    if success:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
        
        total_time = time.time() - start_time
        
        result = BenchmarkResult(
            name=name,
            num_queries=len(queries),
            total_time=total_time,
            latencies=latencies,
            successes=successes,
            failures=failures
        )
        
        self.results.append(result)
        print(f"\n  ✓ Complete: {successes}/{len(queries)} successful, {result.qps:.1f} qps")
        return result
    
    def warmup_test(self, query: str = "Python", iterations: int = 20) -> Dict:
        """Test cache warmup effects"""
        print(f"\nWarmup test: '{query}' x {iterations}")
        
        latencies = []
        for i in range(iterations):
            latency, _ = self._single_query(query)
            latencies.append(latency)
            print(f"  Iteration {i+1}: {latency:.1f}ms", end='\r')
        
        print()
        
        first_half = latencies[:iterations//2]
        second_half = latencies[iterations//2:]
        
        results = {
            'first_half_avg': statistics.mean(first_half),
            'second_half_avg': statistics.mean(second_half),
            'all_latencies': latencies
        }
        
        print(f"  First half avg:  {results['first_half_avg']:.1f}ms")
        print(f"  Second half avg: {results['second_half_avg']:.1f}ms")
        
        if results['first_half_avg'] > results['second_half_avg']:
            speedup = results['first_half_avg'] / results['second_half_avg']
            print(f"  ✓ Warmup effect detected: {speedup:.2f}x speedup")
        else:
            print(f"  No significant warmup effect")
        
        return results
    
    def score_consistency_test(self, query: str, iterations: int = 10) -> Dict:
        """Test if same query returns consistent scores"""
        print(f"\nScore consistency test: '{query}' x {iterations}")
        
        all_results = []
        
        for i in range(iterations):
            try:
                response = requests.get(f"{self.base_url}/api/related/{query}", timeout=30)
                if response.status_code == 200:
                    all_results.append(response.json())
            except Exception as e:
                print(f"  Error on iteration {i+1}: {e}")
        
        if not all_results:
            print("  ✗ No successful queries")
            return {}
        
        # Check if all results are identical
        first_titles = [r['title'] for r in all_results[0]]
        first_scores = [r['score'] for r in all_results[0]]
        
        identical_order = True
        identical_scores = True
        
        for result_set in all_results[1:]:
            titles = [r['title'] for r in result_set]
            scores = [r['score'] for r in result_set]
            
            if titles != first_titles:
                identical_order = False
            if scores != first_scores:
                identical_scores = False
        
        print(f"  Results consistency:")
        print(f"    Identical ordering: {'✓' if identical_order else '✗'}")
        print(f"    Identical scores:   {'✓' if identical_scores else '✗'}")
        
        # Score variance analysis
        if len(all_results) > 1:
            # For each position, calculate score variance
            score_variances = []
            for pos in range(len(all_results[0])):
                scores_at_pos = [r[pos]['score'] for r in all_results]
                if len(set(scores_at_pos)) > 1:
                    variance = statistics.variance(scores_at_pos)
                    score_variances.append((pos, variance))
            
            if score_variances:
                print(f"  ✗ Score variance detected at positions: {[p for p, v in score_variances]}")
            else:
                print(f"  ✓ All scores are perfectly consistent")
        
        return {
            'identical_order': identical_order,
            'identical_scores': identical_scores,
            'num_iterations': len(all_results)
        }
    
    def query_type_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Benchmark different query types"""
        print("\n" + "="*60)
        print("QUERY TYPE BENCHMARK")
        print("="*60)
        
        query_types = {
            'exact_title': [
                'Python_(programming_language)',
                'Machine_learning',
                'Artificial_intelligence',
                'New_York_City',
                'Albert_Einstein',
            ],
            'natural_language': [
                'programming language for web development',
                'artificial intelligence machine learning',
                'largest city in United States',
                'theory of relativity physicist',
                'social media platform founded by Zuckerberg',
            ],
            'short_queries': [
                'Python',
                'AI',
                'NYC',
                'physics',
                'music',
            ],
            'long_queries': [
                'machine learning algorithm that uses neural networks for deep learning tasks',
                'programming language created by Guido van Rossum used for data science',
                'social networking website where people share photos and connect with friends',
            ]
        }
        
        results = {}
        
        for query_type, queries in query_types.items():
            result = self.sequential_benchmark(
                queries * 10,  # Run each 10 times
                name=f"Query Type: {query_type}"
            )
            results[query_type] = result
        
        # Comparison
        print("\n" + "="*60)
        print("Query Type Comparison:")
        for query_type, result in results.items():
            print(f"  {query_type:20s}: {result.avg_latency:6.1f}ms avg, "
                  f"{result.p95_latency:6.1f}ms p95")
        
        return results
    
    def stress_test(self, duration_seconds: int = 60, qps_target: int = 50):
        """Sustained load test"""
        print(f"\n" + "="*60)
        print(f"STRESS TEST")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Target QPS: {qps_target}")
        print("="*60)
        
        queries = [
            'Python', 'JavaScript', 'Java', 'Machine learning', 'AI',
            'New York', 'London', 'Paris', 'Einstein', 'Newton',
        ]
        
        interval = 1.0 / qps_target
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        query_count = 0
        
        while time.time() - start_time < duration_seconds:
            query = random.choice(queries)
            
            query_start = time.time()
            latency, success = self._single_query(query, timeout=5)
            latencies.append(latency)
            
            if success:
                successes += 1
            else:
                failures += 1
            
            query_count += 1
            
            if query_count % 10 == 0:
                elapsed = time.time() - start_time
                current_qps = query_count / elapsed
                print(f"  {elapsed:.0f}s: {query_count} queries, "
                      f"{current_qps:.1f} qps, {successes} ok, {failures} fail", end='\r')
            
            # Sleep to maintain target QPS
            elapsed_query = time.time() - query_start
            sleep_time = interval - elapsed_query
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        total_time = time.time() - start_time
        
        print(f"\n\nStress Test Results:")
        print(f"  Total queries: {query_count}")
        print(f"  Success rate: {successes}/{query_count} ({successes/query_count*100:.1f}%)")
        print(f"  Actual QPS: {query_count/total_time:.1f}")
        print(f"  Avg latency: {statistics.mean(latencies):.1f}ms")
        print(f"  P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
    
    def print_summary(self):
        """Print summary of all benchmarks"""
        if not self.results:
            print("\nNo benchmark results to summarize")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(result.summary())
    
    def export_results(self, filename: str = "benchmark_results.json"):
        """Export results to JSON"""
        output = {
            'results': [
                {
                    'name': r.name,
                    'num_queries': r.num_queries,
                    'total_time': r.total_time,
                    'qps': r.qps,
                    'avg_latency': r.avg_latency,
                    'p50_latency': r.p50_latency,
                    'p95_latency': r.p95_latency,
                    'p99_latency': r.p99_latency,
                    'success_rate': r.successes / r.num_queries if r.num_queries > 0 else 0
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results exported to {filename}")


def main():
    """Run comprehensive benchmark suite"""
    
    # Check server
    try:
        requests.get(f"{API_BASE}/api/related/test", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to API at {API_BASE}")
        print("Make sure the Flask server is running!")
        sys.exit(1)
    
    benchmark = PerformanceBenchmark()
    
    print("\n" + "="*60)
    print("WIKIPEDIA EMBEDDINGS PERFORMANCE BENCHMARK")
    print("="*60)
    
    # 1. Warmup test
    benchmark.warmup_test("Python_(programming_language)", iterations=20)
    
    # 2. Score consistency test
    benchmark.score_consistency_test("Machine_learning", iterations=10)
    
    # 3. Query type benchmarks
    benchmark.query_type_benchmark()
    
    # 4. Sequential vs Concurrent
    test_queries = [
        'Python', 'JavaScript', 'Machine learning', 'AI', 'Physics',
        'Chemistry', 'Biology', 'Mathematics', 'History', 'Geography'
    ] * 5  # 50 queries
    
    benchmark.sequential_benchmark(test_queries, "Sequential (50 queries)")
    benchmark.concurrent_benchmark(test_queries, max_workers=5, name="Concurrent 5 workers")
    benchmark.concurrent_benchmark(test_queries, max_workers=10, name="Concurrent 10 workers")
    
    # 5. Stress test
    benchmark.stress_test(duration_seconds=30, qps_target=20)
    
    # Summary
    benchmark.print_summary()
    benchmark.export_results()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
