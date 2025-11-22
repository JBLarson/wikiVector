#!/usr/bin/env python3
"""
WIKIPEDIA EMBEDDINGS DASHBOARD GENERATOR

Generates a beautiful HTML dashboard from test results stored in the
results.db database.

Can generate:
1. A report for a single run_id.
2. A comparison report between two run_ids.
"""

import json
import os
import sys
import argparse
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional

DB_PATH = "results.db"

# --- Data Fetching Functions ---

def connect_db():
    """Connects to the results database."""
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Please run 'python3 db_logger.py init' first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_run_metadata(conn, run_id: int) -> Dict[str, Any]:
    """Fetches the master TestRun record."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM TestRun WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        print(f"ERROR: No run found with run_id {run_id}")
        sys.exit(1)
    return dict(run)

def get_api_test_data(conn, run_id: int) -> Dict[str, Any]:
    """Fetches all API test data for a given run_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ApiTestResult WHERE run_id = ?", (run_id,))
    results = [dict(row) for row in cursor.fetchall()]
    
    if not results:
        return {'summary': {}, 'tests': []}

    total = len(results)
    passed = sum(1 for r in results if r['success'])
    avg_recall = sum(r['recall'] for r in results) / total
    avg_latency = sum(r['latency_ms'] for r in results) / total
    
    return {
        'summary': {
            'total_tests': total,
            'passed': passed,
            'avg_recall': avg_recall,
            'avg_latency_ms': avg_latency
        },
        'tests': results
    }

def get_benchmark_data(conn, run_id: int) -> Dict[str, Any]:
    """Fetches all benchmark data for a given run_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM BenchmarkResult WHERE run_id = ?", (run_id,))
    results = [dict(row) for row in cursor.fetchall()]
    return {'results': results}

def get_quality_data(conn, run_id: int) -> Dict[str, Any]:
    """Fetches and reconstructs the quality analysis data."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM QualityResult WHERE run_id = ?", (run_id,))
    results = [dict(row) for row in cursor.fetchall()]
    
    quality_report = {
        'is_a': [],
        'part_of': [],
        'clustering': {},
        'analogies': [],
        'multi_hop': [],
        'disambiguation': []
    }
    
    for row in results:
        details = json.loads(row['details'])
        test_type = row['test_type']
        
        if test_type in ['is_a', 'part_of']:
            quality_report[test_type].append(details)
        elif test_type == 'clustering':
            quality_report['clustering'][row['test_name']] = details
        elif test_type == 'analogy':
            quality_report['analogies'].append(details)
        elif test_type == 'multi_hop':
            quality_report['multi_hop'].append(details)
        elif test_type == 'disambiguation':
            quality_report['disambiguation'].append(details)
            
    return quality_report

def get_full_run_data(run_id: int) -> Dict[str, Any]:
    """Fetches all data for a single run_id."""
    conn = connect_db()
    data = {
        'run_metadata': get_run_metadata(conn, run_id),
        'test_results': get_api_test_data(conn, run_id),
        'benchmark_results': get_benchmark_data(conn, run_id),
        'quality_results': get_quality_data(conn, run_id)
    }
    conn.close()
    return data

# --- HTML Generation Functions ---

def format_diff(new_val: float, old_val: float, precision: int = 1, is_percent=False, lower_is_better=False) -> str:
    """Generates a color-coded diff string, e.g., (+10.5)"""
    diff = new_val - old_val
    unit = "%" if is_percent else ""
    
    if diff > 0.001:
        color = "green" if not lower_is_better else "red"
        return f' <span style="color:{color}; font-size: 0.8em;">({diff:+.{precision}f}{unit})</span>'
    elif diff < -0.001:
        color = "red" if not lower_is_better else "green"
        return f' <span style="color:{color}; font-size: 0.8em;">({diff:+.{precision}f}{unit})</span>'
    return ""

def generate_dashboard(output_file: str, run_data: Dict, old_run_data: Optional[Dict] = None):
    """Generate HTML dashboard"""
    
    # --- Extract Metrics ---
    new_test_summary = run_data['test_results'].get('summary', {})
    new_total_tests = new_test_summary.get('total_tests', 0)
    new_passed_tests = new_test_summary.get('passed', 0)
    new_avg_recall = new_test_summary.get('avg_recall', 0)
    new_avg_latency = new_test_summary.get('avg_latency_ms', 0)
    new_bench_results = run_data['benchmark_results'].get('results', [])
    new_quality_results = run_data['quality_results']

    run_meta = run_data['run_metadata']
    run_time = datetime.fromtimestamp(run_meta['run_timestamp']).strftime('%B %d, %Y at %I:%M %p')
    title = f"Run {run_meta['run_id']} ({run_meta['git_commit_hash']})"
    subtitle = f"Notes: {run_meta.get('notes') or 'N/A'} | Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"

    # --- Handle Comparison ---
    is_comparison = old_run_data is not None
    old_test_summary = {}
    old_bench_map = {}
    old_quality_map = {}
    
    if is_comparison:
        old_run_meta = old_run_data['run_metadata']
        title = f"Comparison: Run {old_run_meta['run_id']} vs Run {run_meta['run_id']}"
        subtitle = f"Run {old_run_meta['run_id']} ({old_run_meta['git_commit_hash']}) → Run {run_meta['run_id']} ({run_meta['git_commit_hash']})"
        old_test_summary = old_run_data['test_results'].get('summary', {})
        old_bench_map = {r['name']: r for r in old_run_data['benchmark_results'].get('results', [])}
        
        # Simple map for old quality results
        old_quality_map = {}
        for item in old_run_data['quality_results'].get('is_a', []): old_quality_map[f"is_a_{item['relationship']}"] = item['success']
        for item in old_run_data['quality_results'].get('part_of', []): old_quality_map[f"part_of_{item['relationship']}"] = item['success']
        for name, data in old_run_data['quality_results'].get('clustering', {}).items(): old_quality_map[f"cluster_{name}"] = data['connectivity']
        for item in old_run_data['quality_results'].get('multi_hop', []): old_quality_map[f"multi_hop_{item['start']}"] = item['success']


    # --- Extract Quality Metrics ---
    is_a_success = 0
    if new_quality_results.get('is_a'):
        is_a_success = sum(1 for r in new_quality_results['is_a'] if r.get('success', False))
        is_a_total = len(new_quality_results['is_a'])
    else:
        is_a_total = 0

    pass_rate_val = (new_passed_tests/new_total_tests*100 if new_total_tests > 0 else 0)
    pass_rate_diff = ""
    recall_diff = ""
    latency_diff = ""
    
    if is_comparison:
        old_pass_rate = (old_test_summary.get('passed', 0) / old_test_summary.get('total_tests', 1) * 100)
        pass_rate_diff = format_diff(pass_rate_val, old_pass_rate, precision=1, is_percent=True)
        recall_diff = format_diff(new_avg_recall, old_test_summary.get('avg_recall', 0), precision=3)
        latency_diff = format_diff(new_avg_latency, old_test_summary.get('avg_latency_ms', 0), precision=0, lower_is_better=True)

    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Embeddings Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; padding: 20px; min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 30px; }}
        h1 {{ font-size: 2.5em; color: #667eea; margin-bottom: 10px; }}
        .subtitle {{ color: #666; font-size: 1.1em; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.15); transition: transform 0.2s; }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: #667eea; margin: 10px 0; }}
        .metric-label {{ color: #666; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
        .metric-subtext {{ color: #999; font-size: 0.85em; margin-top: 5px; }}
        .section {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.15); margin-bottom: 20px; }}
        h2 {{ color: #667eea; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #eee; }}
        .progress-bar {{ background: #eee; height: 30px; border-radius: 15px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; transition: width 1s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }}
        .benchmark-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .benchmark-table th, .benchmark-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        .benchmark-table th {{ background: #f8f9fa; font-weight: 600; color: #667eea; }}
        .benchmark-table tr:hover {{ background: #f8f9fa; }}
        .status-badge {{ padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
        .status-pass {{ background: #d4edda; color: #155724; }}
        .status-fail {{ background: #f8d7da; color: #721c24; }}
        .status-neutral {{ background: #e2e3e5; color: #383d41; }}
        .chart-container {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .latency-bars {{ display: flex; align-items: flex-end; justify-content: space-around; height: 200px; margin-top: 20px; }}
        .latency-bar {{ flex: 1; margin: 0 10px; display: flex; flex-direction: column; align-items: center; }}
        .bar {{ width: 100%; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); border-radius: 5px 5px 0 0; position: relative; }}
        .bar-label {{ margin-top: 10px; font-size: 0.9em; color: #666; }}
        .bar-value {{ position: absolute; top: -25px; width: 100%; text-align: center; font-weight: bold; color: #667eea; }}
        .footer {{ text-align: center; color: white; margin-top: 30px; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Wikipedia Embeddings Dashboard</h1>
            <p class="subtitle"><b>{title}</b></p>
            <p class="subtitle" style="font-size: 0.9em; margin-top: 5px;">{subtitle}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Test Pass Rate</div>
                <div class="metric-value">{pass_rate_val:.1f}%{pass_rate_diff}</div>
                <div class="metric-subtext">{new_passed_tests} of {new_total_tests} tests passed</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Average Recall</div>
                <div class="metric-value">{new_avg_recall:.3f}{recall_diff}</div>
                <div class="metric-subtext">Semantic accuracy score</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Average Latency</div>
                <div class="metric-value">{new_avg_latency:.0f}<span style="font-size:0.5em">ms</span>{latency_diff}</div>
                <div class="metric-subtext">API test response time</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Relationship Quality</div>
                <div class="metric-value">{(is_a_success/is_a_total*100 if is_a_total > 0 else 0):.0f}%</div>
                <div class="metric-subtext">IS-A relationships detected</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Test Results Overview</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {(new_passed_tests/new_total_tests*100 if new_total_tests > 0 else 0):.1f}%">
                    {new_passed_tests}/{new_total_tests} Passed
                </div>
            </div>
"""
    
    # Add test category breakdown
    if 'tests' in run_data['test_results']:
        categories = {}
        for test in run_data['test_results']['tests']:
            cat = test.get('category', 'Unknown')
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if test.get('success', False):
                categories[cat]['passed'] += 1
        
        html += """
            <h3 style="margin-top: 30px; color: #666;">Performance by Category</h3>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Tests</th>
                        <th>Pass Rate</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for cat, data in sorted(categories.items()):
            pass_rate = (data['passed'] / data['total'] * 100) if data['total'] > 0 else 0
            status_class = 'status-pass' if pass_rate >= 70 else 'status-fail'
            html += f"""
                    <tr>
                        <td><strong>{cat}</strong></td>
                        <td>{data['passed']}/{data['total']}</td>
                        <td>{pass_rate:.1f}%</td>
                        <td><span class="status-badge {status_class}">{'PASS' if pass_rate >= 70 else 'NEEDS WORK'}</span></td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
"""
    
    # Add benchmark results
    if new_bench_results:
        html += """
        <div class="section">
            <h2>⚡ Performance Benchmarks</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>QPS</th>
                        <th>Avg Latency</th>
                        <th>P95 Latency</th>
                        <th>Success Rate</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for result in new_bench_results:
            name = result.get('name', 'Unknown')
            qps_diff = ""
            avg_lat_diff = ""
            p95_lat_diff = ""
            
            if is_comparison and name in old_bench_map:
                old_res = old_bench_map[name]
                qps_diff = format_diff(result.get('qps', 0), old_res.get('qps', 0), precision=1)
                avg_lat_diff = format_diff(result.get('avg_latency', 0), old_res.get('avg_latency', 0), precision=1, lower_is_better=True)
                p95_lat_diff = format_diff(result.get('p95_latency', 0), old_res.get('p95_latency', 0), precision=1, lower_is_better=True)

            html += f"""
                    <tr>
                        <td><strong>{name}</strong></td>
                        <td>{result.get('qps', 0):.1f}{qps_diff}</td>
                        <td>{result.get('avg_latency', 0):.1f}ms{avg_lat_diff}</td>
                        <td>{result.get('p95_latency', 0):.1f}ms{p95_lat_diff}</td>
                        <td>{result.get('success_rate', 0):.1%}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
"""
        
        # Add latency visualization
        query_type_results = [r for r in new_bench_results if 'Query Type' in r.get('name', '')]
        if not query_type_results:
            query_type_results = new_bench_results[:4] # Fallback
            
        max_latency = max(r.get('p95_latency', 0) for r in query_type_results) if query_type_results else 1
            
        html += """
            <div class="chart-container">
                <h3 style="color: #666; margin-bottom: 10px;">Latency Distribution (P95)</h3>
                <div class="latency-bars">
"""
            
        for result in query_type_results:
            name = result.get('name', 'Unknown').replace('Query Type: ', '')[:20]
            p95 = result.get('p95_latency', 0)
            height = (p95 / max_latency * 100) if max_latency > 0 else 0
            
            html += f"""
                    <div class="latency-bar">
                        <div class="bar" style="height: {height}%">
                            <div class="bar-value">{p95:.0f}ms</div>
                        </div>
                        <div class="bar-label">{name}</div>
                    </div>
"""
            
        html += """
                </div>
            </div>
"""
        
        html += """
        </div>
"""
    
    # Quality metrics section
    if new_quality_results:
        html += """
        <div class="section">
            <h2>🎯 Semantic Quality Analysis</h2>
"""
        
        # Domain clustering
        if 'clustering' in new_quality_results:
            html += """
            <h3 style="color: #666; margin: 20px 0 10px 0;">Domain Clustering</h3>
            <p style="color: #666; margin-bottom: 15px;">How well do related articles cluster together?</p>
"""
            
            for domain, data in new_quality_results['clustering'].items():
                connectivity = data.get('connectivity', 0) * 100
                conn_diff = ""
                if is_comparison:
                    old_conn = old_quality_map.get(f"cluster_{domain}", 0) * 100
                    conn_diff = format_diff(connectivity, old_conn, precision=1, is_percent=True)

                html += f"""
            <div style="margin: 15px 0;">
                <strong>{domain}</strong>
                <div class="progress-bar" style="height: 25px;">
                    <div class="progress-fill" style="width: {connectivity:.1f}%">
                        {connectivity:.1f}% connectivity{conn_diff}
                    </div>
                </div>
            </div>
"""
        
        html += """
        </div>
"""
    
    # Footer
    html += """
        <div class="footer">
            <p>Wikipedia Embeddings Testing Suite</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Dashboard generated: {output_file}")
    return output_file

def main():
    """Generate dashboard from existing results"""
    
    parser = argparse.ArgumentParser(description="Wikipedia Embeddings Dashboard Generator")
    parser.add_argument("--run_id", type=int, help="Generate report for this single run_id")
    parser.add_argument("--compare", type=int, help="The OLD run_id for comparison")
    parser.add_argument("--with", type=int, dest="with_id", help="The NEW run_id for comparison")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output HTML file (e.g., results/report.html)")
    args = parser.parse_args()

    if args.compare and args.with_id:
        # --- Comparison Report ---
        print(f"Generating comparison dashboard for runs: {args.compare} vs {args.with_id}")
        print("-" * 60)
        old_data = get_full_run_data(args.compare)
        new_data = get_full_run_data(args.with_id)
        
        generate_dashboard(args.output_file, new_data, old_run_data=old_data)
        
    elif args.run_id:
        # --- Single Run Report ---
        print(f"Generating single-run dashboard for run_id: {args.run_id}")
        print("-" * 60)
        run_data = get_full_run_data(args.run_id)
        
        generate_dashboard(args.output_file, run_data)
        
    else:
        print("Error: You must specify either --run_id OR --compare and --with")
        sys.exit(1)
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(args.output_file)}')
        print("  (Attempting to open in browser...)")
    except:
        pass

if __name__ == "__main__":
    main()