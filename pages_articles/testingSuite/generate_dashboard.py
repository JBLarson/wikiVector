#!/usr/bin/env python3
"""
WIKIPEDIA EMBEDDINGS DASHBOARD GENERATOR

Generates a beautiful HTML dashboard from your test results.
Run after test_wiki_api.py and benchmark_wiki.py to visualize results.
"""

import json
import os
from datetime import datetime
from typing import Dict, List

def load_json_safe(filename: str) -> Dict:
    """Load JSON file if it exists"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def generate_dashboard(
    test_results: Dict,
    benchmark_results: Dict,
    quality_results: Dict,
    output_file: str = "wiki_dashboard.html"
):
    """Generate HTML dashboard"""
    
    # Extract key metrics
    test_summary = test_results.get('summary', {})
    total_tests = test_summary.get('total_tests', 0)
    passed_tests = test_summary.get('passed', 0)
    avg_recall = test_summary.get('avg_recall', 0)
    avg_latency = test_summary.get('avg_latency_ms', 0)
    
    # Benchmark metrics
    bench_results = benchmark_results.get('results', [])
    
    # Quality metrics
    is_a_success = 0
    if quality_results.get('is_a'):
        is_a_success = sum(1 for r in quality_results['is_a'] if r.get('success', False))
        is_a_total = len(quality_results['is_a'])
    else:
        is_a_total = 0
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Embeddings Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        h1 {{
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-subtext {{
            color: #999;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        }}
        
        h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        
        .progress-bar {{
            background: #eee;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 1s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        .benchmark-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .benchmark-table th,
        .benchmark-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        
        .benchmark-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #667eea;
        }}
        
        .benchmark-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .status-badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .status-pass {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .latency-bars {{
            display: flex;
            align-items: flex-end;
            justify-content: space-around;
            height: 200px;
            margin-top: 20px;
        }}
        
        .latency-bar {{
            flex: 1;
            margin: 0 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .bar {{
            width: 100%;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px 5px 0 0;
            position: relative;
        }}
        
        .bar-label {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .bar-value {{
            position: absolute;
            top: -25px;
            width: 100%;
            text-align: center;
            font-weight: bold;
            color: #667eea;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Wikipedia Embeddings Dashboard</h1>
            <p class="subtitle">Performance & Quality Metrics | Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Test Pass Rate</div>
                <div class="metric-value">{(passed_tests/total_tests*100 if total_tests > 0 else 0):.1f}%</div>
                <div class="metric-subtext">{passed_tests} of {total_tests} tests passed</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Average Recall</div>
                <div class="metric-value">{avg_recall:.3f}</div>
                <div class="metric-subtext">Semantic accuracy score</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Average Latency</div>
                <div class="metric-value">{avg_latency:.0f}<span style="font-size:0.5em">ms</span></div>
                <div class="metric-subtext">Response time</div>
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
                <div class="progress-fill" style="width: {(passed_tests/total_tests*100 if total_tests > 0 else 0):.1f}%">
                    {passed_tests}/{total_tests} Passed
                </div>
            </div>
"""
    
    # Add test category breakdown
    if 'tests' in test_results:
        categories = {}
        for test in test_results['tests']:
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
    if bench_results:
        html += """
        <div class="section">
            <h2>⚡ Performance Benchmarks</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Queries</th>
                        <th>QPS</th>
                        <th>Avg Latency</th>
                        <th>P95 Latency</th>
                        <th>Success Rate</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for result in bench_results:
            html += f"""
                    <tr>
                        <td><strong>{result.get('name', 'Unknown')}</strong></td>
                        <td>{result.get('num_queries', 0)}</td>
                        <td>{result.get('qps', 0):.1f}</td>
                        <td>{result.get('avg_latency', 0):.1f}ms</td>
                        <td>{result.get('p95_latency', 0):.1f}ms</td>
                        <td>{result.get('success_rate', 0):.1%}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
"""
        
        # Add latency visualization
        if len(bench_results) > 0:
            max_latency = max(r.get('p95_latency', 0) for r in bench_results[:4])  # Top 4
            
            html += """
            <div class="chart-container">
                <h3 style="color: #666; margin-bottom: 10px;">Latency Distribution (P95)</h3>
                <div class="latency-bars">
"""
            
            for result in bench_results[:4]:
                name = result.get('name', 'Unknown')[:20]
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
    if quality_results:
        html += """
        <div class="section">
            <h2>🎯 Semantic Quality Analysis</h2>
"""
        
        # Domain clustering
        if 'clustering' in quality_results:
            html += """
            <h3 style="color: #666; margin: 20px 0 10px 0;">Domain Clustering</h3>
            <p style="color: #666; margin-bottom: 15px;">How well do related articles cluster together?</p>
"""
            
            for domain, data in quality_results['clustering'].items():
                connectivity = data.get('connectivity', 0) * 100
                html += f"""
            <div style="margin: 15px 0;">
                <strong>{domain}</strong>
                <div class="progress-bar" style="height: 25px;">
                    <div class="progress-fill" style="width: {connectivity:.1f}%">
                        {connectivity:.1f}% connectivity
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
            <p style="font-size: 0.9em; margin-top: 10px;">
                Testing ~7M articles • Powered by FAISS + Sentence Transformers
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Dashboard generated: {output_file}")
    return output_file

def main():
    """Generate dashboard from existing results"""
    
    print("Generating Wikipedia Embeddings Dashboard...")
    print("-" * 60)
    
    # Load results files
    test_results = load_json_safe('wiki_test_results.json')
    benchmark_results = load_json_safe('benchmark_results.json')
    quality_results = load_json_safe('semantic_quality_report.json')
    
    if not any([test_results, benchmark_results, quality_results]):
        print("⚠️  No result files found!")
        print("\nPlease run tests first:")
        print("  python test_wiki_api.py")
        print("  python benchmark_wiki.py")
        print("  python analyze_quality.py")
        return
    
    # Generate dashboard
    output_file = generate_dashboard(test_results, benchmark_results, quality_results)
    
    print(f"\n✓ Dashboard ready! Open {output_file} in your browser.")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        print("  (Attempting to open in browser...)")
    except:
        pass

if __name__ == "__main__":
    main()
