#!/usr/bin/env python3
"""
INTERACTIVE WIKIPEDIA EMBEDDINGS EXPLORER

Interactive CLI tool for exploring the semantic relationships in your Wikipedia vector DB.
Allows you to:
- Run queries and see detailed results
- Compare queries side-by-side
- Explore semantic neighborhoods
- Visualize scoring breakdowns
- Test edge cases interactively
"""

import requests
import json
import sys
from typing import List, Dict
from collections import defaultdict

API_BASE = "http://localhost:5001"

class WikiExplorer:
    """Interactive Wikipedia embeddings explorer"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.history: List[Dict] = []
    
    def query(self, text: str, verbose: bool = True) -> List[Dict]:
        """Execute a query and return results"""
        url = f"{self.base_url}/api/related/{text}"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                if verbose:
                    print(f"\n✗ Article '{text}' not found in database")
                return []
            
            if response.status_code != 200:
                if verbose:
                    print(f"\n✗ API error: {response.status_code}")
                return []
            
            results = response.json()
            
            if verbose:
                self.display_results(text, results)
            
            # Store in history
            self.history.append({
                'query': text,
                'results': results
            })
            
            return results
            
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Cannot connect to API at {self.base_url}")
            print("Make sure the Flask server is running!")
            return []
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return []
    
    def display_results(self, query: str, results: List[Dict]):
        """Display query results in a nice format"""
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        if not results:
            print("No results found")
            return
        
        print(f"\nFound {len(results)} related articles:\n")
        
        for i, item in enumerate(results, 1):
            title = item['title']
            score = item['score']
            
            # Visual score bar
            bar_length = int(score / 100 * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            
            print(f"{i}. {title}")
            print(f"   Score: {score}/100  [{bar}]")
            print()
    
    def compare_queries(self, query1: str, query2: str):
        """Compare results from two different queries"""
        print(f"\n{'='*80}")
        print(f"COMPARING QUERIES")
        print(f"{'='*80}")
        
        results1 = self.query(query1, verbose=False)
        results2 = self.query(query2, verbose=False)
        
        if not results1 or not results2:
            print("One or both queries returned no results")
            return
        
        titles1 = {r['title'] for r in results1}
        titles2 = {r['title'] for r in results2}
        
        overlap = titles1 & titles2
        unique1 = titles1 - titles2
        unique2 = titles2 - titles1
        
        print(f"\nQuery 1: '{query1}'")
        print(f"Query 2: '{query2}'")
        print(f"\nOverlap: {len(overlap)}/{len(results1)} articles")
        
        if overlap:
            print(f"\nShared results:")
            for title in list(overlap)[:5]:
                score1 = next(r['score'] for r in results1 if r['title'] == title)
                score2 = next(r['score'] for r in results2 if r['title'] == title)
                print(f"  • {title}")
                print(f"    Q1 score: {score1}/100, Q2 score: {score2}/100")
        
        if unique1:
            print(f"\nUnique to '{query1}':")
            for title in list(unique1)[:3]:
                print(f"  • {title}")
        
        if unique2:
            print(f"\nUnique to '{query2}':")
            for title in list(unique2)[:3]:
                print(f"  • {title}")
    
    def explore_neighborhood(self, article: str, depth: int = 2):
        """Explore semantic neighborhood by following links"""
        print(f"\n{'='*80}")
        print(f"EXPLORING SEMANTIC NEIGHBORHOOD")
        print(f"{'='*80}")
        print(f"Starting from: {article}")
        print(f"Depth: {depth}")
        
        visited = set()
        current_level = [article]
        
        for level in range(depth):
            print(f"\n--- Level {level + 1} ---")
            next_level = []
            
            for item in current_level:
                if item in visited:
                    continue
                
                visited.add(item)
                results = self.query(item, verbose=False)
                
                if results:
                    print(f"\n{item} →")
                    for r in results[:3]:  # Top 3 only
                        print(f"  • {r['title']} ({r['score']})")
                        if r['title'] not in visited:
                            next_level.append(r['title'])
            
            current_level = next_level[:5]  # Limit breadth
            
            if not current_level:
                print("\nNo more articles to explore")
                break
    
    def test_query_variations(self, base_query: str):
        """Test variations of a query to see how robust the search is"""
        print(f"\n{'='*80}")
        print(f"TESTING QUERY VARIATIONS")
        print(f"{'='*80}")
        
        variations = [
            base_query,
            base_query.lower(),
            base_query.upper(),
            base_query.replace(' ', '_'),
            base_query.replace('_', ' '),
        ]
        
        results_map = {}
        
        for var in variations:
            print(f"\nTrying: '{var}'")
            results = self.query(var, verbose=False)
            
            if results:
                top_title = results[0]['title']
                top_score = results[0]['score']
                results_map[var] = (top_title, top_score)
                print(f"  → {top_title} (score: {top_score})")
            else:
                print(f"  → No results")
        
        # Check consistency
        print(f"\n{'='*40}")
        top_results = [r[0] for r in results_map.values()]
        if len(set(top_results)) == 1:
            print("✓ All variations returned the same top result")
        else:
            print("✗ Variations returned different top results:")
            for var, (title, score) in results_map.items():
                print(f"  '{var}' → {title}")
    
    def analyze_score_distribution(self, query: str):
        """Analyze the score distribution for a query"""
        results = self.query(query, verbose=False)
        
        if not results:
            return
        
        print(f"\n{'='*80}")
        print(f"SCORE DISTRIBUTION ANALYSIS")
        print(f"{'='*80}")
        print(f"Query: {query}\n")
        
        scores = [r['score'] for r in results]
        
        print(f"Number of results: {len(scores)}")
        print(f"Score range: {min(scores)} - {max(scores)}")
        print(f"Average score: {sum(scores)/len(scores):.1f}")
        
        # Score gaps
        gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        print(f"\nScore gaps between results:")
        for i, gap in enumerate(gaps):
            print(f"  {i+1}→{i+2}: {gap} points")
        
        # Histogram
        print(f"\nScore distribution:")
        buckets = defaultdict(int)
        for score in scores:
            bucket = (score // 10) * 10
            buckets[bucket] += 1
        
        for bucket in sorted(buckets.keys(), reverse=True):
            count = buckets[bucket]
            bar = '█' * count
            print(f"  {bucket:3d}-{bucket+9:3d}: {bar} ({count})")
    
    def show_history(self):
        """Show query history"""
        if not self.history:
            print("\nNo query history yet")
            return
        
        print(f"\n{'='*80}")
        print(f"QUERY HISTORY")
        print(f"{'='*80}")
        
        for i, entry in enumerate(self.history[-10:], 1):  # Last 10
            query = entry['query']
            num_results = len(entry['results'])
            top_result = entry['results'][0]['title'] if entry['results'] else 'None'
            print(f"{i}. '{query}' → {num_results} results (top: {top_result})")
    
    def interactive_mode(self):
        """Interactive exploration mode"""
        print("\n" + "="*80)
        print("WIKIPEDIA EMBEDDINGS INTERACTIVE EXPLORER")
        print("="*80)
        print("\nCommands:")
        print("  query <text>           - Search for related articles")
        print("  compare <q1> | <q2>    - Compare two queries")
        print("  explore <article>      - Explore semantic neighborhood")
        print("  variations <query>     - Test query variations")
        print("  analyze <query>        - Analyze score distribution")
        print("  history                - Show query history")
        print("  quit                   - Exit")
        print()
        
        while True:
            try:
                cmd = input("\n> ").strip()
                
                if not cmd:
                    continue
                
                if cmd in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                parts = cmd.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == 'query':
                    if args:
                        self.query(args)
                    else:
                        print("Usage: query <text>")
                
                elif command == 'compare':
                    if '|' in args:
                        q1, q2 = args.split('|')
                        self.compare_queries(q1.strip(), q2.strip())
                    else:
                        print("Usage: compare <query1> | <query2>")
                
                elif command == 'explore':
                    if args:
                        self.explore_neighborhood(args, depth=2)
                    else:
                        print("Usage: explore <article>")
                
                elif command == 'variations':
                    if args:
                        self.test_query_variations(args)
                    else:
                        print("Usage: variations <query>")
                
                elif command == 'analyze':
                    if args:
                        self.analyze_score_distribution(args)
                    else:
                        print("Usage: analyze <query>")
                
                elif command == 'history':
                    self.show_history()
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' to see available commands")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Entry point"""
    explorer = WikiExplorer()
    
    # Check server connection
    try:
        requests.get(f"{API_BASE}/api/related/test", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to API at {API_BASE}")
        print("Make sure the Flask server is running:")
        print("  python server.py")
        sys.exit(1)
    
    # Run in interactive mode
    if len(sys.argv) > 1:
        # Direct query mode
        query = " ".join(sys.argv[1:])
        explorer.query(query)
    else:
        # Interactive mode
        explorer.interactive_mode()


if __name__ == "__main__":
    main()
