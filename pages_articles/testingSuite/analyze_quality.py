#!/usr/bin/env python3
"""
SEMANTIC QUALITY ANALYZER

Analyzes the semantic quality of Wikipedia embeddings by testing:
- Conceptual relationships (is-a, part-of, related-to)
- Analogies and reasoning
- Domain clustering
- Multi-hop relationships
- Disambiguation quality
"""

import requests
import json
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

API_BASE = "http://localhost:5001"

@dataclass
class RelationshipTest:
    """Test for a specific semantic relationship"""
    source: str
    relationship: str
    expected_targets: List[str]
    description: str = ""

@dataclass
class AnalogyTest:
    """Test for analogical reasoning: A is to B as C is to ?"""
    a: str
    b: str
    c: str
    expected_d: List[str]
    description: str = ""

class SemanticQualityAnalyzer:
    """Analyzes semantic quality of embeddings"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.cache: Dict[str, List[Dict]] = {}
    
    def query(self, text: str) -> List[Dict]:
        """Query with caching"""
        if text in self.cache:
            return self.cache[text]
        
        try:
            response = requests.get(f"{self.base_url}/api/related/{text}", timeout=30)
            if response.status_code == 200:
                results = response.json()
                self.cache[text] = results
                return results
        except Exception as e:
            print(f"Query error for '{text}': {e}")
        
        return []
    
    def get_titles(self, query: str) -> Set[str]:
        """Get set of result titles"""
        results = self.query(query)
        return {r['title'].lower() for r in results}
    
    def test_relationship(self, test: RelationshipTest) -> Dict:
        """Test a semantic relationship"""
        results = self.query(test.source)
        
        if not results:
            return {
                'success': False,
                'found': [],
                'expected': test.expected_targets,
                'relationship': test.relationship
            }
        
        result_titles = [r['title'] for r in results]
        result_titles_lower = [t.lower() for t in result_titles]
        
        # Check how many expected targets were found
        found = []
        for expected in test.expected_targets:
            if any(expected.lower() in title for title in result_titles_lower):
                found.append(expected)
        
        success = len(found) > 0
        
        return {
            'success': success,
            'found': found,
            'expected': test.expected_targets,
            'top_results': result_titles[:5],
            'relationship': test.relationship
        }
    
    def test_analogy(self, test: AnalogyTest) -> Dict:
        """Test analogical reasoning"""
        # Get neighborhoods
        a_neighbors = self.get_titles(test.a)
        b_neighbors = self.get_titles(test.b)
        c_neighbors = self.get_titles(test.c)
        
        # What does A→B tell us? Look for pattern
        # Then apply to C
        
        # Simple approach: find D in C's neighborhood
        c_results = self.query(test.c)
        
        if not c_results:
            return {
                'success': False,
                'found': None,
                'expected': test.expected_d
            }
        
        c_titles = [r['title'] for r in c_results]
        c_titles_lower = [t.lower() for t in c_titles]
        
        # Check if expected D is in results
        found = []
        for expected in test.expected_d:
            if any(expected.lower() in title for title in c_titles_lower):
                found.append(expected)
        
        return {
            'success': len(found) > 0,
            'found': found,
            'expected': test.expected_d,
            'c_results': c_titles[:5]
        }
    
    def test_domain_clustering(self, domain: str, articles: List[str]) -> Dict:
        """Test if articles in the same domain cluster together"""
        print(f"\nTesting domain clustering: {domain}")
        
        # For each article, check if other domain articles appear in results
        connections = defaultdict(set)
        
        for article in articles:
            results = self.query(article)
            result_titles = {r['title'].lower() for r in results}
            
            # Which other domain articles appear?
            for other in articles:
                if other != article:
                    if any(other.lower() in title for title in result_titles):
                        connections[article].add(other)
        
        # Calculate connectivity
        total_possible = len(articles) * (len(articles) - 1)
        total_found = sum(len(connected) for connected in connections.values())
        
        connectivity = total_found / total_possible if total_possible > 0 else 0
        
        print(f"  Articles: {len(articles)}")
        print(f"  Connections found: {total_found}/{total_possible}")
        print(f"  Connectivity: {connectivity:.2%}")
        
        # Show connection matrix
        print(f"\n  Connection matrix:")
        for article in articles:
            connected = connections.get(article, set())
            print(f"    {article:30s} → {len(connected)} connections")
        
        return {
            'domain': domain,
            'num_articles': len(articles),
            'connectivity': connectivity,
            'connections': dict(connections)
        }
    
    def test_multi_hop(self, start: str, target: str, max_hops: int = 3) -> Dict:
        """Test if we can reach target from start in N hops"""
        print(f"\nMulti-hop test: {start} → {target} (max {max_hops} hops)")
        
        visited = set()
        current_level = {start}
        
        for hop in range(max_hops):
            print(f"  Hop {hop + 1}:", end=" ")
            next_level = set()
            
            for article in current_level:
                if article in visited:
                    continue
                
                visited.add(article)
                results = self.query(article)
                
                for r in results:
                    title = r['title']
                    title_lower = title.lower()
                    
                    # Found target?
                    if target.lower() in title_lower:
                        print(f"✓ Found '{title}'!")
                        return {
                            'success': True,
                            'hops': hop + 1,
                            'path': list(visited) + [title]
                        }
                    
                    next_level.add(title)
            
            print(f"{len(next_level)} new articles")
            current_level = next_level
            
            if not current_level:
                break
        
        print(f"  ✗ Target not found within {max_hops} hops")
        return {
            'success': False,
            'hops': max_hops,
            'visited': list(visited)
        }
    
    def test_disambiguation(self, ambiguous_term: str, contexts: List[Tuple[str, str]]) -> Dict:
        """Test disambiguation quality
        
        contexts: list of (context_query, expected_result) tuples
        """
        print(f"\nDisambiguation test: '{ambiguous_term}'")
        
        results = {}
        
        for context, expected in contexts:
            query = f"{ambiguous_term} {context}"
            query_results = self.query(query)
            
            if not query_results:
                results[context] = {'success': False, 'found': None}
                continue
            
            top_titles = [r['title'] for r in query_results[:3]]
            
            # Check if expected appears in top results
            found = any(expected.lower() in title.lower() for title in top_titles)
            
            results[context] = {
                'success': found,
                'expected': expected,
                'top_results': top_titles
            }
            
            status = "✓" if found else "✗"
            print(f"  {status} '{context}' → {top_titles[0]}")
        
        success_rate = sum(1 for r in results.values() if r['success']) / len(results)
        
        return {
            'term': ambiguous_term,
            'success_rate': success_rate,
            'contexts': results
        }
    
    def run_comprehensive_analysis(self):
        """Run all semantic quality tests"""
        
        print("="*80)
        print("SEMANTIC QUALITY ANALYSIS")
        print("="*80)
        
        all_results = {}
        
        # 1. IS-A Relationships (Taxonomy)
        print("\n--- IS-A Relationships ---")
        
        is_a_tests = [
            RelationshipTest(
                "Python_(programming_language)",
                "is-a",
                ["Programming language", "Object-oriented"],
                "Python is a programming language"
            ),
            RelationshipTest(
                "Golden_Retriever",
                "is-a",
                ["Dog", "Mammal", "Animal"],
                "Golden Retriever is a dog"
            ),
            RelationshipTest(
                "Paris",
                "is-a",
                ["City", "Capital"],
                "Paris is a city"
            ),
        ]
        
        is_a_results = []
        for test in is_a_tests:
            result = self.test_relationship(test)
            is_a_results.append(result)
            status = "✓" if result['success'] else "✗"
            print(f"{status} {test.source}: found {result['found']}")
        
        all_results['is_a'] = is_a_results
        
        # 2. PART-OF Relationships
        print("\n--- PART-OF Relationships ---")
        
        part_of_tests = [
            RelationshipTest(
                "Heart",
                "part-of",
                ["Circulatory system", "Body", "Organ"],
                "Heart is part of circulatory system"
            ),
            RelationshipTest(
                "Wheel",
                "part-of",
                ["Car", "Vehicle", "Automobile"],
                "Wheel is part of car"
            ),
        ]
        
        part_of_results = []
        for test in part_of_tests:
            result = self.test_relationship(test)
            part_of_results.append(result)
            status = "✓" if result['success'] else "✗"
            print(f"{status} {test.source}: found {result['found']}")
        
        all_results['part_of'] = part_of_results
        
        # 3. Domain Clustering
        print("\n--- Domain Clustering ---")
        
        domains = {
            'Programming Languages': [
                'Python_(programming_language)',
                'JavaScript',
                'Java_(programming_language)',
                'C++',
                'Ruby_(programming_language)'
            ],
            'Renaissance Artists': [
                'Leonardo_da_Vinci',
                'Michelangelo',
                'Raphael',
                'Donatello',
            ],
            'Physics Concepts': [
                'Quantum_mechanics',
                'General_relativity',
                'Thermodynamics',
                'Electromagnetism'
            ]
        }
        
        clustering_results = {}
        for domain, articles in domains.items():
            result = self.test_domain_clustering(domain, articles)
            clustering_results[domain] = result
        
        all_results['clustering'] = clustering_results
        
        # 4. Analogies
        print("\n--- Analogical Reasoning ---")
        
        analogy_tests = [
            AnalogyTest(
                "Paris", "France",
                "London", ["United Kingdom", "England"],
                "Capital city analogy"
            ),
            AnalogyTest(
                "Einstein", "Relativity",
                "Newton", ["Gravity", "Motion", "Physics"],
                "Scientist-Theory analogy"
            ),
        ]
        
        analogy_results = []
        for test in analogy_tests:
            result = self.test_analogy(test)
            analogy_results.append(result)
            status = "✓" if result['success'] else "✗"
            print(f"{status} {test.a}:{test.b} :: {test.c}:? → {result['found']}")
        
        all_results['analogies'] = analogy_results
        
        # 5. Multi-hop reasoning
        print("\n--- Multi-hop Relationships ---")
        
        multi_hop_tests = [
            ("Python_(programming_language)", "Guido van Rossum"),
            ("Machine_learning", "Neural network"),
            ("Beatles", "Liverpool"),
        ]
        
        multi_hop_results = []
        for start, target in multi_hop_tests:
            result = self.test_multi_hop(start, target, max_hops=3)
            multi_hop_results.append(result)
        
        all_results['multi_hop'] = multi_hop_results
        
        # 6. Disambiguation
        print("\n--- Disambiguation ---")
        
        disambiguation_tests = [
            ("Mercury", [
                ("planet", "Planet"),
                ("element", "Chemical element"),
                ("mythology", "Roman mythology")
            ]),
            ("Apple", [
                ("fruit", "Fruit"),
                ("company", "Apple Inc"),
                ("technology", "Apple Inc")
            ]),
        ]
        
        disambiguation_results = []
        for term, contexts in disambiguation_tests:
            result = self.test_disambiguation(term, contexts)
            disambiguation_results.append(result)
        
        all_results['disambiguation'] = disambiguation_results
        
        # Summary
        self.print_analysis_summary(all_results)
        
        return all_results
    
    def print_analysis_summary(self, results: Dict):
        """Print summary of analysis"""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        # IS-A relationships
        is_a_success = sum(1 for r in results['is_a'] if r['success'])
        print(f"\nIS-A Relationships: {is_a_success}/{len(results['is_a'])} passed")
        
        # PART-OF relationships
        part_of_success = sum(1 for r in results['part_of'] if r['success'])
        print(f"PART-OF Relationships: {part_of_success}/{len(results['part_of'])} passed")
        
        # Domain clustering
        print(f"\nDomain Clustering:")
        for domain, data in results['clustering'].items():
            print(f"  {domain:25s}: {data['connectivity']:.1%} connectivity")
        
        # Analogies
        analogy_success = sum(1 for r in results['analogies'] if r['success'])
        print(f"\nAnalogical Reasoning: {analogy_success}/{len(results['analogies'])} passed")
        
        # Multi-hop
        multi_hop_success = sum(1 for r in results['multi_hop'] if r['success'])
        print(f"Multi-hop Reasoning: {multi_hop_success}/{len(results['multi_hop'])} passed")
        
        # Disambiguation
        avg_disambiguation = sum(r['success_rate'] for r in results['disambiguation']) / len(results['disambiguation'])
        print(f"Disambiguation Quality: {avg_disambiguation:.1%}")
        
        print("\n" + "="*80)


def main():
    """Run semantic quality analysis"""
    
    # Check server
    try:
        requests.get(f"{API_BASE}/api/related/test", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to API at {API_BASE}")
        print("Make sure the Flask server is running!")
        return
    
    analyzer = SemanticQualityAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    # Export results
    with open('semantic_quality_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✓ Results exported to semantic_quality_report.json")


if __name__ == "__main__":
    main()
