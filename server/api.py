#!/usr/bin/env python3
"""
WIKIPEDIA SEMANTIC SEARCH API - MULTI-SIGNAL RANKING

Combines multiple signals for high-quality search results:
1. Semantic Similarity (50%) - How relevant to the query
2. PageRank (30%) - How important/central the article is
3. Pageviews (10%) - How popular/trending the article is
4. Title Match (10%) - How well the title matches the query

This creates a balanced ranking that surfaces both relevant AND important articles.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import faiss
import sqlite3
import math
from sentence_transformers import SentenceTransformer
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Multi-Signal Ranking Weights ---
# These determine how much each signal contributes to the final score
WEIGHT_SEMANTIC = 0.50      # How semantically similar to query (primary signal)
WEIGHT_PAGERANK = 0.30      # How important/central the article is (PageRank)
WEIGHT_PAGEVIEWS = 0.10     # How popular/trendy the article is (recent interest)
WEIGHT_TITLE_MATCH = 0.10   # How well the title matches the query (specificity)

EPSILON = 1e-8              # Small constant to prevent log(0) issues

# --- Search Parameters ---
CANDIDATE_POOL_SIZE = 200   # Initial semantic search pool (larger = better quality)
RESULTS_TO_RETURN = 32      # Final results returned to user

# --- File Paths ---
INDEX_PATH = "../data/index.faiss"
METADATA_PATH = "../data/metadata.db"

# ============================================================================
# INITIALIZE RESOURCES
# ============================================================================

print("="*80)
print("WIKIPEDIA SEMANTIC SEARCH API")
print("="*80)

print("\nLoading FAISS index...")
index = faiss.read_index(INDEX_PATH)
try:
    ivf_index = faiss.downcast_index(index.index)
    ivf_index.nprobe = 32
    print(f"✓ IVF index loaded (nprobe={ivf_index.nprobe})")
except:
    print("✓ Flat index loaded")

print("\nLoading metadata database...")
db = sqlite3.connect(METADATA_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row

# Verify available signals
cursor = db.cursor()
cursor.execute("PRAGMA table_info(articles)")
columns = {row[1] for row in cursor.fetchall()}

available_signals = {
    'pagerank': 'pagerank' in columns,
    'pageviews': 'pageviews' in columns,
    'backlinks': 'backlinks' in columns
}

print(f"\nAvailable signals:")
print(f"  PageRank: {'✓' if available_signals['pagerank'] else '✗'}")
print(f"  Pageviews: {'✓' if available_signals['pageviews'] else '✗'}")
print(f"  Backlinks: {'✓' if available_signals['backlinks'] else '✗'}")

print("\nLoading sentence transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("\n" + "="*80)
print("✓ API READY")
print("="*80)
print(f"\nRanking weights:")
print(f"  Semantic similarity: {WEIGHT_SEMANTIC:.0%}")
print(f"  PageRank (importance): {WEIGHT_PAGERANK:.0%}")
print(f"  Pageviews (popularity): {WEIGHT_PAGEVIEWS:.0%}")
print(f"  Title match: {WEIGHT_TITLE_MATCH:.0%}")
print()

# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_pagerank(pagerank_score):
    """
    Normalize PageRank score (0-100) to 0-1 range.
    
    PageRank is already scaled 0-100 in the database, with:
    - 100 = most important article (e.g., United_States)
    - 50 = moderately important
    - 0-10 = obscure articles
    
    We apply a slight boost to high-importance articles.
    """
    if pagerank_score is None or pagerank_score <= 0:
        return 0.0
    
    # Scale 0-100 to 0-1, with slight exponential boost for top articles
    normalized = pagerank_score / 100.0
    
    # Apply power curve: boost high-ranking articles
    # This makes the difference between PR=80 and PR=100 more significant
    return normalized ** 0.8

def normalize_pageviews(pageview_count):
    """
    Normalize pageview count on a log scale.
    
    Pageview distribution is extremely skewed:
    - Popular: 100,000+ views/month
    - Moderate: 1,000-10,000 views/month  
    - Obscure: <100 views/month
    
    Use log scale to compress the range sensibly.
    """
    if pageview_count is None or pageview_count < 1:
        return 0.0
    
    # Log scale normalization
    # Min: 10 views (log10=1.0)
    # Max: 10,000,000 views (log10=7.0)
    min_log = 1.0
    max_log = 7.0
    
    score = (math.log10(pageview_count) - min_log) / (max_log - min_log)
    return min(1.0, max(0.0, score))

def calculate_title_match_score(title: str, query: str) -> float:
    """
    Calculate how well the title matches the query.
    
    Rewards:
    - Exact matches or very close matches
    - Titles where query words are prominent
    
    Penalizes:
    - Overly specific titles (e.g., "Education_in_Nigeria")
    - Year-specific articles (e.g., "2023_in_science")
    - List/index articles
    
    Returns: 0.0 to 1.0
    """
    title_lower = title.lower().replace('_', ' ')
    query_lower = query.lower()
    
    title_words = set(title_lower.split())
    query_words = set(query_lower.split())
    
    if not title_words:
        return 0.0
    
    # Base score: Jaccard similarity
    intersection = len(title_words & query_words)
    union = len(title_words | query_words)
    
    if union == 0:
        return 0.0
    
    base_score = intersection / union
    
    # Boost for exact or near-exact matches
    if title_lower == query_lower:
        return 1.0
    elif title_lower.startswith(query_lower) or query_lower in title_lower:
        base_score = min(1.0, base_score * 1.5)
    
    # PENALTIES for overly-specific articles
    
    # Pattern 1: Geographic specificity ("Topic in Place")
    if " in " in title_lower:
        parts = title_lower.split(" in ")
        if len(parts) == 2:
            place_indicators = [
                "africa", "asia", "europe", "america", "states", "kingdom",
                "china", "india", "russia", "france", "germany", "japan",
                "canada", "australia", "brazil", "mexico", "italy", "spain",
                "california", "texas", "york", "london", "paris", "tokyo"
            ]
            if any(place in parts[1] for place in place_indicators):
                base_score *= 0.5  # 50% penalty
    
    # Pattern 2: Time specificity (year-based articles)
    import re
    if re.match(r'^\d{4}', title_lower):  # Starts with year
        base_score *= 0.4  # 60% penalty
    
    # Pattern 3: List/meta articles
    meta_prefixes = ["list of", "index of", "glossary of", "timeline of", 
                     "outline of", "history of"]
    if any(title_lower.startswith(prefix) for prefix in meta_prefixes):
        base_score *= 0.3  # 70% penalty
    
    return min(1.0, max(0.0, base_score))

def is_meta_page(title):
    """Quick filter for obvious meta/administrative pages."""
    lower = title.lower()
    bad_prefixes = [
        'wikipedia:', 'template:', 'category:', 'portal:', 'help:', 
        'user:', 'talk:', 'file:', 'mediawiki:'
    ]
    return any(lower.startswith(p) for p in bad_prefixes) or '(disambiguation)' in lower

# ============================================================================
# RANKING ALGORITHM
# ============================================================================

def calculate_multisignal_score(
    semantic_similarity,
    pagerank_score,
    pageview_count,
    title,
    query
):
    """
    Calculate final ranking score using multiple signals.
    
    Uses geometric mean with weights for each signal:
    - Higher weights = more influence on final score
    - Geometric mean penalizes articles weak in any single signal
    
    Formula:
        score = (sem^w1 * pr^w2 * pv^w3 * title^w4)
        
    Where weights sum to 1.0 and exponents determine influence.
    """
    
    # Normalize each signal to 0-1 range
    sem_norm = float(semantic_similarity)
    pr_norm = normalize_pagerank(pagerank_score)
    pv_norm = normalize_pageviews(pageview_count)
    title_norm = calculate_title_match_score(title, query)
    
    # Add epsilon to prevent log(0) in geometric mean
    sem_norm = max(sem_norm, EPSILON)
    pr_norm = max(pr_norm, EPSILON)
    pv_norm = max(pv_norm, EPSILON)
    title_norm = max(title_norm, EPSILON)
    
    # Weighted geometric mean
    # This naturally balances all signals and penalizes extreme weaknesses
    score = (
        (sem_norm ** WEIGHT_SEMANTIC) *
        (pr_norm ** WEIGHT_PAGERANK) *
        (pv_norm ** WEIGHT_PAGEVIEWS) *
        (title_norm ** WEIGHT_TITLE_MATCH)
    )
    
    return score

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/related/<path:query>', methods=['GET'])
def get_related(query):
    """
    Main search endpoint.
    
    Query parameters:
    - k: Number of results to return (default: 32)
    - ranking: Ranking mode ('default' or 'semantic_only')
    
    Returns:
    - JSON array of {title, score, debug_info} objects
    """
    cursor = db.cursor()
    
    # Parse parameters
    ranking_mode = request.args.get('ranking', 'default')
    debug_mode = request.args.get('debug', 'false').lower() == 'true'
    
    try:
        k_results = int(request.args.get('k', RESULTS_TO_RETURN))
        k_results = min(k_results, 100)  # Cap at 100
    except:
        k_results = RESULTS_TO_RETURN
    
    # Find the query article (to exclude from results)
    exclude_id = None
    
    # Try multiple strategies to find the article
    for lookup_strategy in [
        ("SELECT article_id FROM articles WHERE title = ?", query),
        ("SELECT article_id FROM articles WHERE title = ?", query.replace('_', ' ')),
        ("SELECT article_id FROM articles WHERE lookup_title = ?", query.lower()),
    ]:
        cursor.execute(*lookup_strategy)
        row = cursor.fetchone()
        if row:
            exclude_id = int(row['article_id'])
            break
    
    # Encode query to embedding
    try:
        search_text = query.replace('_', ' ')
        embedding = model.encode(
            [search_text], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        ).astype(np.float32)
    except Exception as e:
        print(f"Error encoding query '{query}': {e}")
        return jsonify({"error": "Failed to encode query"}), 500
    
    # Search FAISS index for candidates
    search_size = CANDIDATE_POOL_SIZE + 1 if exclude_id else CANDIDATE_POOL_SIZE
    distances, indices = index.search(embedding, search_size)
    
    # Extract candidate article IDs and their semantic scores
    candidate_ids = []
    semantic_scores = {}
    
    for i, idx in enumerate(indices[0]):
        idx_int = int(idx)
        if idx_int >= 0 and idx_int != exclude_id:
            candidate_ids.append(idx_int)
            semantic_scores[idx_int] = float(distances[0][i])
    
    if not candidate_ids:
        return jsonify([])
    
    # Fetch metadata for all candidates
    placeholders = ','.join('?' * len(candidate_ids))
    
    # Build query dynamically based on available columns
    query_columns = ['article_id', 'title']
    if available_signals['pagerank']:
        query_columns.append('pagerank')
    if available_signals['pageviews']:
        query_columns.append('pageviews')
    if available_signals['backlinks']:
        query_columns.append('backlinks')
    
    query_sql = f"SELECT {', '.join(query_columns)} FROM articles WHERE article_id IN ({placeholders})"
    cursor.execute(query_sql, candidate_ids)
    
    results = []
    data_map = {row['article_id']: row for row in cursor.fetchall()}
    
    # Score each candidate
    for cand_id in candidate_ids:
        data = data_map.get(cand_id)
        
        if not data or is_meta_page(data['title']):
            continue
        
        semantic_score = semantic_scores.get(cand_id, 0.0)
        
        # Calculate final score based on ranking mode
        if ranking_mode == 'semantic_only':
            final_score = semantic_score
            debug_info = {
                'semantic': semantic_score
            }
        else:
            # Multi-signal ranking
            pagerank = data.get('pagerank', 0) if available_signals['pagerank'] else 0
            pageviews = data.get('pageviews', 0) if available_signals['pageviews'] else 0
            
            final_score = calculate_multisignal_score(
                semantic_similarity=semantic_score,
                pagerank_score=pagerank,
                pageview_count=pageviews,
                title=data['title'],
                query=query
            )
            
            # Debug information (only included if debug=true)
            debug_info = {
                'semantic': semantic_score,
                'semantic_norm': float(semantic_score),
                'pagerank': float(pagerank) if pagerank else 0,
                'pagerank_norm': normalize_pagerank(pagerank),
                'pageviews': int(pageviews) if pageviews else 0,
                'pageviews_norm': normalize_pageviews(pageviews),
                'title_match': calculate_title_match_score(data['title'], query),
                'final_score': final_score
            }
        
        result = {
            "title": data['title'],
            "score": int(final_score * 100),
            "score_float": float(final_score)
        }
        
        if debug_mode:
            result['debug'] = debug_info
        
        results.append(result)
    
    # Sort by final score
    results.sort(key=lambda x: x['score_float'], reverse=True)
    
    # Return top K results
    final_results = [
        {k: v for k, v in r.items() if k != 'score_float'}
        for r in results[:k_results]
    ]
    
    return jsonify(final_results)


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns system configuration and status.
    """
    try:
        nprobe = faiss.downcast_index(index.index).nprobe
    except:
        nprobe = "N/A (Flat Index)"
    
    # Get some statistics
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_articles = cursor.fetchone()[0]
    
    # Count articles with each signal
    signal_coverage = {}
    if available_signals['pagerank']:
        cursor.execute("SELECT COUNT(*) FROM articles WHERE pagerank > 0")
        signal_coverage['pagerank'] = cursor.fetchone()[0]
    
    if available_signals['pageviews']:
        cursor.execute("SELECT COUNT(*) FROM articles WHERE pageviews > 0")
        signal_coverage['pageviews'] = cursor.fetchone()[0]
    
    if available_signals['backlinks']:
        cursor.execute("SELECT COUNT(*) FROM articles WHERE backlinks > 0")
        signal_coverage['backlinks'] = cursor.fetchone()[0]
    
    return jsonify({
        "status": "ok",
        "index_path": INDEX_PATH,
        "metadata_path": METADATA_PATH,
        "total_articles": total_articles,
        "index_total_vectors": index.ntotal,
        "nprobe": nprobe,
        "ranking_weights": {
            "semantic": WEIGHT_SEMANTIC,
            "pagerank": WEIGHT_PAGERANK,
            "pageviews": WEIGHT_PAGEVIEWS,
            "title_match": WEIGHT_TITLE_MATCH
        },
        "available_signals": available_signals,
        "signal_coverage": signal_coverage,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "default_results": RESULTS_TO_RETURN
    })


@app.route('/api/article/<path:title>', methods=['GET'])
def get_article_details(title):
    """
    Get detailed information about a specific article.
    
    Useful for debugging and understanding individual article scores.
    """
    cursor = db.cursor()
    
    # Try to find the article
    cursor.execute(
        "SELECT * FROM articles WHERE title = ? OR lookup_title = ?",
        (title, title.lower())
    )
    
    article = cursor.fetchone()
    
    if not article:
        return jsonify({"error": "Article not found"}), 404
    
    # Convert to dict
    article_dict = dict(article)
    
    # Add normalized scores
    article_dict['normalized_scores'] = {
        'pagerank': normalize_pagerank(article_dict.get('pagerank')),
        'pageviews': normalize_pageviews(article_dict.get('pageviews'))
    }
    
    return jsonify(article_dict)


if __name__ == '__main__':
    app.run(port=5001, debug=False, threaded=True)