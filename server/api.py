# server/api.py

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

# --- Config ---
WEIGHT_SEMANTIC = 0.40
WEIGHT_BACKLINKS = 0.50
WEIGHT_TITLE_OVERLAP = 0.10
EPSILON = 1e-6
# ----------------

CANDIDATE_POOL_SIZE = 100
RESULTS_TO_RETURN = 32
INDEX_PATH = "../data/index.faiss"
METADATA_PATH = "../data/metadata.db"
# --------------


# --- Load resources ---
print("Loading index...")
index = faiss.read_index(INDEX_PATH)
try:
    ivf_index = faiss.downcast_index(index.index)
    ivf_index.nprobe = 32
    print(f"Index is IVF, nprobe={ivf_index.nprobe}")
except:
    print("Index is Flat.")

print("Loading database...")
db = sqlite3.connect(METADATA_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row

print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Ready!")
# ---------------------


def normalize_backlinks(backlink_count):
    """
    Normalizes backlink count on a log scale.
    
    Important articles have 1000-50000+ backlinks.
    Obscure articles have <100 backlinks.
    
    Min: 10 backlinks (log10=1.0)
    Max: 50,000 backlinks (log10=4.7)
    Maps 1.0-4.7 to 0.0-1.0
    """
    if backlink_count is None or backlink_count < 10:
        return 0.0
    
    min_log = 1.0      # log10(10) = 1.0
    max_log = 4.699    # log10(50,000) = 4.699
    
    score = (math.log10(backlink_count) - min_log) / (max_log - min_log)
    return min(1.0, max(0.0, score))

def calculate_title_query_overlap(title: str, query: str) -> float:
    """
    Calculate what fraction of the title is covered by query terms.
    
    Higher score = query terms dominate the title (more general topic)
    Lower score = title has many extra terms (more specific topic)
    
    Examples:
        title="Education", query="education" → 1.0 (perfect match)
        title="Philosophy_of_education", query="education" → 0.33
        title="Education_in_Nigeria", query="education" → 0.33
    """
    title_words = set(title.lower().replace('_', ' ').split())
    query_words = set(query.lower().split())
    
    if not title_words:
        return 0.0
    
    # How many title words are in the query?
    overlap = len(title_words & query_words)
    ratio = overlap / len(title_words)
    
    return ratio

def calculate_specificity_penalty(title: str, query: str) -> float:
    """
    Penalize overly-specific article titles.
    
    Returns: 0.5 to 1.0 (multiplier)
    """
    title_lower = title.lower()
    query_lower = query.lower()
    
    # Pattern 1: "[Topic] in [Place]" - geographic specificity
    if " in " in title_lower or "_in_" in title_lower:
        # Check if it's actually a place-specific article
        # (not a phrase like "interest in" or "increase in")
        parts = title_lower.split("_in_")
        if len(parts) == 2:
            # Common place indicators
            place_words = ["africa", "asia", "europe", "america", "states", "kingdom", 
                          "china", "india", "russia", "france", "germany", "japan",
                          "canada", "australia", "brazil", "mexico", "italy", "spain"]
            if any(place in parts[1] for place in place_words):
                return 0.6  # 40% penalty for geographic specificity
    
    # Pattern 2: "[Year] [Topic]" or "[Year]-[Year] [Topic]" - time-specific
    import re
    if re.match(r'^\d{4}[-_]', title_lower):
        return 0.5  # 50% penalty for year-specific articles
    
    # Pattern 3: "List_of_", "Index_of_", etc (should have been filtered, but just in case)
    if any(title_lower.startswith(prefix) for prefix in [
        "list_of_", "index_of_", "glossary_of_", "timeline_of_"
    ]):
        return 0.4  # 60% penalty
    
    return 1.0  # No penalty

def is_meta_page(title):
    lower = title.lower()
    bad_prefixes = ['wikipedia:', 'template:', 'category:', 'portal:', 'help:', 'user:', 
                    'talk:', 'file:', 'list of', 'outline of', 'timeline of', 'history of']
    return any(lower.startswith(p) for p in bad_prefixes) or '(disambiguation)' in lower

@app.route('/api/related/<path:query>', methods=['GET'])
def get_related(query):
    cursor = db.cursor()
    
    ranking_mode = request.args.get('ranking', 'default')
    try:
        k_results = request.args.get('k', default=RESULTS_TO_RETURN, type=int)
    except:
        k_results = RESULTS_TO_RETURN
    
    exclude_id = None
    
    # Strategy 1: Exact match with underscores
    cursor.execute("SELECT article_id FROM articles WHERE title = ?", (query,))
    row = cursor.fetchone()
    if row:
        exclude_id = int(row['article_id'])
    
    # Strategy 2: Try with spaces instead of underscores
    if not exclude_id:
        cursor.execute("SELECT article_id FROM articles WHERE title = ?", (query.replace('_', ' '),))
        row = cursor.fetchone()
        if row:
            exclude_id = int(row['article_id'])
    
    # Strategy 3: Case-insensitive search (using pre-normalized column)
    if not exclude_id:
        cursor.execute("SELECT article_id FROM articles WHERE lookup_title = ?", (query.lower(),))
        row = cursor.fetchone()
        if row:
            exclude_id = int(row['article_id'])
            
    try:
        search_text = query.replace('_', ' ')
        embedding = model.encode([search_text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    except Exception as e:
        print(f"Error encoding query '{query}': {e}")
        return jsonify([])
    
    search_size = CANDIDATE_POOL_SIZE + 1 if exclude_id else CANDIDATE_POOL_SIZE
    distances, indices = index.search(embedding, search_size)
    
    candidate_ids = []
    candidate_scores = {}
    
    for i, idx in enumerate(indices[0]):
        idx_int = int(idx)
        if idx_int >= 0 and idx_int != exclude_id:
            candidate_ids.append(idx_int)
            candidate_scores[idx_int] = distances[0][i]
            
    if not candidate_ids:
        return jsonify([])
    
    # Fetch metadata (now including backlinks)
    placeholders = ','.join('?' * len(candidate_ids))
    cursor.execute(
        f"SELECT article_id, title, backlinks FROM articles WHERE article_id IN ({placeholders})",
        candidate_ids
    )
    
    results = []
    data_map = {r['article_id']: r for r in cursor.fetchall()}
    
    for cand_id in candidate_ids:
        data = data_map.get(cand_id)
        
        if not data or is_meta_page(data['title']):
            continue
        
        semantic_score = candidate_scores.get(cand_id, 0.0)
        
        if ranking_mode == 'semantic':
            final_score = semantic_score
        else:
            # --- NEW HYBRID RANKING WITH BACKLINKS ---
            
            # 1. Semantic similarity (primary signal - 70%)
            semantic_weight = WEIGHT_SEMANTIC
            
            # 2. Backlinks (centrality/importance - 20%)
            backlinks = data['backlinks'] if data['backlinks'] is not None else 0
            centrality_score = normalize_backlinks(backlinks)
            centrality_weight = WEIGHT_BACKLINKS
            
            # 3. Title-Query Overlap (generality - 10%)
            overlap_score = calculate_title_query_overlap(data['title'], query)
            overlap_weight = WEIGHT_TITLE_OVERLAP
            
            # 4. Specificity Penalty (pattern-based filter)
            specificity = calculate_specificity_penalty(data['title'], query)
            
            # Weighted Geometric Mean with Penalty
            final_score = (
                (semantic_score + EPSILON) ** semantic_weight *
                (centrality_score + EPSILON) ** centrality_weight *
                (overlap_score + EPSILON) ** overlap_weight *
                specificity
            )
            # ----------------------------------------------
        
        results.append({
            "title": data['title'],
            "score_float": final_score,
            "score": int(final_score * 100)
        })
    
    results.sort(key=lambda x: x['score_float'], reverse=True)
    
    final_results = [{"title": r["title"], "score": r["score"]} for r in results[:k_results]]
    
    return jsonify(final_results)


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        nprobe = faiss.downcast_index(index.index).nprobe
    except:
        nprobe = "N/A (Flat Index)"
    
    # Check if backlinks column exists
    cursor = db.cursor()
    cursor.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cursor.fetchall()]
    backlinks_available = 'backlinks' in columns
    
    return jsonify({
        "status": "ok",
        "index_path": INDEX_PATH,
        "metadata_path": METADATA_PATH,
        "index_total_vectors": index.ntotal,
        "nprobe": nprobe,
        "weight_semantic": WEIGHT_SEMANTIC,
        "weight_backlinks": WEIGHT_BACKLINKS,
        "weight_title_overlap": WEIGHT_TITLE_OVERLAP,
        "backlinks_available": backlinks_available,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "results_to_return": RESULTS_TO_RETURN
    })

if __name__ == '__main__':
    app.run(port=5001, debug=False, threaded=True)