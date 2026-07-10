"""
Hybrid retrieval - combines dense (FAISS) and sparse (BM25) scores
for the same corpus, so a query benefits from both semantic similarity
AND exact keyword/term matching.
"""


def normalize_scores(results: list) -> list:
    """Min-max normalize scores to [0, 1] so dense and BM25 scores
    (which live on very different numeric scales) can be combined fairly."""
    if not results:
        return results
    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    range_s = max_s - min_s if max_s != min_s else 1.0
    for r in results:
        r["normalized_score"] = (r["score"] - min_s) / range_s
    return results


def hybrid_search(dense_results: list, bm25_results: list, alpha: float = 0.5, top_k: int = 6) -> list:
    """
    Combines dense and BM25 results via weighted score fusion.

    Args:
        dense_results: output of FaissVectorStore.search()
        bm25_results: output of BM25Store.search()
        alpha: weight for dense score (1-alpha goes to BM25).
               0.5 = equal weighting, 1.0 = pure dense, 0.0 = pure BM25
    """
    dense_results = normalize_scores(list(dense_results))
    bm25_results = normalize_scores(list(bm25_results))

    combined = {}

    for r in dense_results:
        key = (r["metadata"]["source"], r["metadata"]["parent_id"])
        combined[key] = {
            "chunk": r["chunk"],
            "metadata": r["metadata"],
            "dense_score": r["normalized_score"],
            "bm25_score": 0.0
        }

    for r in bm25_results:
        key = (r["metadata"]["source"], r["metadata"]["parent_id"])
        if key in combined:
            combined[key]["bm25_score"] = r["normalized_score"]
        else:
            combined[key] = {
                "chunk": r["chunk"],
                "metadata": r["metadata"],
                "dense_score": 0.0,
                "bm25_score": r["normalized_score"]
            }

    for key, item in combined.items():
        item["score"] = alpha * item["dense_score"] + (1 - alpha) * item["bm25_score"]

    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

def reciprocal_rank_fusion(dense_results: list, bm25_results: list, k: int = 60, top_k: int = 6) -> list:
    """
    Combines dense and BM25 rankings via Reciprocal Rank Fusion (RRF) -
    a standard IR technique that fuses based on RANK POSITION, not raw
    or normalized score magnitude. This avoids the problem where one
    retriever's high-confidence WRONG match (e.g. BM25 scoring 1.000 on
    an irrelevant chunk) can outweigh another retriever's correct but
    less "confident-looking" match, since score scales aren't directly
    comparable between different retrieval methods.
    """
    rrf_scores = {}
    chunk_lookup = {}

    for rank, r in enumerate(dense_results, start=1):
        key = (r["metadata"]["source"], r["metadata"]["parent_id"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        chunk_lookup[key] = r

    for rank, r in enumerate(bm25_results, start=1):
        key = (r["metadata"]["source"], r["metadata"]["parent_id"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        chunk_lookup[key] = r

    ranked_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in ranked_keys[:top_k]:
        r = chunk_lookup[key]
        results.append({
            "chunk": r["chunk"],
            "metadata": r["metadata"],
            "score": rrf_scores[key]
        })
    return results