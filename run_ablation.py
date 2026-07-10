"""
Retrieval ablation: dense-only vs. BM25-only vs. hybrid, all measured
against the exact same eval set and entry-level recall metric already
built and verified. This directly tests whether BM25's exact keyword
matching fixes the dense-embedding weakness observed earlier (e.g.
"partial" outranking "cache"/"lru_cache" for a caching-specific question).
"""

import sys
sys.path.insert(0, ".")

from build_pipeline_v2 import build_pipeline, dedupe_by_parent
from bm25_store import BM25Store
from hybrid_store import hybrid_search
from eval_questions import EVAL_QUESTIONS


def build_bm25_index(store):
    """Extracts the same chunks/metadata already in the FAISS store,
    builds a parallel BM25 index over identical content."""
    bm25 = BM25Store()
    # FaissVectorStore keeps chunks/metadata as parallel lists internally
    bm25.add(store.chunks, store.metadata)
    return bm25


def evaluate_strategy(strategy_name: str, search_fn, questions: list, k_values=[1, 3, 5]):
    results = []

    for q in questions:
        if not q["answerable"]:
            continue

        retrieved = search_fn(q["query"])
        retrieved_entries = [r["metadata"].get("entry_name", "unknown") for r in retrieved]

        expected_entries = set((q.get("expected_entry") or "").split(","))
        expected_entries.discard("")

        row = {"query": q["query"], "category": q["category"]}
        for k in k_values:
            top_k_entries = set(retrieved_entries[:k])
            row[f"recall@{k}"] = bool(expected_entries & top_k_entries)

        results.append(row)

    return results


def summarize_strategy(strategy_name: str, results: list, k_values=[1, 3, 5]):
    print(f"\n{'='*70}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*70}")
    total = len(results)
    for k in k_values:
        hits = sum(1 for r in results if r[f"recall@{k}"])
        print(f"  Recall@{k}: {hits}/{total} = {hits/total*100:.1f}%")


if __name__ == "__main__":
    store, embed_model = build_pipeline()
    bm25 = build_bm25_index(store)

    def dense_search(query):
        query_vec = embed_model.embed_one(query)
        return dedupe_by_parent(store.search(query_vec, top_k=12))[:5]

    def bm25_search(query):
        raw = bm25.search(query, top_k=12)
        seen = set()
        deduped = []
        for r in raw:
            key = r["metadata"]["parent_id"]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:5]

    def hybrid_search_fn(query):
        query_vec = embed_model.embed_one(query)
        dense_raw = store.search(query_vec, top_k=12)
        bm25_raw = bm25.search(query, top_k=12)
        return hybrid_search(dense_raw, bm25_raw, alpha=0.5, top_k=5)

    print("Running ablation: dense vs BM25 vs hybrid...\n")

    dense_results = evaluate_strategy("Dense (FAISS)", dense_search, EVAL_QUESTIONS)
    bm25_results = evaluate_strategy("BM25", bm25_search, EVAL_QUESTIONS)
    hybrid_results = evaluate_strategy("Hybrid (alpha=0.5)", hybrid_search_fn, EVAL_QUESTIONS)

    summarize_strategy("Dense (FAISS)", dense_results)
    summarize_strategy("BM25", bm25_results)
    summarize_strategy("Hybrid (alpha=0.5)", hybrid_results)

    # Specifically check the two known problem questions
    print(f"\n{'='*70}")
    print("SPOT CHECK: the two questions where dense retrieval failed at recall@1")
    print(f"{'='*70}")
    problem_queries = [
        "How can I avoid recomputing an expensive function call with the same inputs?",
        "How does partial relate to the args and keywords attributes?"
    ]
    for pq in problem_queries:
        print(f"\nQuery: {pq}")
        for name, fn in [("Dense", dense_search), ("BM25", bm25_search), ("Hybrid", hybrid_search_fn)]:
            result = fn(pq)
            top_entry = result[0]["metadata"].get("entry_name", "unknown") if result else "none"
            print(f"  {name:10s} top result: {top_entry}")