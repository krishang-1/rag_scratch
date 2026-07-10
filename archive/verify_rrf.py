import sys
sys.path.insert(0, ".")
from build_pipeline_v2 import build_pipeline
from bm25_store import BM25Store
from hybrid_store import reciprocal_rank_fusion

store, embed_model = build_pipeline()
bm25 = BM25Store(min_content_tokens=3)
bm25.add(store.chunks, store.metadata)

test_queries = [
    "How can I avoid recomputing an expensive function call with the same inputs?",
    "How does partial relate to the args and keywords attributes?",
    "How does caching work?"
]

for query in test_queries:
    query_vec = embed_model.embed_one(query)
    dense_raw = store.search(query_vec, top_k=12)
    bm25_raw = bm25.search(query, top_k=12)
    result = reciprocal_rank_fusion(dense_raw, bm25_raw, top_k=3)

    print(f"\nQuery: {query}")
    for i, r in enumerate(result, 1):
        print(f"  [{i}] {r['metadata'].get('entry_name')} (RRF score={r['score']:.4f})")