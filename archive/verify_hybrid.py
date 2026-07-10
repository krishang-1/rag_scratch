import sys
sys.path.insert(0, ".")
from build_pipeline_v2 import build_pipeline, dedupe_by_parent
from bm25_store import BM25Store
from hybrid_store import hybrid_search

store, embed_model = build_pipeline()
bm25 = BM25Store(min_content_tokens=3)
bm25.add(store.chunks, store.metadata)

test_queries = [
    "How can I avoid recomputing an expensive function call with the same inputs?",  # BM25 fixed this
    "How does partial relate to the args and keywords attributes?",                   # BM25 fixed this
    "How does caching work?"                                                          # dense handles this, BM25 can't
]

for query in test_queries:
    query_vec = embed_model.embed_one(query)
    dense_raw = store.search(query_vec, top_k=12)
    bm25_raw = bm25.search(query, top_k=12)
    result = hybrid_search(dense_raw, bm25_raw, alpha=0.5, top_k=3)

    print(f"\nQuery: {query}")
    for i, r in enumerate(result, 1):
        print(f"  [{i}] {r['metadata'].get('entry_name')} (dense={r['dense_score']:.3f}, bm25={r['bm25_score']:.3f})")