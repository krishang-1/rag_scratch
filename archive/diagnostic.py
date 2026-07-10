import sys
sys.path.insert(0, ".")
from build_pipeline_v2 import build_pipeline
from bm25_store import BM25Store

store, _ = build_pipeline()
bm25 = BM25Store(min_content_tokens=3)
bm25.add(store.chunks, store.metadata)

results = bm25.search("How does caching work?", top_k=6)
for i, r in enumerate(results, 1):
    print(f"[{i}] entry: {r['metadata'].get('entry_name')} | score: {r['score']:.4f}")