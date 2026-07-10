import sys
sys.path.insert(0, ".")
from build_pipeline_v2 import build_pipeline
from bm25_store import BM25Store

store, _ = build_pipeline()
bm25 = BM25Store(min_content_tokens=3)
bm25.add(store.chunks, store.metadata)

raw_results = bm25.search("How does caching work?", top_k=15)

seen_parents = set()
deduped = []
for r in raw_results:
    pid = r["metadata"]["parent_id"]
    if pid not in seen_parents:
        seen_parents.add(pid)
        deduped.append(r)

for i, r in enumerate(deduped[:6], 1):
    print(f"[{i}] entry: {r['metadata'].get('entry_name')} | score: {r['score']:.4f}")