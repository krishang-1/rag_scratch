import sys
sys.path.insert(0, ".")
from build_pipeline_v2 import build_pipeline, dedupe_by_parent

store, embed_model = build_pipeline()

query = "What is the syntax for a Python lambda function?"
query_vec = embed_model.embed_one(query)
results = dedupe_by_parent(store.search(query_vec, top_k=5))

for i, r in enumerate(results, 1):
    print(f"[{i}] Score: {r['score']:.4f} | {r['metadata']['source']}")
    print(f"    {r['chunk'][:200]}\n")