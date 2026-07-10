import sys
sys.path.insert(0, ".")
from build_pipeline_v2 import build_pipeline, dedupe_by_parent
from generate import run_and_save, print_result

store, embed_model = build_pipeline()

query = "What is the syntax for a Python lambda function?"
query_vec = embed_model.embed_one(query)
retrieved = dedupe_by_parent(store.search(query_vec, top_k=6))[:3]

result = run_and_save(query, retrieved)
print_result(result)