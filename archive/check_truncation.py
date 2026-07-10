from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Max sequence length: {model.max_seq_length}")

# Test against your actual longest chunks
import sys
sys.path.insert(0, "modules")
from dom_chunking import load_and_chunk_all

results = load_and_chunk_all()
longest = max(results, key=lambda r: len(r["chunk"]))
print(f"\nLongest chunk: {len(longest['chunk'])} chars, from {longest['source']}")

tokens = model.tokenizer(longest["chunk"])
token_count = len(tokens["input_ids"])
print(f"Token count: {token_count}")
print(f"Exceeds max_seq_length ({model.max_seq_length})? {token_count > model.max_seq_length}")