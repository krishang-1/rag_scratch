"""
Phase 3B: FAISS-based vector search - same operation as Phase 3A,
swapped in for comparison. FAISS uses approximate/optimized search
under the hood; at small scale it should return identical top-k
results to the brute-force version.
"""

import numpy as np
import faiss


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # IndexFlatIP = exact search via inner product (dot product).
        # Since we normalize vectors first, inner product == cosine similarity.
        # This is FAISS's "exact" index - genuinely comparable to brute-force,
        # not yet using approximate methods like HNSW (that's the next step
        # if we wanted to test the accuracy/speed tradeoff explicitly).
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []
        self.metadata = []

    def add(self, vectors: np.ndarray, chunks: list, metadata: list):
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.index.add(vectors_norm.astype(np.float32))
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list:
        query_norm = query_vector / np.linalg.norm(query_vector)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)

        scores, indices = self.index.search(query_norm, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "metadata": self.metadata[idx]
            })
        return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from embedding import EmbeddingModel
    from vector_store import NaiveVectorStore
    import time

    model = EmbeddingModel()

    sample_chunks = [
        "The lru_cache decorator caches function return values.",
        "Python's asyncio module handles asynchronous programming.",
        "The weather today is sunny with clear skies.",
        "functools.reduce applies a function cumulatively to items.",
    ]

    vectors = model.embed(sample_chunks)
    metadata = [{"source": "test.txt"} for _ in sample_chunks]

    query = "How do I cache function results in Python?"
    query_vec = model.embed_one(query)

    # --- Naive (Phase 3A) ---
    naive_store = NaiveVectorStore()
    naive_store.add(vectors, sample_chunks, metadata)
    naive_results = naive_store.search(query_vec, top_k=2)

    # --- FAISS (Phase 3B) ---
    faiss_store = FaissVectorStore(dim=model.dim)
    faiss_store.add(vectors, sample_chunks, metadata)
    faiss_results = faiss_store.search(query_vec, top_k=2)

    print("=== Naive (brute-force) results ===")
    for r in naive_results:
        print(f"  Score: {r['score']:.4f} | {r['chunk']}")

    print("\n=== FAISS results ===")
    for r in faiss_results:
        print(f"  Score: {r['score']:.4f} | {r['chunk']}")

    # Verify they match
    naive_top_chunk = naive_results[0]["chunk"]
    faiss_top_chunk = faiss_results[0]["chunk"]
    print(f"\nTop result matches: {naive_top_chunk == faiss_top_chunk}")