"""
FAISS-based vector search - exact search via inner product (cosine
similarity, since vectors are normalized before indexing).

Includes save/load support so the index doesn't need to be rebuilt
from scratch on every script run.
"""

import pickle
from pathlib import Path
import numpy as np
import faiss


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
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
            if idx == -1:  # FAISS returns -1 for unfilled slots if fewer than top_k exist
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "metadata": self.metadata[idx]
            })
        return results

    def save(self, save_dir: str = "index_cache"):
        """Persists the FAISS index and parallel chunks/metadata lists to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        faiss.write_index(self.index, str(save_path / "index.faiss"))

        with open(save_path / "chunks_metadata.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata,
                "dim": self.dim
            }, f)

    @classmethod
    def load(cls, save_dir: str = "index_cache"):
        """Loads a previously saved FAISS index and chunks/metadata. Returns None if not found."""
        save_path = Path(save_dir)
        index_file = save_path / "index.faiss"
        pkl_file = save_path / "chunks_metadata.pkl"

        if not index_file.exists() or not pkl_file.exists():
            return None

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        store = cls(dim=data["dim"])
        store.index = faiss.read_index(str(index_file))
        store.chunks = data["chunks"]
        store.metadata = data["metadata"]

        return store