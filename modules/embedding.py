"""
Phase 2: Embedding generation.

Uses all-MiniLM-L6-v2 - a small, free, well-tested sentence embedding
model, trained specifically for semantic similarity (a different
objective from a generative LLM's next-token prediction).
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dim}")

    def embed(self, texts: list) -> np.ndarray:
        """
        Args:
            texts: list of strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience method for a single string, returns shape (dim,)."""
        return self.embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors - the standard retrieval
    similarity metric: dot product normalized by both vectors' magnitudes."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    model = EmbeddingModel()

    # Sanity check (Phase 2 done-criterion): semantically similar
    # sentences should have higher similarity than unrelated ones
    sentence_a = "The lru_cache decorator caches function results for speed."
    sentence_b = "Using lru_cache can make repeated function calls faster."
    sentence_c = "The weather today is sunny with a chance of rain."

    emb_a = model.embed_one(sentence_a)
    emb_b = model.embed_one(sentence_b)
    emb_c = model.embed_one(sentence_c)

    sim_ab = cosine_similarity(emb_a, emb_b)  # should be HIGH (similar meaning)
    sim_ac = cosine_similarity(emb_a, emb_c)  # should be LOW (unrelated)

    print(f"\nSimilarity (related sentences):   {sim_ab:.4f}")
    print(f"Similarity (unrelated sentences): {sim_ac:.4f}")

    assert sim_ab > sim_ac, "Sanity check FAILED: related sentences should be more similar!"
    print("\nSanity check PASSED: related sentences score higher than unrelated ones.")