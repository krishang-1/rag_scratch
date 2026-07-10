"""
BM25 retrieval - a separate, keyword/term-frequency-based retrieval
method, used here specifically to compare against the dense embedding
approach on the same corpus and eval set.

Unlike dense embeddings (which measure semantic similarity via vector
distance), BM25 scores documents by how often query terms literally
appear, weighted by term rarity - closer to classic keyword search.

Stopword removal is applied CONDITIONALLY - only when enough content
tokens remain afterward. Short/vague queries (e.g. "How does caching
work?") lose too much signal if stopwords are stripped unconditionally,
since only 1-2 content words may remain - not enough for reliable
matching. Longer, content-word-dense queries benefit from stopword
removal since common words dilute term-frequency scoring across the
whole corpus. This was diagnosed empirically: unconditional stopword
removal fixed two adversarial queries but regressed one short, vague
query; conditional removal (this version) preserves both fixes without
the regression.
"""

from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "how", "what", "when", "where", "why", "does", "do", "did", "can",
    "could", "would", "should", "i", "you", "it", "this", "that", "with",
    "for", "of", "in", "on", "to", "and", "or", "but", "as", "by", "at",
    "from", "into", "about", "same", "have", "has", "had"
}


class BM25Store:
    def __init__(self, min_content_tokens: int = 3):
        """
        Args:
            min_content_tokens: minimum number of tokens that must remain
                after stopword removal for the filtered version to be used.
                If removal would leave fewer tokens than this, the original
                (unfiltered) tokens are used instead.
        """
        self.chunks = []
        self.metadata = []
        self.bm25 = None
        self._tokenized_corpus = []
        self.min_content_tokens = min_content_tokens

    def _tokenize(self, text: str) -> list:
        """
        Removes stopwords, but only if enough content tokens remain
        afterward - very short/vague queries lose too much signal from
        stopword removal (e.g. "How does caching work?" -> just "caching
        work" after stripping "how"/"does", too little left to match
        reliably). Falls back to keeping stopwords if removal would leave
        fewer than min_content_tokens.
        """
        tokens = text.lower().split()
        filtered = [t for t in tokens if t not in STOPWORDS]

        if len(filtered) < self.min_content_tokens:
            return tokens  # not enough signal left - keep original tokens instead

        return filtered

    def add(self, chunks: list, metadata: list):
        """
        Args:
            chunks: list of chunk text strings (the parent_text values,
                    same as what's stored in the FAISS index)
            metadata: list of metadata dicts, same shape as FaissVectorStore
        """
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        self._tokenized_corpus = [self._tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 6) -> list:
        """
        Args:
            query: raw query STRING (not a vector - BM25 works on text directly)
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })
        return results