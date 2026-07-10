"""
Full RAG retrieval pipeline with parent-document retrieval and
INDEX PERSISTENCE: the chunk/embed/index steps only run when the
source documents in data/raw/ have actually changed (detected via
content hash), rather than on every single script invocation.
"""

import sys
import hashlib
from pathlib import Path

sys.path.insert(0, "modules")

from dom_chunking import load_and_chunk_all
from chunking import fixed_size_chunk
from embedding import EmbeddingModel
from vector_store_faiss import FaissVectorStore

MAX_CHARS_SAFE = 900
INDEX_CACHE_DIR = "index_cache"


def compute_corpus_hash(raw_dir: str = "data/raw") -> str:
    """
    Hashes the combined content of every .html file in raw_dir, sorted
    by filename for determinism. If any file changes, is added, or is
    removed, this hash changes - which is how we detect the cached
    index is stale and needs rebuilding.
    """
    hasher = hashlib.sha256()
    html_files = sorted(Path(raw_dir).glob("*.html"))
    for file_path in html_files:
        hasher.update(file_path.name.encode())
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def cap_oversized_chunks_with_parent(chunk_records: list, max_chars: int = MAX_CHARS_SAFE) -> list:
    """Every sub-chunk carries BOTH its small embedding-friendly text
    AND the full parent text, tagged with a shared parent_id and the
    original entry_name."""
    capped = []
    for parent_id, record in enumerate(chunk_records):
        chunk = record["chunk"]
        entry_name = record.get("entry_name", "unknown")
        if len(chunk) <= max_chars:
            capped.append({
                **record,
                "parent_id": parent_id,
                "parent_text": chunk,
                "entry_name": entry_name
            })
        else:
            sub_chunks = fixed_size_chunk(chunk, chunk_size=max_chars, overlap=80)
            for j, sub in enumerate(sub_chunks):
                capped.append({
                    "source": record["source"],
                    "chunk": sub,
                    "parent_text": chunk,
                    "parent_id": parent_id,
                    "chunk_index": f"{record['chunk_index']}.{j}",
                    "entry_name": entry_name
                })
    return capped


def dedupe_by_parent(results: list) -> list:
    """Multiple sub-chunks from the same parent might all rank highly -
    collapse to one entry per unique parent, keeping the best-ranked occurrence."""
    seen_parents = set()
    deduped = []
    for r in results:
        pid = r["metadata"]["parent_id"]
        if pid not in seen_parents:
            seen_parents.add(pid)
            deduped.append(r)
    return deduped


def _build_fresh(raw_dir: str = "data/raw"):
    """Runs the full chunk -> cap -> embed -> index pipeline from scratch."""
    print("Step 1: DOM-chunking real documentation...")
    raw_records = load_and_chunk_all(raw_dir)
    print(f"  {len(raw_records)} raw chunks\n")

    print("Step 2: Capping oversized chunks (tracking parent text + entry names)...")
    records = cap_oversized_chunks_with_parent(raw_records)
    print(f"  {len(records)} sub-chunks after capping\n")

    print("Step 3: Embedding sub-chunks...")
    model = EmbeddingModel()
    chunks_text = [r["chunk"] for r in records]
    vectors = model.embed(chunks_text)
    print(f"  Embedded {len(chunks_text)} chunks, shape {vectors.shape}\n")

    print("Step 4: Building FAISS index...")
    store = FaissVectorStore(dim=model.dim)
    metadata = [
        {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "parent_id": r["parent_id"],
            "entry_name": r.get("entry_name", "unknown")
        }
        for r in records
    ]
    parent_texts = [r["parent_text"] for r in records]
    store.add(vectors, parent_texts, metadata)
    print(f"  Index built with {len(chunks_text)} vectors\n")

    return store, model


def build_pipeline(raw_dir: str = "data/raw", force_rebuild: bool = False):
    """
    Returns (store, embed_model). Loads from index_cache/ if a valid,
    up-to-date cache exists; otherwise rebuilds from scratch and saves
    the new result for next time.
    """
    current_hash = compute_corpus_hash(raw_dir)
    hash_file = Path(INDEX_CACHE_DIR) / "corpus_hash.txt"

    if not force_rebuild and hash_file.exists():
        saved_hash = hash_file.read_text().strip()
        if saved_hash == current_hash:
            cached_store = FaissVectorStore.load(INDEX_CACHE_DIR)
            if cached_store is not None:
                print(f"Loaded cached index ({len(cached_store.chunks)} vectors) - "
                      f"corpus unchanged, skipping rebuild.\n")
                model = EmbeddingModel()  # still need the model loaded for query embedding
                return cached_store, model
        else:
            print("Corpus has changed since last cached index - rebuilding.\n")

    store, model = _build_fresh(raw_dir)

    Path(INDEX_CACHE_DIR).mkdir(exist_ok=True)
    store.save(INDEX_CACHE_DIR)
    (Path(INDEX_CACHE_DIR) / "corpus_hash.txt").write_text(current_hash)
    print(f"Saved index to {INDEX_CACHE_DIR}/ for future runs.\n")

    return store, model


if __name__ == "__main__":
    store, model = build_pipeline()

    test_query = "How does lru_cache decide what to keep in the cache?"
    query_vec = model.embed_one(test_query)
    raw_results = store.search(query_vec, top_k=6)
    results = dedupe_by_parent(raw_results)

    print(f"Query: {test_query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f} | {r['metadata']['source']} "
              f"(entry: {r['metadata']['entry_name']}, parent {r['metadata']['parent_id']})")
        print(f"    {r['chunk'][:250]}\n")