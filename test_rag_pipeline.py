"""
Automated test suite covering the core, previously-manually-verified
behaviors of the RAG pipeline: chunking correctness, parent-document
retrieval, the faithfulness override mechanism, and recall thresholds.

Run with: pytest test_rag_pipeline.py -v
"""

import sys
sys.path.insert(0, ".")

import pytest
from dom_chunking import dom_chunk_sphinx_docs
from build_pipeline_v2 import build_pipeline, dedupe_by_parent, cap_oversized_chunks_with_parent
from eval_questions import EVAL_QUESTIONS


# ============================================================
# FIXTURES - shared, expensive setup run once per test session
# ============================================================

@pytest.fixture(scope="session")
def pipeline():
    """Builds (or loads cached) pipeline once for all tests in this run."""
    store, embed_model = build_pipeline()
    return store, embed_model


# ============================================================
# CHUNKING TESTS - the DOM-boundary logic, verified against the
# realistic HTML sample built earlier in this project
# ============================================================

SAMPLE_HTML = """
<html><body>
<div role="main">
<h1>functools — Higher-order functions</h1>
<p>The functools module is for higher-order functions.</p>

<dl class="py function">
<dt id="functools.lru_cache">functools.lru_cache(maxsize=128)</dt>
<dd><p>Decorator to wrap a function with a memoizing callable.</p></dd>
</dl>

<dl class="py class">
<dt id="functools.partial">class functools.partial(func, /, *args, **keywords)</dt>
<dd><p>Return a new partial object.</p>
<dl class="py attribute">
<dt id="functools.partial.args">partial.args</dt>
<dd><p>The leftmost positional arguments.</p></dd>
</dl>
</dd>
</dl>
</div>
</body></html>
"""


def test_dom_chunking_produces_correct_entry_count():
    """Should produce exactly 3 chunks: intro, lru_cache, partial (with
    nested args attached, not split into a 4th chunk)."""
    chunks, entry_names = dom_chunk_sphinx_docs(SAMPLE_HTML)
    assert len(chunks) == 3
    assert "intro" in entry_names


def test_dom_chunking_keeps_nested_attributes_with_parent():
    """The nested partial.args entry should be INSIDE the partial chunk,
    not split out as a separate top-level chunk - this was the original
    bug (Concept: DOM boundary correctness) fixed early in this project."""
    chunks, entry_names = dom_chunk_sphinx_docs(SAMPLE_HTML)
    partial_chunk = next(c for c, name in zip(chunks, entry_names) if name == "partial")
    assert "leftmost positional arguments" in partial_chunk


def test_dom_chunking_extracts_correct_entry_names():
    _, entry_names = dom_chunk_sphinx_docs(SAMPLE_HTML)
    assert "lru_cache" in entry_names
    assert "partial" in entry_names


# ============================================================
# CAPPING/PARENT-DOCUMENT RETRIEVAL TESTS - the fix for oversized
# chunks getting silently truncated by the embedding model's 256
# token limit, and the parent-document retrieval pattern
# ============================================================

def test_oversized_chunk_gets_capped():
    """A chunk exceeding MAX_CHARS_SAFE should be split into sub-chunks,
    each individually under the embedding model's safe size."""
    long_record = [{"source": "test.html", "chunk": "x " * 1000, "chunk_index": 0, "entry_name": "test"}]
    capped = cap_oversized_chunks_with_parent(long_record, max_chars=900)
    assert len(capped) > 1
    assert all(len(c["chunk"]) <= 900 for c in capped)


def test_capped_subchunks_retain_full_parent_text():
    """Even when split for embedding, each sub-chunk must carry the FULL
    original text as parent_text - this is what lets retrieval surface
    complete context even if only a peripheral fragment matched the query."""
    long_text = "SIGNATURE " * 50 + "IMPORTANT_DETAIL " * 50 + "TRAILING " * 50
    long_record = [{"source": "test.html", "chunk": long_text, "chunk_index": 0, "entry_name": "test"}]
    capped = cap_oversized_chunks_with_parent(long_record, max_chars=900)

    # Even the FIRST sub-chunk (which won't itself contain "IMPORTANT_DETAIL")
    # should have parent_text containing it
    first_subchunk = capped[0]
    assert "IMPORTANT_DETAIL" in first_subchunk["parent_text"]


def test_normal_sized_chunk_not_split():
    """A chunk under the size limit should pass through unchanged, not
    get needlessly split."""
    short_record = [{"source": "test.html", "chunk": "short text", "chunk_index": 0, "entry_name": "test"}]
    capped = cap_oversized_chunks_with_parent(short_record, max_chars=900)
    assert len(capped) == 1
    assert capped[0]["chunk"] == "short text"


def test_dedupe_by_parent_removes_duplicates():
    """Multiple sub-chunks from the same parent ranking highly should
    collapse to one result, not show the same underlying entry twice."""
    fake_results = [
        {"metadata": {"parent_id": 5, "source": "a.html"}, "chunk": "x", "score": 0.9},
        {"metadata": {"parent_id": 5, "source": "a.html"}, "chunk": "y", "score": 0.8},
        {"metadata": {"parent_id": 7, "source": "b.html"}, "chunk": "z", "score": 0.7},
    ]
    deduped = dedupe_by_parent(fake_results)
    assert len(deduped) == 2
    assert deduped[0]["metadata"]["parent_id"] == 5  # keeps the FIRST (highest-ranked) occurrence


# ============================================================
# RETRIEVAL TESTS - using the real pipeline, checking that known-good
# queries still retrieve their correct entry (regression guard)
# ============================================================

def test_lru_cache_query_retrieves_correct_entry(pipeline):
    """The most basic, repeatedly-verified case throughout this project -
    if this ever breaks, something fundamental regressed."""
    store, embed_model = pipeline
    query_vec = embed_model.embed_one("How does lru_cache decide what to keep in the cache?")
    results = dedupe_by_parent(store.search(query_vec, top_k=6))[:3]

    entry_names = [r["metadata"].get("entry_name") for r in results]
    assert "lru_cache" in entry_names


@pytest.mark.parametrize("question", [q for q in EVAL_QUESTIONS if q["answerable"]])
def test_eval_set_recall_at_3(pipeline, question):
    """Regression guard for the full hand-curated eval set - every
    answerable question should find its expected entry within top-3,
    matching the 100% entry-level recall@3 result measured earlier."""
    store, embed_model = pipeline
    query_vec = embed_model.embed_one(question["query"])
    results = dedupe_by_parent(store.search(query_vec, top_k=12))[:3]

    expected_entries = set((question.get("expected_entry") or "").split(","))
    expected_entries.discard("")

    retrieved_entries = set(r["metadata"].get("entry_name") for r in results)
    assert expected_entries & retrieved_entries, (
        f"Expected one of {expected_entries}, got {retrieved_entries}"
    )


# ============================================================
# NOT-IN-CORPUS TESTS - unanswerable questions should NOT retrieve
# with high confidence (regression guard against false positives)
# ============================================================

@pytest.mark.parametrize("question", [q for q in EVAL_QUESTIONS if not q["answerable"]])
def test_not_in_corpus_questions_score_low(pipeline, question):
    """Unanswerable questions shouldn't match anything with high
    confidence. Threshold set generously above what we measured
    (0.30-0.55 range observed) to avoid false test failures from
    minor embedding variance, while still catching a genuine regression
    where an unrelated query suddenly scores very high."""
    store, embed_model = pipeline
    query_vec = embed_model.embed_one(question["query"])
    results = dedupe_by_parent(store.search(query_vec, top_k=3))

    top_score = results[0]["score"] if results else 0.0
    assert top_score < 0.75, f"Unexpectedly high confidence ({top_score}) on unanswerable question"