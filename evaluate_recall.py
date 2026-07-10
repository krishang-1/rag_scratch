"""
Phase 6: Retrieval evaluation - source-level AND entry-level recall
against the hand-curated, verified eval set.

Source-level recall: did the correct DOCUMENT appear in top-k?
Entry-level recall: did the correct SPECIFIC ENTRY (e.g. "lru_cache",
not just "functools.html") appear in top-k? This is the stricter,
more honest metric - especially important for large documents
(datetime.html, pathlib.html) where source-level recall alone is a
much weaker guarantee than it sounds.

Supports multi-valid ground truth (comma-separated expected_entry,
e.g. "lru_cache,cache") for questions with more than one correct answer.
"""

import sys
sys.path.insert(0, ".")
from build_pipeline_v2 import build_pipeline, dedupe_by_parent
from eval_questions import EVAL_QUESTIONS


def evaluate_recall(store, embed_model, questions: list, k_values: list = [1, 3, 5]):
    results = []

    for q in questions:
        query_vec = embed_model.embed_one(q["query"])
        raw_results = store.search(query_vec, top_k=max(k_values) * 2)
        deduped = dedupe_by_parent(raw_results)

        retrieved_sources = [r["metadata"]["source"] for r in deduped]
        retrieved_entries = [r["metadata"].get("entry_name", "unknown") for r in deduped]
        top_score = deduped[0]["score"] if deduped else 0.0

        row = {
            "query": q["query"],
            "category": q["category"],
            "answerable": q["answerable"],
            "expected_source": q["expected_source"],
            "expected_entry": q.get("expected_entry"),
            "top_score": top_score,
            "retrieved_sources_ordered": retrieved_sources[:max(k_values)],
            "retrieved_entries_ordered": retrieved_entries[:max(k_values)],
        }

        if q["answerable"]:
            expected_entries = set((q.get("expected_entry") or "").split(","))
            expected_entries.discard("")  # in case expected_entry was empty/None

            for k in k_values:
                top_k_sources = retrieved_sources[:k]
                top_k_entries = set(retrieved_entries[:k])

                source_hit = q["expected_source"] in top_k_sources
                entry_hit = bool(expected_entries & top_k_entries) if expected_entries else None

                row[f"recall@{k}_source"] = source_hit
                row[f"recall@{k}_entry"] = entry_hit
        else:
            for k in k_values:
                row[f"recall@{k}_source"] = None
                row[f"recall@{k}_entry"] = None

        results.append(row)

    return results


def summarize(results: list, k_values: list = [1, 3, 5]):
    print("=" * 70)
    print("RECALL@K SUMMARY (answerable questions only)")
    print("=" * 70)

    answerable = [r for r in results if r["answerable"]]

    print(f"\n{'k':<5}{'Source-level recall':<25}{'Entry-level recall':<25}")
    for k in k_values:
        source_hits = sum(1 for r in answerable if r[f"recall@{k}_source"])
        entry_hits = sum(1 for r in answerable if r[f"recall@{k}_entry"])
        total = len(answerable)
        print(f"{k:<5}{f'{source_hits}/{total} = {source_hits/total*100:.1f}%':<25}"
              f"{f'{entry_hits}/{total} = {entry_hits/total*100:.1f}%':<25}")

    print("\n" + "=" * 70)
    print("BY CATEGORY (recall@3, entry-level)")
    print("=" * 70)
    categories = set(r["category"] for r in answerable)
    for cat in sorted(categories):
        cat_results = [r for r in answerable if r["category"] == cat]
        entry_hits = sum(1 for r in cat_results if r["recall@3_entry"])
        print(f"  {cat}: {entry_hits}/{len(cat_results)} at recall@3 (entry-level)")

    print("\n" + "=" * 70)
    print("QUESTIONS WHERE SOURCE-LEVEL PASSED BUT ENTRY-LEVEL FAILED")
    print("=" * 70)
    print("(These would have shown as false '100% recall' under the old,")
    print(" looser metric - this is the gap the entry-level fix closes)")
    any_gap = False
    for r in answerable:
        if r["recall@3_source"] and not r["recall@3_entry"]:
            any_gap = True
            print(f"  \"{r['query'][:60]}...\"")
            print(f"    expected_entry: {r['expected_entry']} | "
                  f"retrieved_entries: {r['retrieved_entries_ordered']}")
    if not any_gap:
        print("  None - entry-level recall matches source-level recall exactly.")

    print("\n" + "=" * 70)
    print("NOT-IN-CORPUS QUESTIONS (checking honesty, not recall)")
    print("=" * 70)
    not_in_corpus = [r for r in results if not r["answerable"]]
    for r in not_in_corpus:
        print(f"  \"{r['query'][:50]}...\" -> top score: {r['top_score']:.4f}")
    print("\n  (Low scores here are GOOD - means nothing falsely matched.")
    print("   High scores would indicate the retriever might mislead")
    print("   the generator into hallucinating on unanswerable questions.)")


if __name__ == "__main__":
    store, embed_model = build_pipeline()
    results = evaluate_recall(store, embed_model, EVAL_QUESTIONS)
    summarize(results)

    import json
    with open("outputs/recall_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to outputs/recall_eval_results.json")