"""
Batch faithfulness verification across the full eval set.

Tests whether the independent verify_faithfulness() check ever
disagrees with self-reported sufficiency on the ANSWERABLE questions -
specifically checking for FALSE POSITIVES (the override incorrectly
rejecting a genuinely correct, well-grounded answer), which would be
a real cost of the safety mechanism we haven't measured yet.
"""

import sys
import json
sys.path.insert(0, ".")

from build_pipeline_v2 import build_pipeline, dedupe_by_parent
from generate import run_and_save
from eval_questions import EVAL_QUESTIONS


def run_batch_faithfulness_check(store, embed_model, questions: list):
    results = []

    for q in questions:
        query_vec = embed_model.embed_one(q["query"])
        raw_results = store.search(query_vec, top_k=6)
        retrieved = dedupe_by_parent(raw_results)[:3]

        result = run_and_save(q["query"], retrieved, output_dir="outputs/batch_faithfulness")

        row = {
            "query": q["query"],
            "category": q["category"],
            "answerable": q["answerable"],
            "self_reported_sufficient": result["model_response"].get("context_sufficient"),
            "independently_faithful": result["faithfulness_check"].get("faithful"),
            "disagreement": result.get("self_report_disagreement"),
            "final_answer": result.get("final_answer"),
        }
        results.append(row)
        print(f"  Processed: {q['query'][:50]}... -> disagreement={row['disagreement']}")

    return results


def summarize_batch(results: list):
    print("\n" + "=" * 70)
    print("BATCH FAITHFULNESS SUMMARY")
    print("=" * 70)

    answerable = [r for r in results if r["answerable"]]
    not_answerable = [r for r in results if not r["answerable"]]

    disagreements_on_answerable = [r for r in answerable if r["disagreement"]]
    disagreements_on_not_answerable = [r for r in not_answerable if r["disagreement"]]

    print(f"\nAnswerable questions: {len(answerable)}")
    print(f"  Disagreements (self-report vs independent check): {len(disagreements_on_answerable)}")
    if disagreements_on_answerable:
        print("  -> These are POTENTIAL FALSE POSITIVES (override may have rejected a correct answer):")
        for r in disagreements_on_answerable:
            print(f"     - \"{r['query'][:60]}...\"")
            print(f"       final_answer: {r['final_answer'][:150]}")

    print(f"\nNot-in-corpus questions: {len(not_answerable)}")
    print(f"  Disagreements: {len(disagreements_on_not_answerable)}")
    print("  (Disagreement here is FINE/expected - self-report might wrongly")
    print("   claim sufficiency, and we WANT the override to catch that)")

    print("\n" + "=" * 70)
    if len(disagreements_on_answerable) == 0:
        print("No false positives detected: override never triggered on a")
        print("genuinely answerable question in this eval set.")
    else:
        print(f"WARNING: {len(disagreements_on_answerable)} potential false positive(s) -")
        print("inspect the final_answer text above to confirm whether the")
        print("override incorrectly discarded a correct response.")
    print("=" * 70)


if __name__ == "__main__":
    store, embed_model = build_pipeline()
    results = run_batch_faithfulness_check(store, embed_model, EVAL_QUESTIONS)
    summarize_batch(results)

    with open("outputs/batch_faithfulness_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to outputs/batch_faithfulness_summary.json")