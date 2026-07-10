"""
Phase 5: Generation via Groq.

The LLM itself is instructed (and structurally constrained via
response_format) to return JSON directly - not a plain-text answer
that we wrap into JSON afterward. This is genuine structured output
prompting (Concept 206), not string formatting.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_prompt(query: str, retrieved_chunks: list) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"]["source"]
        context_blocks.append(f"[Source {i}: {source}]\n{chunk['chunk']}")

    context_text = "\n\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the context provided below.

Context:
{context_text}

Question: {query}

Before answering, evaluate the context carefully:
- A concept being MENTIONED or USED AS AN EXAMPLE inside unrelated documentation
  (e.g. a lambda expression appearing inside functools.reduce's example code)
  does NOT count as the context EXPLAINING that concept.
- Only set context_sufficient=true if the context actually DEFINES, DESCRIBES,
  or DOCUMENTS the specific thing the question asks about.

IMPORTANT: If context_sufficient is false, the "answer" field MUST explicitly
state that the retrieved context does not contain enough information to answer
the question - it must NOT contain a guessed or extracted answer, even if a
superficially relevant snippet exists in the context.

Respond with a JSON object matching EXACTLY this schema:
{{
  "answer": "<your answer if context_sufficient=true, OR an explicit statement like 'The retrieved context does not contain sufficient information to answer this question' if context_sufficient=false>",
  "context_sufficient": <true or false>,
  "relevance_reasoning": "<one sentence explaining WHY the context does or does not actually address the question>",
  "sources_referenced": [<list of integers actually used>]
}}

Return ONLY the JSON object, no other text before or after it."""

    return prompt

def generate_answer(query: str, retrieved_chunks: list, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Returns the model's own structured JSON output as a parsed dict,
    not a plain string.
    """
    prompt = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
        response_format={"type": "json_object"}  # structurally enforces valid JSON output
    )

    raw_content = response.choices[0].message.content

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        # Fallback if the model somehow still returns malformed JSON -
        # surface this as a real, visible failure rather than silently
        # hiding it behind a fabricated structure
        parsed = {
            "answer": raw_content,
            "context_sufficient": None,
            "sources_referenced": [],
            "_parse_error": "Model output was not valid JSON despite response_format constraint"
        }

    return parsed


def verify_faithfulness(answer: str, retrieved_chunks: list, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Independent faithfulness check - a SEPARATE call from the generation
    call, specifically asked to verify whether the answer's claims can
    be found in the context, without any framing pressure to produce
    a "real" answer. RAGAS-style separate-judge pattern.
    """
    context_text = "\n\n".join(c["chunk"] for c in retrieved_chunks)

    verify_prompt = f"""Here is a CONTEXT and an ANSWER that was generated from it.

Context:
{context_text}

Answer to verify: "{answer}"

Does the context ACTUALLY contain the specific information stated in the answer?
Be strict: a general fact being true in the real world does NOT count if the
context itself doesn't state it. Check word-for-word whether the answer's
claims are traceable to specific text in the context above.

Respond with JSON:
{{"faithful": <true or false>, "evidence": "<quote the exact context text that supports the answer, or state 'no supporting text found'>"}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": verify_prompt}],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def run_and_save(
    query: str,
    retrieved_chunks: list,
    model: str = "llama-3.1-8b-instant",
    output_dir: str = "outputs"
) -> dict:
    model_output = generate_answer(query, retrieved_chunks, model=model)

    # If the generator ITSELF already reported insufficient context,
    # there's no substantive claim to fact-check - running the
    # claim-verification prompt against a refusal produces awkward,
    # semantically-mismatched reasoning (as observed with the asyncio
    # case). Skip straight to a clean, honest "trivially faithful"
    # result, and treat this as NO disagreement, since self-report
    # and the final outcome already agree in this branch.
    self_reported_sufficient = model_output.get("context_sufficient")

    if self_reported_sufficient is False:
        faithfulness_check = {
            "faithful": True,
            "evidence": "Generator already reported insufficient context; no substantive claim was made to verify."
        }
        disagreement = False  # Nothing to disagree about - both already agree it's insufficient
    else:
        # This is the case that actually matters for catching hallucination -
        # generator claimed sufficiency, so verify that claim independently.
        faithfulness_check = verify_faithfulness(
            model_output.get("answer", ""), retrieved_chunks, model=model
        )
        is_faithful_check = faithfulness_check.get("faithful", False)
        disagreement = (self_reported_sufficient != is_faithful_check)

    is_faithful = faithfulness_check.get("faithful", False)

    if is_faithful:
        final_answer = model_output.get("answer")
    else:
        final_answer = ("The retrieved context does not contain sufficient information "
                         "to answer this question. (Independent verification overrode the "
                         "generator's own claim of sufficiency.)")

    result = {
        "query": query,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "retrieved_sources": [
            {
                "rank": i + 1,
                "source": chunk["metadata"]["source"],
                "parent_id": chunk["metadata"].get("parent_id"),
                "score": chunk["score"],
                "chunk_text": chunk["chunk"]
            }
            for i, chunk in enumerate(retrieved_chunks)
        ],
        "model_response": model_output,
        "faithfulness_check": faithfulness_check,
        "final_answer": final_answer,
        "self_report_disagreement": disagreement
    }

    Path(output_dir).mkdir(exist_ok=True)
    safe_query_slug = "".join(c if c.isalnum() else "_" for c in query[:40]).strip("_")
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{timestamp_slug}_{safe_query_slug}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["_saved_to"] = str(output_path)
    return result


def print_result(result: dict):
    print(f"Query: {result['query']}\n")

    print("Retrieved context sources:")
    for src in result["retrieved_sources"]:
        print(f"  [{src['rank']}] {src['source']} (parent {src['parent_id']}, score {src['score']:.4f})")

    print("\n" + "=" * 60)
    print("GENERATOR'S RAW SELF-REPORT (not fully trusted)")
    print("=" * 60)
    print(f"Answer: {result['model_response'].get('answer')}")
    print(f"Context sufficient (self-reported): {result['model_response'].get('context_sufficient')}")

    print("\n" + "=" * 60)
    print("INDEPENDENT FAITHFULNESS CHECK (authoritative)")
    print("=" * 60)
    fc = result["faithfulness_check"]
    print(f"Faithful (verified separately): {fc.get('faithful')}")
    print(f"Evidence: {fc.get('evidence')}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER (after override, this is what a user would see)")
    print("=" * 60)
    print(result.get("final_answer", "(final_answer not computed - check run_and_save)"))
    print(f"\nSelf-report disagreed with independent check: {result.get('self_report_disagreement')}")

    if "_saved_to" in result:
        print(f"\nSaved JSON result to: {result['_saved_to']}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from build_pipeline_v2 import build_pipeline, dedupe_by_parent

    store, embed_model = build_pipeline()

    test_query = "How does lru_cache decide what to keep in the cache?"
    query_vec = embed_model.embed_one(test_query)
    raw_results = store.search(query_vec, top_k=6)
    retrieved = dedupe_by_parent(raw_results)[:3]

    print("Generating structured answer via Groq...\n")
    result = run_and_save(test_query, retrieved)

    print_result(result)