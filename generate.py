"""
Phase 5: Generation via Groq.

The LLM itself is instructed (and structurally constrained via
response_format) to return JSON directly - not a plain-text answer
that we wrap into JSON afterward. This is genuine structured output
prompting (Concept 206), not string formatting.

Includes:
- Retry logic with exponential backoff for transient API failures
- Graceful degradation (structured error result, not a crash) if
  retries are exhausted
- Concurrency-safe output filenames (microsecond precision + UUID,
  preventing collisions if two requests complete in the same second)
- Structured logging of latency, retrieval scores, and disagreement
  rate for ongoing observability
"""

import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from groq import Groq, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from dotenv import load_dotenv

from logging_utils import log_query_event

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2


def _call_groq_with_retry(messages: list, model: str, temperature: float, max_tokens: int) -> dict:
    """
    Wraps a Groq chat completion call with retry logic and exponential
    backoff, specifically for transient failures - rate limits, timeouts,
    and connection errors. Non-transient errors (e.g. auth failures,
    invalid requests) are NOT retried, since retrying those would just
    waste time on a failure that will never succeed.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return {"success": True, "content": response.choices[0].message.content}

        except RateLimitError as e:
            last_error = f"Rate limited by Groq: {e}"
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  [Retry {attempt}/{MAX_RETRIES}] Rate limited, waiting {wait}s before retry...")
            time.sleep(wait)

        except APITimeoutError as e:
            last_error = f"Groq request timed out: {e}"
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  [Retry {attempt}/{MAX_RETRIES}] Timeout, waiting {wait}s before retry...")
            time.sleep(wait)

        except APIConnectionError as e:
            last_error = f"Connection error reaching Groq: {e}"
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  [Retry {attempt}/{MAX_RETRIES}] Connection error, waiting {wait}s before retry...")
            time.sleep(wait)

        except APIStatusError as e:
            if e.status_code and 500 <= e.status_code < 600:
                last_error = f"Groq server error ({e.status_code}): {e}"
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Server error {e.status_code}, waiting {wait}s...")
                time.sleep(wait)
            else:
                return {"success": False, "error": f"Groq client error ({e.status_code}): {e}"}

    return {"success": False, "error": f"Exhausted {MAX_RETRIES} retries. Last error: {last_error}"}


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
    Returns the model's own structured JSON output as a parsed dict.

    If the API call fails after all retries, returns a graceful
    fallback dict rather than raising an exception.
    """
    prompt = build_prompt(query, retrieved_chunks)
    messages = [{"role": "user", "content": prompt}]

    result = _call_groq_with_retry(messages, model=model, temperature=0.3, max_tokens=400)

    if not result["success"]:
        return {
            "answer": "Unable to generate an answer - the generation service is currently unavailable.",
            "context_sufficient": None,
            "sources_referenced": [],
            "_api_error": result["error"]
        }

    raw_content = result["content"]

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
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
    call. Fails SAFE if the API call itself fails: treats unverifiable
    as faithful=False, triggering the override rather than risking an
    unverified answer reaching the user.
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

    messages = [{"role": "user", "content": verify_prompt}]
    result = _call_groq_with_retry(messages, model=model, temperature=0.0, max_tokens=200)

    if not result["success"]:
        return {
            "faithful": False,
            "evidence": "Verification unavailable - failing safe (treating as unverified).",
            "_api_error": result["error"]
        }

    return json.loads(result["content"])


def run_and_save(
    query: str,
    retrieved_chunks: list,
    model: str = "llama-3.1-8b-instant",
    output_dir: str = "outputs"
) -> dict:
    """
    Runs generation, independent faithfulness verification (skipped
    when self-report already refused), logs the query event, and
    saves a structured JSON result with a concurrency-safe filename.
    """
    gen_start = time.time()
    model_output = generate_answer(query, retrieved_chunks, model=model)
    gen_latency = time.time() - gen_start

    self_reported_sufficient = model_output.get("context_sufficient")

    verify_start = time.time()
    if self_reported_sufficient is False:
        faithfulness_check = {
            "faithful": True,
            "evidence": "Generator already reported insufficient context; no substantive claim was made to verify."
        }
        disagreement = False
    else:
        faithfulness_check = verify_faithfulness(
            model_output.get("answer", ""), retrieved_chunks, model=model
        )
        is_faithful_check = faithfulness_check.get("faithful", False)
        disagreement = (self_reported_sufficient != is_faithful_check)
    verify_latency = time.time() - verify_start

    is_faithful = faithfulness_check.get("faithful", False)

    if is_faithful:
        final_answer = model_output.get("answer")
    else:
        final_answer = ("The retrieved context does not contain sufficient information "
                         "to answer this question. (Independent verification overrode the "
                         "generator's own claim of sufficiency.)")

    had_api_error = "_api_error" in model_output or "_api_error" in faithfulness_check
    top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

    # NOTE: retrieval happens before this function is called (in the
    # caller's code - test scripts, app.py), so retrieval_latency_s is
    # not measured here. Stated explicitly rather than faking a number.
    log_query_event(
        query=query,
        retrieval_latency_s=0.0,
        generation_latency_s=gen_latency + verify_latency,
        top_retrieval_score=top_score,
        self_report_disagreement=disagreement,
        had_api_error=had_api_error
    )

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
        "self_report_disagreement": disagreement,
        "had_api_error": had_api_error
    }

    Path(output_dir).mkdir(exist_ok=True)
    safe_query_slug = "".join(c if c.isalnum() else "_" for c in query[:40]).strip("_")

    # Microsecond precision + random suffix - eliminates filename
    # collision risk if two requests complete within the same second.
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_suffix = uuid.uuid4().hex[:6]
    output_path = Path(output_dir) / f"{timestamp_slug}_{unique_suffix}_{safe_query_slug}.json"

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
    if "_api_error" in result["model_response"]:
        print(f"API ERROR encountered: {result['model_response']['_api_error']}")

    print("\n" + "=" * 60)
    print("INDEPENDENT FAITHFULNESS CHECK (authoritative)")
    print("=" * 60)
    fc = result["faithfulness_check"]
    print(f"Faithful (verified separately): {fc.get('faithful')}")
    print(f"Evidence: {fc.get('evidence')}")
    if "_api_error" in fc:
        print(f"API ERROR encountered: {fc['_api_error']}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER (after override, this is what a user would see)")
    print("=" * 60)
    print(result.get("final_answer", "(final_answer not computed - check run_and_save)"))
    print(f"\nSelf-report disagreed with independent check: {result.get('self_report_disagreement')}")
    print(f"Had API error during this run: {result.get('had_api_error')}")

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