"""
logging_utils.py - structured, persistent logging for the RAG pipeline.

Tracks per-query latency, retrieval scores, and faithfulness
disagreement rate over time - the observability signals a production
system needs to detect regressions or drift.
"""

import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs/query_log.jsonl")


def log_query_event(
    query: str,
    retrieval_latency_s: float,
    generation_latency_s: float,
    top_retrieval_score: float,
    self_report_disagreement: bool,
    had_api_error: bool
):
    """Appends one structured log line (JSON Lines format) per query handled."""
    LOG_FILE.parent.mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieval_latency_s": round(retrieval_latency_s, 3),
        "generation_latency_s": round(generation_latency_s, 3),
        "total_latency_s": round(retrieval_latency_s + generation_latency_s, 3),
        "top_retrieval_score": round(top_retrieval_score, 4),
        "self_report_disagreement": self_report_disagreement,
        "had_api_error": had_api_error
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def summarize_logs():
    """Reads the full log and prints aggregate statistics."""
    if not LOG_FILE.exists():
        print("No logs yet.")
        return

    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        print("Log file is empty.")
        return

    total = len(entries)
    avg_total_latency = sum(e["total_latency_s"] for e in entries) / total
    disagreement_rate = sum(1 for e in entries if e["self_report_disagreement"]) / total
    error_rate = sum(1 for e in entries if e["had_api_error"]) / total

    print(f"Total queries logged: {total}")
    print(f"Average total latency: {avg_total_latency:.3f}s")
    print(f"Self-report disagreement rate: {disagreement_rate*100:.1f}%")
    print(f"API error rate: {error_rate*100:.1f}%")


if __name__ == "__main__":
    summarize_logs()