"""
Hand-curated evaluation set for the RAG pipeline.
Ground truth (expected_source) is the actual entry each question
should retrieve, verified by manual inspection of the real corpus.
"""

EVAL_QUESTIONS = [
    # --- Exact lookup ---
    {"query": "What is the default maxsize for lru_cache?", "category": "exact_lookup",
     "expected_source": "functools.html", "expected_entry": "lru_cache", "answerable": True},

    {"query": "What does itertools.chain do?", "category": "exact_lookup",
     "expected_source": "itertools.html", "expected_entry": "chain", "answerable": True},

    {"query": "What is a Counter in the collections module?", "category": "exact_lookup",
     "expected_source": "collections.html", "expected_entry": "Counter", "answerable": True},

    # --- Paraphrase (different wording than the doc) ---
    {"query": "How can I avoid recomputing an expensive function call with the same inputs?",
     "category": "paraphrase", "expected_source": "functools.html",
     "expected_entry": "lru_cache,cache",  # both valid - cache IS lru_cache(maxsize=None);
                                            # the question doesn't specify bounded vs unbounded,
                                            # so either answer is legitimately correct
     "answerable": True},

    {"query": "How do I combine multiple lists into one iterator without copying them?",
     "category": "paraphrase", "expected_source": "itertools.html",
     "expected_entry": "chain", "answerable": True},

    {"query": "What data structure automatically gives dictionary keys a default value?",
     "category": "paraphrase", "expected_source": "collections.html",
     "expected_entry": "defaultdict", "answerable": True},

    # --- Multi-concept (requires connecting related entries) ---
    {"query": "What's the difference between functools.cache and functools.lru_cache?",
     "category": "multi_concept", "expected_source": "functools.html",
     "expected_entry": "cache,lru_cache", "answerable": True},

    {"query": "How is itertools.groupby different from a regular for loop with a dictionary?",
     "category": "multi_concept", "expected_source": "itertools.html",
     "expected_entry": "groupby", "answerable": True},

    {"query": "How does partial relate to the args and keywords attributes?",
     "category": "multi_concept", "expected_source": "functools.html",
     "expected_entry": "partial", "answerable": True},

    # --- Not in corpus (should say "I don't know", not hallucinate) ---
    {"query": "How do I create a pandas DataFrame from a CSV file?",
     "category": "not_in_corpus", "expected_source": None,
     "expected_entry": None, "answerable": False},

    {"query": "What is the syntax for a Python lambda function?",
     "category": "not_in_corpus", "expected_source": None,
     "expected_entry": None, "answerable": False},

    {"query": "How does Python's asyncio event loop work?",
     "category": "not_in_corpus", "expected_source": None,
     "expected_entry": None, "answerable": False},

    # --- Ambiguous / adversarial ---
    {"query": "How does caching work?",
     "category": "ambiguous", "expected_source": "functools.html",
     "expected_entry": "lru_cache,cache,cached_property", "answerable": True},

    {"query": "What's the best data structure for counting things?",
     "category": "ambiguous", "expected_source": "collections.html",
     "expected_entry": "Counter", "answerable": True},

    {"query": "Tell me about tuples",
     "category": "ambiguous", "expected_source": "collections.html",
     "expected_entry": "namedtuple", "answerable": True},  # deliberately vague - namedtuple is the closest relevant entry
]