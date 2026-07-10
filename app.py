"""
Streamlit interface for the RAG pipeline.

The expensive part (DOM-chunking, embedding, FAISS indexing) runs
ONCE via st.cache_resource, then every query just reuses the cached
index - no re-chunking or re-embedding on each interaction.

Every query now produces BOTH:
  - A human-readable text display in the UI
  - A downloadable/saved JSON file (via generate.py's run_and_save),
    containing the model's own structured response plus retrieval metadata
"""

import streamlit as st
import json
import sys
sys.path.insert(0, ".")

from build_pipeline_v2 import build_pipeline, dedupe_by_parent
from generate import run_and_save


@st.cache_resource
def get_pipeline():
    """Runs once per app session. Cached across every subsequent query."""
    store, model = build_pipeline()
    return store, model


st.set_page_config(page_title="RAG over Python Docs", layout="wide")
st.title("RAG System — Python Standard Library Documentation")
st.caption("Retrieval over functools, itertools, and collections. Generation via Groq (Llama 3.1 8B), structured JSON output.")

with st.spinner("Building index (chunking, embedding, FAISS)... this only happens once."):
    store, embed_model = get_pipeline()

st.success("Index ready. Ask a question below.")

query = st.text_input(
    "Ask a question about functools, itertools, or collections:",
    placeholder="How does lru_cache decide what to keep in the cache?"
)

top_k = st.slider("Number of source chunks to retrieve", min_value=1, max_value=6, value=3)

if st.button("Generate Answer", type="primary") and query:
    with st.spinner("Retrieving relevant context..."):
        query_vec = embed_model.embed_one(query)
        raw_results = store.search(query_vec, top_k=top_k * 2)  # over-fetch, then dedupe
        retrieved = dedupe_by_parent(raw_results)[:top_k]

    with st.spinner("Generating structured answer via Groq..."):
        result = run_and_save(query, retrieved)

    model_response = result["model_response"]

    # --- Human-readable text display ---
    st.subheader("Answer")
    st.write(model_response.get("answer", "(no answer returned)"))

    col1, col2 = st.columns(2)
    with col1:
        sufficient = model_response.get("context_sufficient")
        if sufficient is True:
            st.success("Model reports context was sufficient to answer.")
        elif sufficient is False:
            st.warning("Model reports context was NOT sufficient to answer.")
        else:
            st.info("Context sufficiency not reported.")
    with col2:
        st.write(f"**Sources referenced by model:** {model_response.get('sources_referenced', [])}")

    if "_parse_error" in model_response:
        st.error(f"Note: {model_response['_parse_error']}")

    st.subheader("Retrieved Sources")
    for i, r in enumerate(retrieved, 1):
        with st.expander(f"[{i}] {r['metadata']['source']} — score: {r['score']:.4f}"):
            st.text(r["chunk"][:1000])

    st.subheader("Structured JSON Output")

    result_for_display = {k: v for k, v in result.items() if k != "_saved_to"}
    json_str = json.dumps(result_for_display, indent=2, ensure_ascii=False)

    with st.expander("View raw JSON"):
        st.json(result_for_display)

    st.download_button(
        label="Download this result as .json",
        data=json_str,
        file_name=f"rag_result_{result['timestamp'].replace(':', '-')}.json",
        mime="application/json"
    )

    st.caption(f"Also saved to disk at: {result.get('_saved_to', 'N/A')}")