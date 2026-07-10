"""
DOM-based chunking for Sphinx-generated docs. Operates on raw HTML
directly, using the real <dl class="py ..."> structural boundaries
Sphinx uses for each function/class/method/attribute, rather than
guessing structure from already-flattened text.

Also extracts each entry's fully-qualified name (from its <dt id="...">
attribute) so downstream evaluation can check entry-level recall, not
just source-file-level recall.
"""

from pathlib import Path
from bs4 import BeautifulSoup


def dom_chunk_sphinx_docs(html: str):
    """
    Returns (chunks, entry_names) - two parallel lists, same length,
    where entry_names[i] is the short entry name (e.g. "lru_cache")
    corresponding to chunks[i].
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    main = soup.find("div", {"role": "main"}) or soup.find("main") or soup

    chunks = []
    entry_names = []

    all_dls = main.find_all("dl", class_=lambda c: c and c.startswith("py"))

    for dl in all_dls:
        if dl.find_parent("dl", class_=lambda c: c and c.startswith("py")):
            continue  # nested entry, already captured with its parent

        text = dl.get_text(separator=" ", strip=True)
        if not text:
            continue

        dt = dl.find("dt")
        entry_id = dt.get("id") if dt else None
        entry_name = entry_id.split(".")[-1] if entry_id else "unknown"

        chunks.append(text)
        entry_names.append(entry_name)

    intro_parts = [el.get_text(strip=True) for el in main.find_all(["h1", "p"], recursive=False)]
    intro_parts = [p for p in intro_parts if p]
    if intro_parts:
        chunks.insert(0, " ".join(intro_parts))
        entry_names.insert(0, "intro")

    return chunks, entry_names


def load_and_chunk_all(raw_dir: str = "data/raw") -> list:
    """Returns list of {"source": filename, "chunk": text, "chunk_index": i, "entry_name": str}"""
    results = []
    for file_path in Path(raw_dir).glob("*.html"):
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks, entry_names = dom_chunk_sphinx_docs(html)
        for i, (c, entry_name) in enumerate(zip(chunks, entry_names)):
            results.append({
                "source": file_path.name,
                "chunk": c,
                "chunk_index": i,
                "entry_name": entry_name
            })
    return results


if __name__ == "__main__":
    results = load_and_chunk_all()
    by_source = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r)

    for source, items in by_source.items():
        print(f"\n=== {source} ===")
        print(f"Number of chunks: {len(items)}")
        print(f"Avg chunk size: {sum(len(i['chunk']) for i in items)/len(items):.0f} chars")
        print(f"Entry names (first 5): {[i['entry_name'] for i in items[:5]]}")
        print(f"\nSample chunk:\n{items[2]['chunk'][:400] if len(items) > 2 else items[0]['chunk'][:400]}")