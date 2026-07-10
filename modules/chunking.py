"""
Phase 1: Document chunking.

Two strategies implemented:
- fixed_size_chunk: simple, fast baseline (Concept 188)
- recursive_chunk: respects paragraph/section structure first,
  falls back to fixed-size splitting only when a unit is too large
"""

def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into fixed-size character chunks with overlap.

    Args:
        text: raw document text
        chunk_size: characters per chunk
        overlap: characters shared between adjacent chunks

    Returns:
        List of chunk strings
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text by paragraphs first (double newline), only falling back
    to fixed-size chunking for paragraphs that exceed chunk_size alone.
    Respects document structure where it exists.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Paragraph itself too large - flush buffer, then split it directly
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(fixed_size_chunk(para, chunk_size, overlap))
        elif len(buffer) + len(para) + 1 <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            chunks.append(buffer.strip())
            buffer = para

    if buffer:
        chunks.append(buffer.strip())

    return chunks


if __name__ == "__main__":
    sample_text = """This is the first paragraph. It talks about something important.

This is the second paragraph. It continues the discussion with more detail and context that matters.

This is a third, much shorter paragraph."""

    print("=== Fixed-size chunks ===")
    for i, c in enumerate(fixed_size_chunk(sample_text, chunk_size=100, overlap=20)):
        print(f"[{i}] {c}\n")

    print("=== Recursive chunks ===")
    for i, c in enumerate(recursive_chunk(sample_text, chunk_size=150)):
        print(f"[{i}] {c}\n")