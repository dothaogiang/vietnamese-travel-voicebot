"""Split attributed knowledge documents into stable retrieval chunks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    title: str
    source_url: str
    category: str
    updated_at: str
    license: str
    text: str


def chunk_text(text: str, *, max_chars: int = 700) -> list[str]:
    """Split on word boundaries, slicing oversized tokens to enforce max_chars."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    paragraphs = [" ".join(part.split()) for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        words = [word[start:start + max_chars]
                 for word in paragraph.split()
                 for start in range(0, len(word), max_chars)]
        piece = ""
        for word in words:
            candidate = f"{piece} {word}".strip()
            if piece and len(candidate) > max_chars:
                if current and len(current) + len(piece) + 1 > max_chars:
                    chunks.append(current)
                    current = ""
                current = f"{current} {piece}".strip()
                piece = word
            else:
                piece = candidate
        if not piece:
            continue
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks
