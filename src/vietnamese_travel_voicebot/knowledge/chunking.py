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
    """Split on paragraphs and then word boundaries without dropping any text."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    paragraphs = [" ".join(part.split()) for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        words = paragraph.split()
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
