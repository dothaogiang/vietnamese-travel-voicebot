"""Dependency-free lexical retrieval for the first local knowledge slice."""

from dataclasses import dataclass

from vietnamese_travel_voicebot.data.normalize import normalize_transcript
from vietnamese_travel_voicebot.knowledge.chunking import KnowledgeChunk

STOPWORDS = {
    "là", "ở", "đâu", "bao", "nhiêu", "gì", "có", "không", "cho", "tôi", "xin",
    "hãy", "giúp", "được", "vậy", "với", "nào", "thế", "này", "kia", "một", "các",
}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float


def retrieve(
    query: str,
    chunks: list[KnowledgeChunk],
    *,
    top_k: int = 3,
    min_overlap: float = 0.25,
) -> list[RetrievedChunk]:
    """Rank chunks by informative query token overlap; require at least 25% overlap."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    terms = {token for token in normalize_transcript(query).split() if len(token) > 1 and token not in STOPWORDS}
    if not terms:
        return []
    ranked: list[RetrievedChunk] = []
    for chunk in chunks:
        document_terms = set(normalize_transcript(chunk.text).split())
        score = len(terms & document_terms) / len(terms)
        if score >= min_overlap:
            ranked.append(RetrievedChunk(chunk, score))
    ranked.sort(key=lambda result: (-result.score, result.chunk.source_id, result.chunk.chunk_id))
    return ranked[:top_k]
