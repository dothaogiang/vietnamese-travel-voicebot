"""Render answers only from retrieved source text and return citation metadata."""

from dataclasses import dataclass

from vietnamese_travel_voicebot.knowledge.retrieval import RetrievedChunk


@dataclass(frozen=True)
class SourceReference:
    title: str
    url: str
    category: str
    updated_at: str
    license: str


@dataclass(frozen=True)
class GroundedAnswer:
    status: str
    answer: str
    sources: list[SourceReference]


def answer_from_evidence(results: list[RetrievedChunk]) -> GroundedAnswer:
    if not results:
        return GroundedAnswer(
            status="insufficient_evidence",
            answer="Mình chưa có dữ liệu có nguồn để trả lời câu hỏi này.",
            sources=[],
        )
    sources: list[SourceReference] = []
    paragraphs: list[str] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        chunk = result.chunk
        key = (chunk.title, chunk.source_url)
        if key in seen:
            continue
        seen.add(key)
        source_number = len(sources) + 1
        sources.append(SourceReference(
            title=chunk.title,
            url=chunk.source_url,
            category=chunk.category,
            updated_at=chunk.updated_at,
            license=chunk.license,
        ))
        paragraphs.append(f"Theo nguồn [{source_number}] “{chunk.title}”: {chunk.text}")
    return GroundedAnswer(status="answered", answer="\n".join(paragraphs), sources=sources)
