"""First text-only slice: query → intent/entities → sourced retrieval → answer."""

from dataclasses import dataclass
from pathlib import Path

from vietnamese_travel_voicebot.knowledge.ingestion import load_knowledge
from vietnamese_travel_voicebot.knowledge.retrieval import retrieve
from vietnamese_travel_voicebot.voicebot.intent import classify_intent, extract_entities
from vietnamese_travel_voicebot.voicebot.rag import SourceReference, answer_from_evidence


@dataclass(frozen=True)
class TextResponse:
    status: str
    intent: str
    entities: dict[str, list[str]]
    answer: str
    sources: list[SourceReference]
    states: list[str]


def answer_text(
    query: str,
    knowledge_root: str | Path,
    *,
    top_k: int = 3,
    max_chunk_chars: int = 700,
    min_overlap: float = 0.25,
) -> TextResponse:
    if not query.strip():
        raise ValueError("query must not be empty")
    intent = classify_intent(query)
    entities = extract_entities(query)
    evidence = retrieve(
        query,
        load_knowledge(knowledge_root, max_chunk_chars=max_chunk_chars),
        top_k=top_k,
        min_overlap=min_overlap,
    )
    answer = answer_from_evidence(evidence)
    return TextResponse(
        status=answer.status,
        intent=intent,
        entities=entities,
        answer=answer.answer,
        sources=answer.sources,
        states=["Processing", "Thinking"],
    )
