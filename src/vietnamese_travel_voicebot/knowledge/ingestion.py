"""Load local Markdown knowledge with required attribution metadata."""

from pathlib import Path

from vietnamese_travel_voicebot.knowledge.chunking import KnowledgeChunk, chunk_text


def _parse_markdown(path: Path, root: Path) -> tuple[dict[str, str], str]:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        raise ValueError(f"missing YAML-style front matter: {path}")
    closing = content.find("\n---\n", 4)
    if closing < 0:
        raise ValueError(f"unterminated front matter: {path}")
    metadata: dict[str, str] = {}
    for line in content[4:closing].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"invalid metadata line in {path}: {line!r}")
        metadata[key.strip()] = value.strip().strip("\"'")
    required = ("title", "source_url", "updated_at", "license")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"missing required metadata {', '.join(missing)} in {path}")
    body = content[closing + 5 :].strip()
    if not body:
        raise ValueError(f"knowledge document has no body: {path}")
    return metadata, body


def load_knowledge(root: str | Path, *, max_chunk_chars: int = 700) -> list[KnowledgeChunk]:
    """Load Markdown recursively; fail on unattributed or empty source files."""
    knowledge_root = Path(root).expanduser().resolve()
    if not knowledge_root.is_dir():
        raise FileNotFoundError(f"knowledge directory does not exist: {knowledge_root}")
    files = sorted(path for path in knowledge_root.rglob("*.md") if path.name.lower() != "readme.md")
    chunks: list[KnowledgeChunk] = []
    for path in files:
        metadata, body = _parse_markdown(path, knowledge_root)
        source_id = path.relative_to(knowledge_root).with_suffix("").as_posix()
        category = metadata.get("category", path.parent.name)
        for index, text in enumerate(chunk_text(body, max_chars=max_chunk_chars), start=1):
            chunks.append(KnowledgeChunk(
                chunk_id=f"{source_id}#chunk-{index}",
                source_id=source_id,
                title=metadata["title"],
                source_url=metadata["source_url"],
                category=category,
                updated_at=metadata["updated_at"],
                license=metadata["license"],
                text=text,
            ))
    return chunks
