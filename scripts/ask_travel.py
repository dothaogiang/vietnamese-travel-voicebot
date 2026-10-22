"""Ask the local, source-grounded text travel assistant."""

import argparse
from pathlib import Path

from vietnamese_travel_voicebot.utils.config import load_yaml_config
from vietnamese_travel_voicebot.voicebot.pipeline import answer_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="a Vietnamese travel question")
    parser.add_argument("--knowledge-root", type=Path, default=Path("knowledge_base"))
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config).get("retrieval", {})
    except RuntimeError as exc:
        parser.error(f"{exc}; install with `python -m pip install -e '.[voicebot]'`")
    response = answer_text(
        args.question,
        args.knowledge_root,
        top_k=config.get("top_k", 3),
        max_chunk_chars=config.get("max_chunk_chars", 700),
        min_overlap=config.get("min_token_overlap", 0.25),
    )
    print(f"intent={response.intent}; status={response.status}")
    print(response.answer)
    for index, source in enumerate(response.sources, start=1):
        print(f"[{index}] {source.title} | {source.url} | cập nhật: {source.updated_at} | license: {source.license}")
    return 0 if response.status == "answered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
