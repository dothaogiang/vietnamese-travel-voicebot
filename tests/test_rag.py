"""Tests for source metadata, retrieval, and grounded travel answers."""

import tempfile
import unittest
from pathlib import Path

from vietnamese_travel_voicebot.knowledge.chunking import chunk_text
from vietnamese_travel_voicebot.knowledge.ingestion import load_knowledge
from vietnamese_travel_voicebot.voicebot.pipeline import answer_text


class TravelKnowledgePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "destinations" / "hue.md"
        source.parent.mkdir()
        source.write_text(
            "---\ntitle: Di tích Huế\nsource_url: https://example.org/hue\n"
            "updated_at: 2026-10-20\nlicense: CC-BY-4.0\n---\n"
            "Quần thể di tích Cố đô Huế nằm ở thành phố Huế.\n\n"
            "Khu di tích có các lăng tẩm và công trình cung đình.\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_loads_markdown_into_chunks_with_citation_metadata(self) -> None:
        chunks = load_knowledge(self.root, max_chunk_chars=30)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk.text) <= 30 for chunk in chunks))
        self.assertEqual(chunks[0].title, "Di tích Huế")
        self.assertEqual(chunks[0].source_url, "https://example.org/hue")
        self.assertEqual(chunks[0].category, "destinations")

    def test_answers_from_retrieved_evidence_and_returns_source(self) -> None:
        response = answer_text("Di tích Huế nằm ở đâu?", self.root)
        self.assertEqual(response.status, "answered")
        self.assertIn("nằm ở thành phố Huế", response.answer)
        self.assertEqual(response.sources[0].url, "https://example.org/hue")
        self.assertEqual(response.intent, "destination_info")
        self.assertIn("Huế", response.entities["destinations"])

    def test_refuses_to_answer_when_knowledge_has_no_support(self) -> None:
        response = answer_text("Giá vé máy bay ngày mai là bao nhiêu?", self.root)
        self.assertEqual(response.status, "insufficient_evidence")
        self.assertIn("chưa có dữ liệu", response.answer.lower())
        self.assertEqual(response.sources, [])

    def test_rejects_markdown_without_source_url(self) -> None:
        path = self.root / "destinations" / "unattributed.md"
        path.write_text("---\ntitle: Thiếu nguồn\n---\nThông tin chưa dẫn nguồn.\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source_url"):
            load_knowledge(self.root)


class ChunkLimitTests(unittest.TestCase):
    def test_splits_long_tokens_without_losing_characters(self) -> None:
        for limit in (1, 8, 30):
            for text in ("https://example.org/" + "a" * 100, "Huế " + "đ" * 90 + " đẹp"):
                with self.subTest(limit=limit, text=text):
                    chunks = chunk_text(text, max_chars=limit)
                    self.assertTrue(all(0 < len(chunk) <= limit for chunk in chunks))
                    self.assertEqual("".join(chunks).replace(" ", ""), text.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
