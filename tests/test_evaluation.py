"""Golden metric examples for the corpus-level ASR protocol."""

import unittest

from vietnamese_travel_voicebot.evaluation.metrics import calculate_asr_metrics


class ASRMetricsTests(unittest.TestCase):
    def test_uses_corpus_denominators_after_normalization(self) -> None:
        result = calculate_asr_metrics(
            ["Đà Nẵng!", "Huế"],
            ["đà lẵng", "HUẾ"],
        )

        self.assertAlmostEqual(result.cer, 1 / 10)
        self.assertAlmostEqual(result.wer, 1 / 3)
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.reference_char_count, 10)
        self.assertEqual(result.reference_token_count, 3)
        self.assertEqual(result.normalization_version, "vi-v1")

    def test_empty_hypothesis_counts_as_deletions(self) -> None:
        result = calculate_asr_metrics(["Huế"], [""])
        self.assertEqual((result.cer, result.wer), (1.0, 1.0))

    def test_rejects_invalid_corpora(self) -> None:
        for references, hypotheses in (
            ([], []),
            (["hà nội"], []),
            (["!!!"], ["hà nội"]),
        ):
            with self.subTest(references=references, hypotheses=hypotheses):
                with self.assertRaises(ValueError):
                    calculate_asr_metrics(references, hypotheses)
