"""Corpus-level ASR metrics for the fixed Vietnamese evaluation protocol."""

from dataclasses import dataclass
from typing import Sequence, TypeVar

from vietnamese_travel_voicebot.data.normalize import (
    TRANSCRIPT_NORMALIZATION_VERSION,
    normalize_transcript,
)


@dataclass(frozen=True)
class ASRMetrics:
    """Micro-averaged error rates and denominators for one corpus."""

    cer: float
    wer: float
    sample_count: int
    reference_char_count: int
    reference_token_count: int
    normalization_version: str


Token = TypeVar("Token")


def _edit_distance(reference: Sequence[Token], hypothesis: Sequence[Token]) -> int:
    """Count insertions, deletions and substitutions using one DP row."""
    if len(hypothesis) > len(reference):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1]
                    + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def calculate_asr_metrics(
    references: Sequence[str], hypotheses: Sequence[str]
) -> ASRMetrics:
    """Calculate CER and whitespace-token WER after protocol vi-v1 normalization.

    CER includes spaces in its character count. Every reference must contain
    speech after normalization; silence evaluation needs a separate protocol.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    if not references:
        raise ValueError("at least one reference and hypothesis are required")

    normalized_references = [normalize_transcript(text) for text in references]
    normalized_hypotheses = [normalize_transcript(text) for text in hypotheses]
    if any(not text for text in normalized_references):
        raise ValueError("references must be non-empty after normalization")

    char_errors = sum(
        _edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(
            normalized_references, normalized_hypotheses
        )
    )
    token_errors = sum(
        _edit_distance(reference.split(), hypothesis.split())
        for reference, hypothesis in zip(
            normalized_references, normalized_hypotheses
        )
    )
    reference_char_count = sum(map(len, normalized_references))
    reference_token_count = sum(
        len(text.split()) for text in normalized_references
    )

    return ASRMetrics(
        cer=char_errors / reference_char_count,
        wer=token_errors / reference_token_count,
        sample_count=len(normalized_references),
        reference_char_count=reference_char_count,
        reference_token_count=reference_token_count,
        normalization_version=TRANSCRIPT_NORMALIZATION_VERSION,
    )
