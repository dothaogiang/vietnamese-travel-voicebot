"""Versioned transcript normalization for Vietnamese ASR evaluation."""

import unicodedata


TRANSCRIPT_NORMALIZATION_VERSION = "vi-v1"


def normalize_transcript(text: str) -> str:
    """Normalize text for protocol vi-v1 while preserving tones and digits.

    Unicode punctuation becomes a space, so adjacent words remain separate.
    Numbers are not verbalized; that policy would require a new protocol version.
    """
    if not isinstance(text, str):
        raise TypeError("transcript must be a string")

    normalized = unicodedata.normalize("NFC", text).lower()
    without_punctuation = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in normalized
    )
    return " ".join(without_punctuation.split())
