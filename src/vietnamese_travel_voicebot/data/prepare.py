"""Public data preparation entry point for the validated VIVOS dataset."""

from pathlib import Path
from typing import Mapping

from vietnamese_travel_voicebot.data.manifest import build_vivos_manifests


def prepare_vivos_data(
    dataset_root: str | Path, manifest_dir: str | Path
) -> Mapping[str, Path]:
    """Validate the local VIVOS train/test files and write their manifests."""
    return build_vivos_manifests(dataset_root, manifest_dir)
