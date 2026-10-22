"""Validate local VIVOS files and create deterministic train/test manifests."""

import argparse
from pathlib import Path

from vietnamese_travel_voicebot.data.prepare import prepare_vivos_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/vivos/VIVOS"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    try:
        manifests = prepare_vivos_data(args.dataset_root, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for split, path in manifests.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
