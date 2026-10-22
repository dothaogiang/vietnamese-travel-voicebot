"""Download the official VIVOS archive or install a previously downloaded copy."""

import argparse
import tarfile
from pathlib import Path

from vietnamese_travel_voicebot.data.download import VIVOS_URL, download_vivos, install_vivos_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/vivos/VIVOS"))
    parser.add_argument("--archive", type=Path, help="use and verify this existing vivos.tar.gz instead of downloading")
    parser.add_argument("--url", help="override the official Zenodo file URL")
    args = parser.parse_args()
    try:
        if args.archive:
            metadata = install_vivos_archive(
                args.archive,
                args.dataset_dir,
                source_url=args.url or VIVOS_URL,
            )
        else:
            metadata = download_vivos(args.dataset_dir, source_url=args.url or VIVOS_URL)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.error(f"could not download/install VIVOS: {exc}; if network access is unavailable, pass --archive <local-vivos.tar.gz>")
    print(f"VIVOS installed at: {args.dataset_dir}")
    print(f"MD5: {metadata['md5']}")
    print(f"SHA-256: {metadata['sha256']}")
    print("License: CC BY-NC-SA 4.0; academic, non-commercial use only.")


if __name__ == "__main__":
    main()
