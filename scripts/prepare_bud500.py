"""Download one gated Bud500 test shard or prepare a supplied local shard."""

import argparse
from pathlib import Path

from vietnamese_travel_voicebot.data.bud500 import prepare_bud500_shard

REPO_ID = "linhtran92/viet_bud500"
TEST_SHARD = "data/test-00000-of-00002-531c1d81edb57297.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="download only the first Bud500 test shard (~306 MB)")
    source.add_argument("--parquet", type=Path, help="local Bud500 test Parquet shard")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/bud500/travel_test_v1"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--filter", choices=("travel", "all"), default="travel")
    args = parser.parse_args()
    revision = None
    if args.download:
        from huggingface_hub import HfApi, hf_hub_download
        try:
            revision = HfApi().dataset_info(REPO_ID).sha
            parquet = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", filename=TEST_SHARD,
                revision=revision, local_dir="data/raw/bud500/source",
            )
        except Exception as exc:
            if type(exc).__name__ == "GatedRepoError":
                parser.error(
                    f"Hugging Face has not granted this account access to {REPO_ID}. "
                    f"Open https://huggingface.co/datasets/{REPO_ID} while signed in, "
                    "select Request access/Agree, and wait until the owner approves it. "
                    "Then rerun this command. Logging in again does not grant dataset access."
                )
            parser.error(
                f"Bud500 download failed ({type(exc).__name__}: {exc}). "
                f"Check network and account access at https://huggingface.co/datasets/{REPO_ID}; "
                "if the dataset is gated, request access and wait for approval. "
                "You can also use --parquet with a local test shard."
            )
    else:
        parquet = args.parquet
        if not parquet.name.startswith("test-"):
            parser.error("--parquet must point to an original Bud500 test shard (filename starts with test-)")
    manifest, metadata = prepare_bud500_shard(
        parquet, args.output_root, limit=args.limit, filter_mode=args.filter, source_revision=revision,
    )
    print(f"manifest={manifest}")
    print(f"metadata={metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
