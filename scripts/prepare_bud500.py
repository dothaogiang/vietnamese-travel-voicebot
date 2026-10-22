"""Prepare a versioned keyword slice across the two authorized Bud500 test shards."""

import argparse
import importlib.util
import shutil
from pathlib import Path

from vietnamese_travel_voicebot.data.bud500 import prepare_bud500_testset, validate_bud500_slice, select_bud500_rows
from vietnamese_travel_voicebot.data.normalize import normalize_transcript
from vietnamese_travel_voicebot.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="download at most two Bud500 test shards (~608 MB total), reusing cached files")
    source.add_argument("--parquet", type=Path, action="append", help="local test shard; repeat for a second shard")
    source.add_argument("--validate-existing", action="store_true", help="verify cached manifest and audio checksums")
    parser.add_argument("--config", type=Path, default=Path("configs/bud500.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/bud500/travel_test_v3"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--filter", choices=("travel", "all"), default="travel")
    args = parser.parse_args()
    if args.validate_existing:
        metadata = validate_bud500_slice(args.output_root)
        print(f"Bud500 cached slice verified: {metadata['sample_count']} records")
        return 0
    if importlib.util.find_spec("pyarrow") is None or shutil.which("ffmpeg") is None:
        parser.error("preparation requires pyarrow and ffmpeg; install '.[bud500]' and the system ffmpeg package")
    config = load_yaml_config(args.config)
    repo_id = config["repo_id"]
    limit = args.limit if args.limit is not None else config["max_samples"]
    if limit <= 0:
        parser.error("--limit must be positive")
    revision = None
    if args.download:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        try:
            revision = config["source_revision"]
            parquets, matches = [], 0
            for filename in config["filenames"]:
                kwargs = dict(repo_id=repo_id, repo_type="dataset", filename=filename,
                              revision=revision, local_dir="data/raw/bud500/source")
                try:
                    parquet = hf_hub_download(**kwargs, local_files_only=True)
                except LocalEntryNotFoundError:
                    parquet = hf_hub_download(**kwargs)
                parquets.append(Path(parquet))
                matches += len(select_bud500_rows(Path(parquet), limit - matches,
                                                [normalize_transcript(term) for term in config["travel_terms"]], args.filter))
                print(f"Bud500: {matches}/{limit} keyword matches after {len(parquets)} test shard(s)", flush=True)
                if matches >= limit:
                    break
        except Exception as exc:
            if type(exc).__name__ == "GatedRepoError":
                parser.error(
                    f"Hugging Face has not granted this account access to {repo_id}. "
                    f"Open https://huggingface.co/datasets/{repo_id} while signed in, "
                    "select Request access/Agree, and wait until the owner approves it. "
                    "Then rerun this command. Logging in again does not grant dataset access."
                )
            parser.error(
                f"Bud500 download failed ({type(exc).__name__}: {exc}). "
                f"Check network and account access at https://huggingface.co/datasets/{repo_id}; "
                "if the dataset is gated, request access and wait for approval. "
                "You can also use --parquet with a local test shard."
            )
    else:
        parquets = args.parquet
        if any(not path.name.startswith("test-") for path in parquets):
            parser.error("--parquet must point to an original Bud500 test shard (filename starts with test-)")
    manifest, metadata = prepare_bud500_testset(
        parquets, args.output_root, limit=limit,
        filter_mode=args.filter, source_revision=revision, travel_terms=config["travel_terms"],
        reusable_slices=[Path(path) for path in config.get("reusable_slices", [])],
    )
    print(f"manifest={manifest}")
    print(f"metadata={metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
