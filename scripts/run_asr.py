"""Run a pinned ASR candidate on a smoke subset or the complete manifest."""

import argparse
import re
from pathlib import Path

from vietnamese_travel_voicebot.asr.whisper import WhisperASRProvider
from vietnamese_travel_voicebot.evaluation.evaluate_asr import evaluate_manifest
from vietnamese_travel_voicebot.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/asr.yaml"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--role", choices=("primary", "comparator"), default="primary")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/vivos/VIVOS"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=8, help="fixed-order smoke subset size (default: 8)")
    parser.add_argument("--full", action="store_true", help="evaluate every row in the manifest")
    parser.add_argument("--dataset-sha256", help="SHA-256 of the source dataset file, if recorded")
    parser.add_argument("--dataset-name", default="VIVOS", help="name recorded in summary.json")
    args = parser.parse_args()
    if args.dataset_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.dataset_sha256):
        parser.error("--dataset-sha256 must be a 64-character hexadecimal SHA-256")

    asr_config = load_yaml_config(args.config)
    evaluation_config = load_yaml_config(args.evaluation_config)
    model = asr_config[args.role]
    manifest = args.manifest or evaluation_config.get("asr", {}).get("manifest")
    if not manifest:
        parser.error("manifest is not configured; pass --manifest after preparing local VIVOS data")
    manifest = Path(manifest)
    if not manifest.is_file():
        parser.error(f"manifest does not exist: {manifest}; run scripts/prepare_data.py after installing VIVOS")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}; run scripts/download_data.py first")
    output_dir = args.output_dir or Path("experiments/001_asr_baseline/outputs") / args.role
    provider = WhisperASRProvider(
        model_id=model["model_id"],
        revision=model["revision"],
        device=asr_config.get("device", "cuda:0"),
        dtype=asr_config.get("inference", {}).get("dtype", "float16"),
        language=asr_config.get("inference", {}).get("language", "vi"),
        task=asr_config.get("inference", {}).get("task", "transcribe"),
    )
    summary = evaluate_manifest(
        provider,
        manifest,
        args.dataset_root,
        output_dir,
        target_sample_rate_hz=asr_config.get("audio", {}).get("sample_rate_hz", 16000),
        limit=None if args.full else args.limit,
        dataset_sha256=args.dataset_sha256,
        dataset_name=args.dataset_name,
        run_metadata={
            "model_role": args.role,
            "asr_config": asr_config,
            "evaluation_config": evaluation_config,
            "limit": None if args.full else args.limit,
            "device_requested": asr_config.get("device", "cuda:0"),
        },
    )
    cer = f"{summary['cer']:.6f}" if summary["cer"] is not None else "unavailable"
    wer = f"{summary['wer']:.6f}" if summary["wer"] is not None else "unavailable"
    rtf = f"{summary['mean_rtf']:.4f}" if summary["mean_rtf"] is not None else "unavailable"
    print(f"CER={cer} WER={wer} n={summary['sample_count']} failed={summary['failed_count']}")
    print(f"mean RTF={rtf}; artifacts={output_dir}")
    return 0 if summary["evaluation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
