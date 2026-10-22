#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source_args=(--download)
if [[ "${1:-}" == "--parquet" ]]; then
  if [[ $# -ne 2 ]]; then
    echo "usage: bash scripts/run_bud500.sh [--parquet /path/to/test-shard.parquet]" >&2
    exit 2
  fi
  source_args=(--parquet "$2")
elif [[ $# -ne 0 ]]; then
  echo "usage: bash scripts/run_bud500.sh [--parquet /path/to/test-shard.parquet]" >&2
  exit 2
fi

dataset_root=data/raw/bud500/travel_test_v1
manifest="$dataset_root/manifest.jsonl"
if [[ ! -f "$manifest" ]]; then
  python scripts/prepare_bud500.py "${source_args[@]}"
fi
shard_sha256="$(python -c 'import json; print(json.load(open("data/raw/bud500/travel_test_v1/metadata.json"))["source_shard_sha256"])')"
run_dir="experiments/002_bud500_eval/outputs/$(date -u +%Y%m%dT%H%M%SZ)"
for role in primary comparator; do
  python scripts/run_asr.py \
    --role "$role" \
    --manifest "$manifest" \
    --dataset-root "$dataset_root" \
    --dataset-name Bud500 \
    --dataset-sha256 "$shard_sha256" \
    --evaluation-config configs/bud500_evaluation.yaml \
    --full \
    --output-dir "$run_dir/$role"
done
echo "Bud500 run: $run_dir"
