#!/usr/bin/env bash
# Run the licensed VIVOS ASR baseline and comparator in sequence on one GPU.

set -euo pipefail

if [[ $# -ne 0 && $# -ne 2 ]]; then
  printf 'Usage: bash scripts/run_phase2.sh [--archive /path/to/vivos.tar.gz]\n' >&2
  exit 2
fi
if [[ $# -eq 2 && "$1" != "--archive" ]]; then
  printf 'Usage: bash scripts/run_phase2.sh [--archive /path/to/vivos.tar.gz]\n' >&2
  exit 2
fi

ASR_ARCHIVE_PATH=""
if [[ $# -eq 2 ]]; then
  ASR_ARCHIVE_PATH="$(readlink -f -- "$2")"
fi

ASR_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ASR_PROJECT_ROOT"
if [[ -n "${ASR_PYTHON:-}" ]]; then
  ASR_PYTHON="$(command -v -- "$ASR_PYTHON" || true)"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  ASR_PYTHON="$CONDA_PREFIX/bin/python"
else
  printf 'Activate Miniconda first: conda activate vietnamese-travel-voicebot\n' >&2
  exit 2
fi
ASR_DATASET_DIR="$ASR_PROJECT_ROOT/data/raw/vivos/VIVOS"
ASR_METADATA_FILE="$ASR_PROJECT_ROOT/data/raw/vivos/vivos_metadata.json"

if [[ -z "$ASR_PYTHON" || ! -x "$ASR_PYTHON" ]]; then
  printf 'Python interpreter not found. Activate the Conda environment or set ASR_PYTHON.\n' >&2
  exit 2
fi

"$ASR_PYTHON" scripts/check_environment.py --require-cuda --min-free-vram-gib 2.5

if [[ ! -f "$ASR_DATASET_DIR/train/prompts.txt" || ! -f "$ASR_DATASET_DIR/test/prompts.txt" ]]; then
  if [[ -n "$ASR_ARCHIVE_PATH" ]]; then
    "$ASR_PYTHON" scripts/download_data.py --archive "$ASR_ARCHIVE_PATH"
  else
    "$ASR_PYTHON" scripts/download_data.py
  fi
fi

if [[ ! -f "$ASR_METADATA_FILE" ]]; then
  printf 'VIVOS metadata is missing: %s. Install the verified archive with scripts/download_data.py.\n' "$ASR_METADATA_FILE" >&2
  exit 2
fi

"$ASR_PYTHON" scripts/prepare_data.py --dataset-root "$ASR_DATASET_DIR"
ASR_DATASET_SHA256="$("$ASR_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["sha256"])' "$ASR_METADATA_FILE")"
ASR_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ASR_OUTPUT_ROOT="$ASR_PROJECT_ROOT/experiments/001_asr_baseline/outputs/$ASR_RUN_ID"

for ASR_ROLE in primary comparator; do
  "$ASR_PYTHON" scripts/run_asr.py --device cuda:0 --dtype float16 --role "$ASR_ROLE" --manifest data/manifests/test.jsonl --dataset-root "$ASR_DATASET_DIR" --dataset-sha256 "$ASR_DATASET_SHA256" --output-dir "$ASR_OUTPUT_ROOT/${ASR_ROLE}_smoke"
  "$ASR_PYTHON" scripts/run_asr.py --device cuda:0 --dtype float16 --role "$ASR_ROLE" --manifest data/manifests/test.jsonl --dataset-root "$ASR_DATASET_DIR" --dataset-sha256 "$ASR_DATASET_SHA256" --full --output-dir "$ASR_OUTPUT_ROOT/${ASR_ROLE}_full"
done

printf 'Phase 2 ASR results: %s\n' "$ASR_OUTPUT_ROOT"
