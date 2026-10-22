#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${ASR_PYTHON:-}" ]]; then
  ASR_PYTHON="$(command -v -- "$ASR_PYTHON")"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  ASR_PYTHON="$CONDA_PREFIX/bin/python"
else
  echo "Activate Miniconda first: conda activate vietnamese-travel-voicebot" >&2
  exit 2
fi

if [[ $# -gt 0 ]]; then
  if [[ "$1" != "--parquet" ]]; then
    echo "usage: bash scripts/run_bud500.sh [--parquet PATH [--parquet PATH]]" >&2
    exit 2
  fi
  dataset_root="$("$ASR_PYTHON" -c 'from vietnamese_travel_voicebot.utils.config import load_yaml_config; print(load_yaml_config("configs/phase2.yaml")["bud500"]["dataset_root"])')"
  "$ASR_PYTHON" scripts/prepare_bud500.py "$@" --output-root "$dataset_root"
fi
exec "$ASR_PYTHON" scripts/complete_phase2.py --run-bud500
