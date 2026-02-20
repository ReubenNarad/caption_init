#!/usr/bin/env bash
set -euo pipefail

TARGET_STAMP="${1:-20251110}"
BASE_DIR="captions"

FILTER_MODELS=(
  "anthropic/claude-3.5-sonnet"
  "anthropic/claude-3.7-sonnet"
  "anthropic/claude-4.5-sonnet"
)

dirs=($(ls -dt ${BASE_DIR}/directional_contest_*"${TARGET_STAMP}"-* 2>/dev/null || true))

if [[ ${#dirs[@]} -eq 0 ]]; then
  echo "No directional_contest folders found for stamp ${TARGET_STAMP}."
  exit 1
fi

echo "Filtering ${#dirs[@]} directional runs tagged with ${TARGET_STAMP}"

for dir in "${dirs[@]}"; do
  csv_path="${dir}/directional_captions.csv"
  if [[ ! -f "$csv_path" ]]; then
    echo "Skipping ${dir}: missing directional_captions.csv"
    continue
  fi

  echo "=== ${dir} ==="
  for model in "${FILTER_MODELS[@]}"; do
    echo "--- Filtering with ${model} ---"
    python filter_directional.py "$csv_path" --model "$model"
  done
done
