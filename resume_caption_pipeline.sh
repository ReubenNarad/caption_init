#!/usr/bin/env bash
set -euo pipefail

RESULTS_FILE="results/elements_10cartoons_20251110-120029.json"
CARTOON_COUNT=10

GEN_MODELS=(
  "anthropic/claude-sonnet-4.5"
  "anthropic/claude-3.5-sonnet"
  "google/gemini-2.5-pro"
  "z-ai/glm-4.6"
  "openai/gpt-5"
)

FILTER_MODELS=(
  "anthropic/claude-3.5-sonnet"
  "anthropic/claude-3.7-sonnet"
  "anthropic/claude-4.5-sonnet"
)

DIR_CAPTION_ARGS=(--n-workers 10)

if [[ ! -f "$RESULTS_FILE" ]]; then
  echo "Results file not found: $RESULTS_FILE"
  exit 1
fi

echo "Resuming directional caption pipeline using $RESULTS_FILE"

for idx in $(seq 1 "$CARTOON_COUNT"); do
  echo "=== Cartoon index $idx/$CARTOON_COUNT ==="
  for model in "${GEN_MODELS[@]}"; do
    echo "--- Generating directional captions with $model ---"
    reasoning_args=()
    if [[ "$model" == "openai/gpt-5" ]]; then
      reasoning_args=(--reasoning-effort medium)
    fi

    cmd=(
      python directional_caption_pairs.py
      --results "$RESULTS_FILE"
      --index "$idx"
      --model "$model"
      "${DIR_CAPTION_ARGS[@]}"
    )
    if [[ "${#reasoning_args[@]}" -gt 0 ]]; then
      cmd+=("${reasoning_args[@]}")
    fi
    "${cmd[@]}"

    latest_dir=$(ls -dt captions/directional_contest_* | head -n 1)
    csv_path="$latest_dir/directional_captions.csv"
    echo "    Captions saved under $latest_dir"

    for filter_model in "${FILTER_MODELS[@]}"; do
      echo "------ Filtering with $filter_model ------"
      python filter_directional.py "$csv_path" --model "$filter_model"
    done
  done
done
