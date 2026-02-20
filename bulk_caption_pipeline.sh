#!/usr/bin/env bash
set -euo pipefail

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

DIR_CAPTION_EXTRA_ARGS=(--n-workers 10)

echo "=== Step 1: generate elements/brainstorms for the first 10 cartoons ==="
python cartoon_workflow.py --count 10
RESULTS_FILE=$(ls -t results/elements_10cartoons_*.json | head -n 1)
echo "Using results file: $RESULTS_FILE"

for idx in $(seq 1 10); do
  echo "=== Cartoon index $idx/10 ==="
  for model in "${GEN_MODELS[@]}"; do
    echo "--- Generating directional captions with $model ---"
    reasoning_flag=()
    if [[ "$model" == "openai/gpt-5" ]]; then
      reasoning_flag=(--reasoning-effort medium)
    fi
    python directional_caption_pairs.py \
      --results "$RESULTS_FILE" \
      --index "$idx" \
      --model "$model" \
      "${DIR_CAPTION_EXTRA_ARGS[@]}" \
      "${reasoning_flag[@]}"
    latest_dir=$(ls -dt captions/directional_contest_* | head -n 1)
    csv_path="$latest_dir/directional_captions.csv"
    echo "      Captions saved under $latest_dir"

    for filter_model in "${FILTER_MODELS[@]}"; do
      echo "------ Filtering with $filter_model ------"
      python filter_directional.py "$csv_path" --model "$filter_model"
    done
  done
done
