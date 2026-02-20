# Joke_Agent

Agentic workflow for generating and filtering New Yorker-style cartoon captions.

## Motivation
Given a cartoon description, generate captions that are not just wordplay noise:
- identify the two core incongruous elements
- explore diverse idea space around each element
- bridge one element's ideas into the other
- filter down to strong finalists

## Pipeline
1. `cartoon_workflow.py`
- Input: `comprehensive_annotations.csv`
- Output: `results/elements_*cartoons_*.json`
- Purpose: extract elements + brainstorm idea lists for each element.

2. `directional_caption_pairs.py`
- Input: one contest/cartoon from `results/*.json`
- Output: `captions/directional_contest_*/directional_captions.csv` + `summary.json`
- Purpose: generate directional captions.
- Modes:
  - idea mode (default): use brainstorm ideas
  - premise mode (`--use-premises`): first generate diverse joke premises, then caption

3. `filter_directional.py`
- Input: one `directional_captions.csv`
- Output: `filtered_*.json` (or `filtered_seeded_*.json`)
- Purpose: ask a judge model for top-k captions.

4. Optional evaluation helpers
- `seed_winner.py`: insert ground-truth winner row into generated CSV
- `compare_filtered.py`: compare multiple filtered outputs and produce agreement heatmap + consensus
- `evaluate_brainstorm.py`: embedding-based idea-vs-winner analysis
- `pairwise_similarity.py`: pairwise similarity matrix between element idea sets

## Quick Start (Single Contest, Low Cost)
```bash
cd /Users/reuben/Desktop/Joke_Agent

# 0) Generate elements/ideas for 1 cartoon (if you need fresh results)
python cartoon_workflow.py --count 1 --model openai/gpt-4o-mini --brainstorm-total 20 --brainstorm-batch 5

# 1) Directional caption generation (cheap run)
python directional_caption_pairs.py \
  --results results/elements_1cartoons_20251016-144553.json \
  --contest 2.0 \
  --model openai/gpt-4o-mini \
  --use-premises \
  --premise-count 4 \
  --skip-similarity \
  --n-workers 2

# 2) Filter finalists
RUN_DIR=$(ls -dt captions/directional_contest_* | head -n 1)
python filter_directional.py "$RUN_DIR/directional_captions.csv" \
  --model openai/gpt-4o-mini \
  --top-k 3
```

## Batch Scripts
- `bulk_caption_pipeline.sh`: full end-to-end batch over multiple models/contests
- `resume_caption_pipeline.sh`: resume from a pinned results file
- `filter_recent_directional.sh`: re-filter recent directional runs

## Notes
- `element_extractor.py` is a legacy one-step extractor kept for reference.
- API keys are read from `.env`:
  - `OPENROUTER_API_KEY`
  - `OPENAI_API_KEY` (for embedding-based similarity)
  - `GEMINI_API_KEY` (only for `embedding.py`)
