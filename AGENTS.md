# AGENTS.md

## Purpose
This repo builds a practical caption-generation pipeline for New Yorker-style cartoons.

Core goal:
- generate many plausible captions from a cartoon description
- keep diversity high
- rank/filter down to strong finalists
- optionally evaluate against known winners

## Repo Map
- `cartoon_workflow.py`: extract cartoon elements + brainstorm ideas
- `directional_caption_pairs.py`: generate directional captions (idea mode or premise mode)
- `filter_directional.py`: select top captions from a generated CSV
- `seed_winner.py`: inject known winner into a generated CSV
- `compare_filtered.py`: compare selector-model agreement
- `evaluate_brainstorm.py`: embedding-based idea-vs-winner evaluation
- `pairwise_similarity.py`: pairwise idea similarity matrix
- `element_extractor.py`: legacy script; prefer `cartoon_workflow.py`

## Environment
Install:
```bash
pip install -r requirements.txt
```

Set keys in `.env`:
```bash
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

## Workflow (Single Contest)
1. Generate elements + brainstorm ideas:
```bash
python cartoon_workflow.py --count 1 --model openai/gpt-4o-mini --brainstorm-total 20 --brainstorm-batch 5
```

2. Generate directional captions (cheap preset):
```bash
python directional_caption_pairs.py \
  --results results/elements_1cartoons_20251016-144553.json \
  --contest 2.0 \
  --model openai/gpt-4o-mini \
  --use-premises \
  --premise-count 4 \
  --skip-similarity \
  --n-workers 2
```

3. Filter top captions:
```bash
RUN_DIR=$(ls -dt captions/directional_contest_* | head -n 1)
python filter_directional.py "$RUN_DIR/directional_captions.csv" --model openai/gpt-4o-mini --top-k 3
```

## Optional Evaluation Steps
Add winner row:
```bash
python seed_winner.py "$RUN_DIR/directional_captions.csv" --contest 2.0
```

Compare model agreement:
```bash
python compare_filtered.py "$RUN_DIR"
```

Embedding evaluation:
```bash
python evaluate_brainstorm.py --results results/elements_1cartoons_20251016-144553.json
```

## Conventions For Agents
- Keep generated artifacts in `captions/`, `results/`, `evaluation/`, `pairwise/`.
- Do not hand-edit generated CSV/JSON outputs unless debugging.
- Prefer adding new scripts over expanding shell wrappers with hidden logic.
- Keep prompts legible and deterministic where possible.
- If changing model defaults, update both `README.md` and this file.
