#!/usr/bin/env python3
"""Evaluate brainstormed ideas against winning captions using OpenAI embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datetime import datetime

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "The openai package is required. Install it with 'pip install openai'."
    ) from exc


DEFAULT_RESULTS_PATH = Path("results")
DEFAULT_EVAL_OUTPUT_DIR = Path("evaluation")
DEFAULT_CSV_PATH = Path("comprehensive_annotations.csv")
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 64
DEFAULT_TOP_K = 10


@dataclass
class CartoonData:
    contest_number: str
    record_index: Optional[int]
    captions: List[str]
    ideas: List[str]
    elements: List[str]


def load_openai_client(env_path: str = ".env") -> OpenAI:
    """Instantiate an OpenAI client using environment variables or .env file."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        dotenv_path = Path(env_path)
        if dotenv_path.is_file():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() == "OPENAI_API_KEY":
                    api_key = value.strip().strip('"').strip("'")
                    break

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in the environment or the provided .env file"
        )

    return OpenAI(api_key=api_key)


def read_captions(csv_path: Path) -> Dict[str, List[str]]:
    """Return a mapping of contest_number -> list of winning captions."""
    captions: Dict[str, List[str]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contest_number = (row.get("contest_number") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not contest_number or not caption:
                continue
            if caption not in captions[contest_number]:
                captions[contest_number].append(caption)
    return captions


def read_brainstorm_results(results_path: Path) -> Dict[str, List[dict]]:
    """Return a mapping of contest_number -> brainstormed idea payloads."""
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    contest_to_ideas: Dict[str, List[dict]] = defaultdict(list)
    seen_per_contest: Dict[str, set] = defaultdict(set)
    for record in data:
        contest_number = str(record.get("contest_number") or "").strip()
        if not contest_number:
            continue
        record_index = record.get("index")
        brainstorm = record.get("brainstorm") or {}
        if not isinstance(brainstorm, dict):
            continue
        for element_name, element_payload in brainstorm.items():
            if not isinstance(element_payload, dict):
                continue
            if "ideas" not in element_payload:
                continue
            ideas = element_payload.get("ideas", [])
            for idea in ideas:
                idea_text = str(idea).strip()
                if not idea_text:
                    continue
                dedupe_key = idea_text.lower()
                if dedupe_key in seen_per_contest[contest_number]:
                    continue
                seen_per_contest[contest_number].add(dedupe_key)
                contest_to_ideas[contest_number].append(
                    {
                        "idea": idea_text,
                        "element": element_name,
                        "record_index": record_index,
                    }
                )
    return contest_to_ideas


def batch(iterable: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield successive slices from *iterable* of length *size*."""
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    *,
    model: str,
    batch_size: int,
) -> np.ndarray:
    """Embed *texts* returning an array of shape (len(texts), embedding_dim)."""
    vectors: List[np.ndarray] = []
    for chunk in batch(texts, batch_size):
        response = client.embeddings.create(model=model, input=list(chunk))
        embeddings = [np.array(item.embedding, dtype=float) for item in response.data]
        vectors.extend(embeddings)

    if not vectors:
        raise RuntimeError("No embeddings were returned by the OpenAI API")

    return np.vstack(vectors)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity for every pair of rows in *a* (captions) and *b* (ideas)."""
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    a_norms[a_norms == 0] = 1.0
    b_norms[b_norms == 0] = 1.0
    a_norm = a / a_norms
    b_norm = b / b_norms
    return np.matmul(a_norm, b_norm.T)


def build_similarity_rows(
    contest_number: str,
    captions: Sequence[str],
    ideas: Sequence[str],
    idea_elements: Sequence[str],
    record_index: Optional[int],
    *,
    client: OpenAI,
    model: str,
    batch_size: int,
) -> Tuple[List[dict], List[dict]]:
    if not captions or not ideas:
        return [], []

    if idea_elements and len(idea_elements) != len(ideas):
        raise ValueError("Length of idea_elements must match length of ideas")

    caption_vectors = embed_texts(client, captions, model=model, batch_size=batch_size)
    idea_vectors = embed_texts(client, ideas, model=model, batch_size=batch_size)
    similarities = cosine_similarity_matrix(caption_vectors, idea_vectors)

    column_metadata = []
    for cap_idx, caption in enumerate(captions):
        column_metadata.append(
            {
                "column": f"caption_{cap_idx+1}_similarity",
                "caption": caption,
                "caption_index": cap_idx,
            }
        )

    rows: List[dict] = []
    for idea_idx, idea in enumerate(ideas):
        row = {
            "contest_number": contest_number,
            "cartoon_index": record_index,
            "idea": idea,
        }
        if idea_elements:
            row["element"] = idea_elements[idea_idx]
        idea_similarities = similarities[:, idea_idx]
        best_idx = int(np.argmax(idea_similarities))
        best_score = float(idea_similarities[best_idx])
        row["best_caption_index"] = best_idx
        row["best_caption"] = captions[best_idx]
        row["max_similarity"] = round(best_score, 4)

        for meta in column_metadata:
            cap_idx = meta["caption_index"]
            row[meta["column"]] = round(float(idea_similarities[cap_idx]), 4)

        rows.append(row)

    return rows, column_metadata


def resolve_results_file(auto_dir: Path, count: int) -> Optional[Path]:
    """If *auto_dir* is provided, choose the most recent results file for the count."""
    if not auto_dir.is_dir():
        return None

    pattern = f"elements_{count}cartoons_"
    candidates = sorted(auto_dir.glob(f"{pattern}*.json"))
    return candidates[-1] if candidates else None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match brainstormed ideas against winning captions using embeddings"
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Path to the JSON results file produced by cartoon_workflow.py",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help=(
            "If --results is not provided, use the most recent auto-saved file for this "
            "cartoon count (default: 1)"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Directory containing auto-saved results (default: results/)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to comprehensive_annotations.csv (default: project root)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help=f"OpenAI embedding model to use (default: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding request batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional path to .env file containing OPENAI_API_KEY",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT_DIR,
        help="Directory where evaluation CSV files will be written (default: evaluation/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top ideas to report separately (default: {DEFAULT_TOP_K})",
    )
    return parser.parse_args(argv)


def build_cartoon_dataset(
    captions_map: Dict[str, List[str]],
    ideas_map: Dict[str, List[dict]],
) -> List[CartoonData]:
    dataset: List[CartoonData] = []
    for contest_number, captions in captions_map.items():
        idea_payloads = ideas_map.get(contest_number, [])
        if not idea_payloads:
            continue
        ideas = [payload["idea"] for payload in idea_payloads]
        elements = [payload.get("element", "") for payload in idea_payloads]
        record_index = next(
            (payload.get("record_index") for payload in idea_payloads if payload.get("record_index") is not None),
            None,
        )
        dataset.append(
            CartoonData(
                contest_number=contest_number,
                record_index=record_index,
                captions=captions,
                ideas=ideas,
                elements=elements,
            )
        )
    return dataset


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    client = load_openai_client(args.env_file)

    results_path = args.results
    if not results_path:
        results_path = resolve_results_file(args.results_dir, args.count)
        if not results_path:
            print(
                "No results file provided and none found in the results directory. "
                "Use --results to supply the path manually."
            )
            return 1

    captions_map = read_captions(args.csv)
    ideas_map = read_brainstorm_results(results_path)
    dataset = build_cartoon_dataset(captions_map, ideas_map)

    if not dataset:
        print("No cartoons had both winning captions and brainstormed ideas to compare.")
        return 0

    all_rows: List[dict] = []
    caption_metadata: Dict[str, List[dict]] = {}

    for cartoon in dataset:
        rows, column_meta = build_similarity_rows(
            cartoon.contest_number,
            cartoon.captions,
            cartoon.ideas,
            cartoon.elements,
            cartoon.record_index,
            client=client,
            model=args.model,
            batch_size=args.batch_size,
        )
        if rows:
            caption_metadata[cartoon.contest_number] = column_meta
            all_rows.extend(rows)

    if not all_rows:
        print("No brainstorm ideas available to evaluate after filtering.")
        return 0

    df = pd.DataFrame(all_rows)
    df_columns = sorted(
        [col for col in df.columns if col.startswith("caption_") and col.endswith("_similarity")],
        key=lambda name: int(name.split("_")[1]),
    )
    ordered_columns = [
        "contest_number",
        "cartoon_index",
        "element",
        "idea",
        *df_columns,
        "max_similarity",
        "best_caption_index",
        "best_caption",
    ]
    df = df[ordered_columns]

    if "cartoon_index" in df.columns:
        df["cartoon_index"] = df["cartoon_index"].astype("Int64")
    if "element" in df.columns:
        df["element"] = df["element"].fillna("")

    top_k = max(1, args.top_k)
    top_df = df.sort_values("max_similarity", ascending=False).head(top_k)

    contest_count = len(set(df["contest_number"]))

    print(f"Evaluating data from {results_path.resolve()}")
    print(f"Contests evaluated: {contest_count}")
    print(f"Total brainstorm ideas evaluated: {len(df)}")
    print(f"Top {top_k} ideas by max similarity:")
    for _, row in top_df.iterrows():
        idx_val = row.get("cartoon_index")
        element_val = row.get("element", "")
        if pd.isna(idx_val):
            idx_display = "?"
        else:
            idx_display = str(int(idx_val))
        print(
            f"  Contest {row['contest_number']} (cartoon index {idx_display}) | element '{element_val}'"
            f" | max_similarity {row['max_similarity']:.4f}\n"
            f"    Idea:    {row['idea']}\n"
            f"    Caption: {row['best_caption']}"
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = args.output_dir
    if base_output_dir and not base_output_dir.exists():
        base_output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = base_output_dir / f"{contest_count}contests_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = run_dir / "matrix.csv"
    top_path = run_dir / f"top{top_k}.csv"

    df.to_csv(matrix_path, index=False)
    top_df.to_csv(top_path, index=False)

    metadata_path = run_dir / "captions.json"
    metadata_path.write_text(json.dumps(caption_metadata, indent=2), encoding="utf-8")

    histogram_path = run_dir / "histogram.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(df["max_similarity"], bins=30, color="#4C72B0", edgecolor="white")
        plt.xlabel("Max similarity per idea")
        plt.ylabel("Count")
        plt.title("Distribution of brainstorm idea similarities")
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=150)
        plt.close()
        print(f"Saved similarity histogram to {histogram_path.resolve()}")
    except ImportError:
        print("matplotlib not installed; skipping histogram plot. Install it with 'pip install matplotlib'.")

    print(f"Saved evaluation run outputs under {run_dir.resolve()}")
    print(f"  - Full similarity matrix: {matrix_path.name}")
    print(f"  - Top {top_k} ideas: {top_path.name}")
    print(f"  - Caption metadata: {metadata_path.name}")
    if histogram_path.exists():
        print(f"  - Similarity histogram: {histogram_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
