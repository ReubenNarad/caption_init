#!/usr/bin/env python3
"""Compute a pairwise similarity matrix between brainstorm ideas for two elements."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("The openai package is required. Install it with 'pip install openai'.") from exc


DEFAULT_RESULTS_PATH = Path("results")
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_PATH / "elements_1cartoons_20251016-144553.json"
DEFAULT_OUTPUT_DIR = Path("pairwise")
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 64


def load_openai_client(env_path: str = ".env") -> OpenAI:
    """Instantiate an OpenAI client using environment variables or a .env file."""
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
            "OPENAI_API_KEY is not set in the environment or in the provided .env file"
        )

    return OpenAI(api_key=api_key)


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    *,
    model: str,
    batch_size: int,
) -> np.ndarray:
    """Embed *texts* returning an array of shape (len(texts), embedding_dim)."""
    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=model, input=chunk)
        vectors.extend(np.array(item.embedding, dtype=float) for item in response.data)

    if not vectors:
        raise RuntimeError("No embeddings returned by OpenAI for the provided texts")

    return np.vstack(vectors)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cosine similarity matrix between rows of *a* and rows of *b*."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return np.matmul(a / a_norm, (b / b_norm).T)


def load_cartoon_record(
    results_path: Path,
    *,
    index: Optional[int] = None,
    contest_number: Optional[str] = None,
) -> dict:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if contest_number:
        contest_number = str(contest_number).strip()
        for record in data:
            if str(record.get("contest_number") or "").strip() == contest_number:
                return record
        raise ValueError(f"Contest number {contest_number} not found in {results_path}")

    if index is None:
        raise ValueError("Either index or contest_number must be provided")

    if index <= 0 or index > len(data):
        raise IndexError(
            f"Cartoon index {index} out of range for file with {len(data)} entries"
        )
    return data[index - 1]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine similarity between two element idea sets"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=(
            "Path to the results JSON from cartoon_workflow.py "
            f"(default: {DEFAULT_RESULTS_FILE})"
        ),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="1-based index of the cartoon within the results file (default: 1)",
    )
    parser.add_argument(
        "--contest",
        help="Optional contest_number override instead of using --index",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Directory containing auto-saved results (default: results/)",
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
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional path to .env file containing OPENAI_API_KEY",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where similarity outputs will be written (default: pairwise/)",
    )
    return parser.parse_args(argv)


def resolve_results_file(results_dir: Path, contest_count: int | None = None) -> Optional[Path]:
    if not results_dir.is_dir():
        return None
    if contest_count is None:
        candidates = sorted(results_dir.glob("elements_*cartoons_*.json"))
    else:
        candidates = sorted(results_dir.glob(f"elements_{contest_count}cartoons_*.json"))
    return candidates[-1] if candidates else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    results_path = args.results
    if not results_path.exists():
        fallback = resolve_results_file(args.results_dir)
        if fallback:
            print(
                f"Provided results file {results_path} not found; using latest {fallback.name}."
            )
            results_path = fallback
        else:
            print(f"Results file {results_path} does not exist and no fallback was found.")
            return 1

    record = load_cartoon_record(
        results_path,
        index=args.index if not args.contest else None,
        contest_number=args.contest,
    )
    description = record.get("description", "").strip()
    elements = record.get("elements") or []
    brainstorm: Dict[str, dict] = record.get("brainstorm") or {}

    if len(elements) != 2:
        raise RuntimeError(
            f"Expected exactly two elements for contest {args.contest}, found {len(elements)}"
        )

    element_a, element_b = elements
    ideas_a = (brainstorm.get(element_a) or {}).get("ideas") or []
    ideas_b = (brainstorm.get(element_b) or {}).get("ideas") or []

    if not ideas_a or not ideas_b:
        raise RuntimeError(
            "Expected at least one idea for each element. "
            f"Got {len(ideas_a)} for '{element_a}' and {len(ideas_b)} for '{element_b}'."
        )

    client = load_openai_client(args.env_file)

    embeddings_a = embed_texts(client, ideas_a, model=args.model, batch_size=args.batch_size)
    embeddings_b = embed_texts(client, ideas_b, model=args.model, batch_size=args.batch_size)

    similarity = cosine_matrix(embeddings_a, embeddings_b)

    contest_label = str(record.get("contest_number") or "index_{}".format(args.index))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = contest_label.replace("/", "-").replace(" ", "_").replace(".", "_")
    run_dir = base_output_dir / f"contest_{safe_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = run_dir / "similarity_matrix.csv"
    meta_path = run_dir / "metadata.json"

    header = ["idea a", *[idea for idea in ideas_b]]
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header))
        handle.write("\n")
        for idx_a, idea_a in enumerate(ideas_a):
            row_values = [f"{value:.6f}" for value in similarity[idx_a]]
            handle.write(",".join([idea_a, *row_values]))
            handle.write("\n")

    metadata = {
        "contest_number": record.get("contest_number"),
        "cartoon_index": args.index if args.contest is None else record.get("index"),
        "description": description,
        "elements": [element_a, element_b],
        "ideas_a": ideas_a,
        "ideas_b": ideas_b,
        "matrix_file": matrix_path.name,
        "embed_model": args.model,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved similarity matrix to {matrix_path.resolve()}")
    print(f"Saved metadata to {meta_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
