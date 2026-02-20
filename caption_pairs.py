#!/usr/bin/env python3
"""Generate captions for every pair of ideas across two elements using an LLM."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_RESULTS_PATH = Path("results")
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_PATH / "elements_1cartoons_20251016-144553.json"
DEFAULT_OUTPUT_DIR = Path("captions")
DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_TEMPERATURE = 0.7


# Prompt template used for every caption request. Edit this string to tweak phrasing.
PROMPT_TEMPLATE = (
    "You are a super funny New Yorker-style cartoon caption writer, writing WINNING captions for the cartoon caption contest. Your job is to take the cartoon and the given ideas about the cartoon, and fuse it all together into a coherent and hilarious caption. We like dry, we like deadpan, we like extreme, we do not like cringey LLM humor."
    "Cartoon Description:\n{description}\n\n"
    "Two core elements show the cartoon's incongruity:\n"
    "1. {element_a} → idea: {idea_a}\n\n"
    "2. {element_b} → idea: {idea_b}\n\n"
    "Write one witty New Yorker-style caption that weaves these two ideas together. Return only the caption, no other text."
)


def load_openrouter_key(env_path: str = ".env") -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key.strip()

    dotenv_path = Path(env_path)
    if dotenv_path.is_file():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value

    raise RuntimeError("OPENROUTER_API_KEY not found in environment or .env file")


def call_openrouter(
    api_key: str,
    model: str,
    messages: List[dict],
    *,
    temperature: float,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 60,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

    for attempt in range(1, max_retries + 1):
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("OpenRouter response missing choices")
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise RuntimeError("OpenRouter response missing content")
            return content.strip()

        if attempt == max_retries:
            response.raise_for_status()
        sleep_time = min(4 * attempt, 10)
        time.sleep(sleep_time)

    raise RuntimeError("OpenRouter request failed after retries")


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
        description="Generate captions for every cross-element idea pair"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=(
            "Path to the results JSON produced by cartoon_workflow.py "
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
        default=DEFAULT_MODEL,
        help=f"OpenRouter model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for the caption model (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "medium", "maximal"],
        help="Optional reasoning effort for openai/gpt-5 runs via OpenRouter",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional path to .env file containing OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where caption outputs will be written (default: captions/)",
    )
    parser.add_argument(
        "--limit-a",
        type=int,
        help="Optional limit on number of ideas taken from the first element",
    )
    parser.add_argument(
        "--limit-b",
        type=int,
        help="Optional limit on number of ideas taken from the second element",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on total unique pairs to process (use upper triangle ordering)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for caption generation (default: 1)",
    )
    return parser.parse_args(argv)


def resolve_results_file(results_dir: Path) -> Optional[Path]:
    if not results_dir.is_dir():
        return None
    candidates = sorted(results_dir.glob("elements_*cartoons_*.json"))
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

    if args.limit_a is not None:
        ideas_a = ideas_a[: args.limit_a]
    if args.limit_b is not None:
        ideas_b = ideas_b[: args.limit_b]

    if not ideas_a or not ideas_b:
        print("No ideas available for one or both elements; aborting.")
        return 1

    api_key = load_openrouter_key(args.env_file)

    contest_label = str(record.get("contest_number") or f"index_{args.index}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = contest_label.replace("/", "-").replace(" ", "_").replace(".", "_")
    run_dir = base_output_dir / f"contest_{safe_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "captions.csv"
    error_count = 0
    error_details: List[dict] = []

    upper_triangle = len(ideas_a) == len(ideas_b)

    def pair_generator():
        pair_index = 0
        for idx_a, idea_a in enumerate(ideas_a, start=1):
            for idx_b, idea_b in enumerate(ideas_b, start=1):
                if upper_triangle and idx_b < idx_a:
                    continue
                pair_index += 1
                yield pair_index, idx_a, idea_a, idx_b, idea_b

    if args.limit:
        pairs = list(islice(pair_generator(), args.limit))
    else:
        pairs = list(pair_generator())

    total_pairs = len(pairs)

    def process_pair(pair):
        pair_index, idx_a, idea_a, idx_b, idea_b = pair
        prompt = PROMPT_TEMPLATE.format(
            description=description,
            element_a=element_a,
            idea_a=idea_a,
            element_b=element_b,
            idea_b=idea_b,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise, clever New Yorker magazine cartoon captions. "
                    "Respond with a single caption line without quotation marks."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            caption = call_openrouter(
                api_key,
                args.model,
                messages,
                temperature=args.temperature,
            )
            error = None
        except Exception as exc:  # broad to retain record
            caption = ""
            error = str(exc)

        return {
            "pair_index": pair_index,
            "idea1": idea_a,
            "idea2": idea_b,
            "caption": caption,
            "error": error,
            "raw": {
                "idea_1": idea_a,
                "element_1": element_a,
                "idea_2": idea_b,
                "element_2": element_b,
                "prompt": prompt,
            },
        }

    results = []
    progress = tqdm(total=total_pairs, desc="Generating captions") if tqdm else None

    if args.n_workers <= 1:
        for pair in pairs:
            results.append(process_pair(pair))
            if progress:
                progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(process_pair, pair): pair for pair in pairs}
            for future in as_completed(futures):
                results.append(future.result())
                if progress:
                    progress.update(1)

    if progress:
        progress.close()

    results.sort(key=lambda item: item["pair_index"])

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([element_a, element_b, "caption"])
        for item in results:
            writer.writerow([item["idea1"], item["idea2"], item["caption"]])
            if item["error"]:
                error_count += 1
                error_details.append(
                    {
                        **item["raw"],
                        "error": item["error"],
                    }
                )

        print(
            f"Processed {total_pairs} unique pairs from contest {contest_label}. "
            f"Errors: {error_count}."
        )
        print(f"Captions saved to {output_path.resolve()}")

    summary = {
        "contest_number": contest_label,
        "cartoon_index": args.index if args.contest is None else record.get("index"),
        "elements": [element_a, element_b],
        "ideas_a_count": len(ideas_a),
        "ideas_b_count": len(ideas_b),
        "pairs": total_pairs,
        "errors": error_count,
        "error_details": error_details,
        "model": args.model,
        "temperature": args.temperature,
        "prompt_template": PROMPT_TEMPLATE,
        "results_json": str(results_path),
        "upper_triangle": upper_triangle,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run metadata saved to {summary_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
