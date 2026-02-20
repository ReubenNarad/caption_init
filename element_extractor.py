#!/usr/bin/env python3
"""Legacy one-step element extractor.

Prefer `cartoon_workflow.py` for the current pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_INPUT_FILE = "comprehensive_annotations.csv"


def load_api_key(env_path: str = ".env") -> str:
    """Load the OpenRouter API key from environment or a .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()

    path = Path(env_path)
    if not path.is_file():
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set and .env file was not found at "
            f"{path.resolve()}"
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "OPENROUTER_API_KEY":
            value = value.strip().strip('"').strip("'")
            if not value:
                break
            return value

    raise RuntimeError("OPENROUTER_API_KEY not found in environment or .env file")


def read_descriptions(csv_path: Path, limit: int) -> Iterable[dict]:
    """Yield unique cartoon rows keyed by contest_number up to the limit."""
    seen_contests: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for row in reader:
            description = (row.get("description") or "").strip()
            if not description:
                continue
            contest = (row.get("contest_number") or "").strip()
            if contest:
                if contest in seen_contests:
                    continue
                seen_contests.add(contest)
            yield row
            count += 1
            if count >= limit:
                break


def build_user_prompt(description: str) -> str:
    return (
        "You will receive a description of a single New Yorker cartoon. "
        "Identify the main incongruous elements that make the cartoon interesting. "
        "There will typically be two (and rarely 3) elements, that are the main contrasting ideas in the cartoon."
        "Respond with a JSON object containing a 'elements' array holding concise "
        "element names (each four words or fewer). Do not include any commentary.\n\n"
        f"Description: {description}"
    )


def extract_json_payload(content: str) -> dict:
    """Parse the assistant response, tolerating fenced code blocks."""
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def call_openrouter(api_key: str, description: str, model: str, app_id: Optional[str]) -> List[str]:
    """Send the description to OpenRouter and return extracted elements."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if app_id:
        headers["HTTP-Referer"] = app_id
        headers["X-Title"] = "Cartoon element Extractor"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You analyze New Yorker cartoons to label their core elements. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": build_user_prompt(description),
            },
        ],
        "temperature": 0.3,
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=45)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices")
    if not choices:
        raise RuntimeError("OpenRouter response did not include choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError("OpenRouter response did not include content")

    try:
        parsed = extract_json_payload(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse OpenRouter JSON content: {content}") from exc

    elements = parsed.get("elements")
    if not isinstance(elements, list):
        raise RuntimeError(f"Parsed response missing 'elements' list: {parsed}")

    cleaned = [str(element).strip() for element in elements if str(element).strip()]
    if not cleaned:
        raise RuntimeError(f"Received empty elements list: {parsed}")

    return cleaned


def format_row_label(row: dict, index: int) -> str:
    cartoon_id = (row.get("cartoonstock_id") or "").strip()
    contest_number = (row.get("contest_number") or "").strip()
    if cartoon_id:
        return f"Row {index} (cartoonstock_id={cartoon_id})"
    if contest_number:
        return f"Row {index} (contest_number={contest_number})"
    return f"Row {index}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract elements for New Yorker cartoons")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of cartoons to process (default: 1)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(DEFAULT_INPUT_FILE),
        help=f"Path to the CSV file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file containing OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--app-id",
        default=os.environ.get("OPENROUTER_APP_ID"),
        help="Optional App/Referer identifier for OpenRouter usage tracking",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.count <= 0:
        print("--count must be greater than zero", file=sys.stderr)
        return 1

    csv_path = args.csv
    if not csv_path.is_file():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        api_key = load_api_key(args.env_file)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for index, row in enumerate(read_descriptions(csv_path, args.count), start=1):
        description = row["description"].strip()
        label = format_row_label(row, index)
        print(f"=== {label} ===")
        try:
            elements = call_openrouter(api_key, description, args.model, args.app_id)
        except Exception as exc:  # catching broad to keep script running
            print(f"Error extracting elements: {exc}")
            continue

        for element in elements:
            print(f"- {element}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
