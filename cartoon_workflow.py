#!/usr/bin/env python3
"""Extract cartoon elements and brainstorm related things via OpenRouter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from datetime import datetime

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_INPUT_FILE = "comprehensive_annotations.csv"
DEFAULT_BRAINSTORM_BATCH = 10
DEFAULT_BRAINSTORM_MODELS: List[str] = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.1",
    "openai/gpt-4o",
    "google/gemini-2.5-pro",
    "z-ai/glm-4.6",
    "moonshotai/kimi-k2",
    "openai/gpt-5",
    "x-ai/grok-4",
    "openai/gpt-4.5-preview",
]
# DEFAULT_BRAINSTORM_MODELS: List[str] = [
#     "anthropic/claude-3.5-sonnet",
#     "anthropic/claude-sonnet-4.5",
#     "anthropic/claude-opus-4.1",
# ]

DEFAULT_BRAINSTORM_TOTAL = DEFAULT_BRAINSTORM_BATCH * len(DEFAULT_BRAINSTORM_MODELS)


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


def build_element_prompt(description: str) -> str:
    return (
        "You will receive a description of a single New Yorker cartoon. "
        "Identify the main incongruous elements that make the cartoon interesting. "
        "There will typically be two (and rarely 3) elements that are the main contrasting ideas in the cartoon. "
        "Respond with a JSON object containing an 'elements' array holding concise "
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


def extract_list_from_payload(
    payload: dict,
    primary_key: str,
    *,
    alternatives: Sequence[str] = (),
) -> List[str]:
    """Return the first non-empty list under any of the provided keys."""
    for key in (primary_key, *alternatives):
        value = payload.get(key)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                return cleaned
    raise RuntimeError(
        f"Parsed response missing non-empty list for keys {[primary_key, *alternatives]}: {payload}"
    )


def invoke_openrouter_json(
    api_key: str,
    model: str,
    app_id: Optional[str],
    messages: List[dict],
    *,
    temperature: float = 0.3,
    timeout: int = 45,
) -> dict:
    """Call OpenRouter with provided messages and return parsed JSON content."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if app_id:
        headers["HTTP-Referer"] = app_id
        headers["X-Title"] = "Cartoon Element Workflow"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
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
        return extract_json_payload(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse OpenRouter JSON content: {content}") from exc


def extract_elements(api_key: str, description: str, model: str, app_id: Optional[str]) -> List[str]:
    """Send the description to OpenRouter and return extracted elements."""
    parsed = invoke_openrouter_json(
        api_key,
        model,
        app_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze New Yorker cartoons to label their core elements. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": build_element_prompt(description),
            },
        ],
        temperature=0.3,
    )

    elements = parsed.get("elements")
    if not isinstance(elements, list):
        raise RuntimeError(f"Parsed response missing 'elements' list: {parsed}")

    cleaned = [str(element).strip() for element in elements if str(element).strip()]
    if not cleaned:
        raise RuntimeError(f"Received empty elements list: {parsed}")

    return cleaned


def build_brainstorm_prompt(element: str, existing: List[str], request_count: int) -> str:
    items_clause = "[" + ", ".join(existing) + "]" if existing else "[]"
    directive = (
        f"We're doing this in chunks to keep it diverse. Here's the ones we have so far, {items_clause}, make {request_count} new, DIFFERENT ones and append them. "
        "Emphasis on diversity, do not repeat any of the ones we have so far."
    )
    return (
        "You are a highly creative brainstorming assistant for cartoon caption ideation. "
        "Your job is to brainstorm concise things related to the given element. "
        f"The element we are branching from is: '{element}'. "
        "Each idea can be a person, object, scenario, setting, cultural reference, or phrase. "
        "Most should be just a few words (max 8 words). "
        f"{directive} Return JSON with a 'things' array."
    )


def brainstorm_batch(
    api_key: str,
    element: str,
    existing: List[str],
    model: str,
    app_id: Optional[str],
    request_count: int,
) -> List[str]:
    print(f"Brainstorming for {element} with model: {model}")
    parsed = invoke_openrouter_json(
        api_key,
        model,
        app_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a creative brainstorming assistant. Produce only JSON per instructions."
                ),
            },
            {
                "role": "user",
                "content": build_brainstorm_prompt(element, existing, request_count),
            },
        ],
        temperature=0.8,
    )

    return extract_list_from_payload(parsed, "things", alternatives=("ideas", "items"))


def brainstorm_things_for_element(
    api_key: str,
    element: str,
    app_id: Optional[str],
    *,
    total_desired: int,
    batch_size: int,
    models: List[str],
) -> List[str]:
    if total_desired <= 0:
        return []

    available_models = [model for model in models if model]
    if not available_models:
        raise RuntimeError("No brainstorming models were provided")

    collected: List[str] = []
    seen_lower: set[str] = set()
    disabled_models: List[str] = []
    model_index = 0

    while len(collected) < total_desired:
        if not available_models:
            disabled_note = f" Disabled models: {', '.join(disabled_models)}." if disabled_models else ""
            raise RuntimeError(
                "No brainstorming models remain available after repeated failures." + disabled_note
            )

        model = available_models[model_index % len(available_models)]
        remaining = total_desired - len(collected)
        request_count = batch_size
        try:
            batch = brainstorm_batch(
                api_key,
                element,
                collected,
                model,
                app_id,
                request_count,
            )
        except requests.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response is not None else None
            if status_code in {401, 403, 404}:
                print(f"  ! Skipping model {model} due to HTTP {status_code} from OpenRouter")
                disabled_models.append(model)
                available_models.pop(model_index % len(available_models))
                continue
            raise RuntimeError(
                f"Brainstorm request failed for model {model}: HTTP error {status_code}"
            ) from http_err
        except Exception as exc:
            raise RuntimeError(f"Brainstorm request failed for model {model}: {exc}") from exc

        model_index += 1

        new_items = []
        for item in batch:
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_lower:
                continue
            new_items.append((normalized, key))

        if not new_items:
            raise RuntimeError(
                f"Brainstorm model {model} returned no new unique things (received: {batch})"
            )

        for value, key in new_items[:remaining]:
            collected.append(value)
            seen_lower.add(key)

    return collected


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
    parser.add_argument(
        "--brainstorm-total",
        type=int,
        default=DEFAULT_BRAINSTORM_TOTAL,
        help=(
            "Total number of brainstormed things per element (default: "
            f"{DEFAULT_BRAINSTORM_TOTAL}; set to 0 to skip the brainstorming phase)"
        ),
    )
    parser.add_argument(
        "--brainstorm-batch",
        type=int,
        default=DEFAULT_BRAINSTORM_BATCH,
        help=(
            "Number of things to request per brainstorming call (default: "
            f"{DEFAULT_BRAINSTORM_BATCH})"
        ),
    )
    parser.add_argument(
        "--output-elements",
        type=Path,
        help="Write extracted elements (and brainstorming results, if any) to this JSON file",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip writing extracted elements to disk",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.count <= 0:
        print("--count must be greater than zero", file=sys.stderr)
        return 1
    if args.brainstorm_total < 0:
        print("--brainstorm-total cannot be negative", file=sys.stderr)
        return 1
    if args.brainstorm_batch <= 0:
        print("--brainstorm-batch must be greater than zero", file=sys.stderr)
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

    results: List[dict] = []

    for index, row in enumerate(read_descriptions(csv_path, args.count), start=1):
        description = row["description"].strip()
        label = format_row_label(row, index)
        print(f"=== {label} ===")
        record = {
            "index": index,
            "contest_number": row.get("contest_number"),
            "cartoonstock_id": row.get("cartoonstock_id"),
            "description": description,
        }
        try:
            elements = extract_elements(api_key, description, args.model, args.app_id)
        except Exception as exc:  # catching broad to keep script running
            print(f"Error extracting elements: {exc}")
            record["error"] = str(exc)
            results.append(record)
            continue

        record["elements"] = elements

        if args.brainstorm_total > 0:
            brainstorm_results: dict[str, dict] = {}
            for element in elements:
                print(f"- Element: {element}")
                try:
                    things = brainstorm_things_for_element(
                        api_key,
                        element,
                        args.app_id,
                        total_desired=args.brainstorm_total,
                        batch_size=args.brainstorm_batch,
                        models=DEFAULT_BRAINSTORM_MODELS,
                    )
                except Exception as exc:  # broad to keep processing other elements
                    print(f"  Brainstorm error: {exc}")
                    brainstorm_results[element] = {"error": str(exc)}
                    continue

                brainstorm_results[element] = {"ideas": things}
                for thing in things:
                    print(f"  * {thing}")
            record["brainstorm"] = brainstorm_results
        else:
            for element in elements:
                print(f"- {element}")

        results.append(record)

    if args.no_output:
        output_path = None
    elif args.output_elements:
        output_path = args.output_elements
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path("results") / f"elements_{args.count}cartoons_{timestamp}.json"

    if output_path:
        output_path = Path(output_path)
        try:
            if output_path.parent and not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Saved element data to {output_path.resolve()}")
        except Exception as exc:
            print(f"Failed to write results to {output_path}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
