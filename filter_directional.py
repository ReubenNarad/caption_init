#!/usr/bin/env python3
"""Select favorite directional captions via OpenRouter."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5"


def slugify_label(value: str) -> str:
    return (
        value.strip()
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def load_api_key(env_path: str = ".env") -> str:
    """Load the OpenRouter API key from environment or a dotenv file."""
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key.strip()

    raise RuntimeError("OPENROUTER_API_KEY not found in environment or .env file")


def extract_json_payload(content: str) -> dict:
    """Parse assistant response, tolerating fenced code blocks."""
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def invoke_openrouter_json(
    api_key: str,
    model: str,
    messages: List[dict],
    *,
    temperature: float,
    app_id: Optional[str],
    timeout: int = 60,
    extra_body: Optional[dict] = None,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if app_id:
        headers["HTTP-Referer"] = app_id
        headers["X-Title"] = "Directional Caption Favorites"

    payload = {"model": model, "messages": messages, "temperature": temperature}
    if extra_body:
        payload["extra_body"] = extra_body

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter response missing choices")
    content = choices[0].get("message", {}).get("content")
    if not content:
        raise RuntimeError("OpenRouter response missing content")
    return extract_json_payload(content)


def load_run_summary(captions_path: Path) -> dict:
    summary_path = captions_path.parent / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def resolve_cartoon_description(summary: dict) -> Optional[str]:
    results_ref = summary.get("results_json")
    if not results_ref:
        return None

    results_path = Path(results_ref)
    if not results_path.is_absolute():
        results_path = Path.cwd() / results_path
    if not results_path.is_file():
        return None

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    contest_number = str(summary.get("contest_number") or "").strip()
    record = None
    if contest_number:
        for entry in data:
            if str(entry.get("contest_number") or "").strip() == contest_number:
                record = entry
                break

    if record is None:
        index = summary.get("cartoon_index")
        if isinstance(index, int) and 1 <= index <= len(data):
            record = data[index - 1]

    if not record:
        return None

    description = (record.get("description") or "").strip()
    return description or None


def read_directional_captions(
    csv_path: Path,
    desired_count: int,
) -> List[dict]:
    entries: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            caption = (row.get("caption") or "").strip()
            if not caption:
                continue
            entry = {
                "list_index": len(entries) + 1,
                "source_element": (row.get("source_element") or "").strip(),
                "idea": (row.get("idea") or "").strip(),
                "target_element": (row.get("target_element") or "").strip(),
                "caption": caption,
                "is_winner": str(row.get("is_winner") or "").strip().lower()
                in {"1", "true", "yes"},
            }
            entries.append(entry)
            if desired_count and len(entries) >= desired_count:
                break

    if desired_count and len(entries) < desired_count:
        raise RuntimeError(
            f"Requested {desired_count} captions but only found {len(entries)} in {csv_path}"
        )

    if not entries:
        raise RuntimeError(f"No captions found in {csv_path}")

    return entries


def build_selection_prompt(
    entries: Sequence[dict],
    top_k: int,
    *,
    contest_number: Optional[str],
    elements: Sequence[str],
    description: Optional[str],
) -> str:
    header_lines = [
        "You are judging the New Yorker Caption Contest and selecting the strongest winners.",
        f"There are {len(entries)} finalist captions below for contest {contest_number or 'unknown'}.",
        f"Choose your top {top_k} based on wit, originality, and how well they stand alone as winning contest entries.",
        "Each caption is wrapped between [CAPTION #n] and [END #n]. Reference favorites only by those numbers.",
        "Respond with strict JSON: {\"favorites\": [{\"rank\":1,\"caption_index\":12,\"reason\":\"...\"}, ...]}",
        "caption_index must match the number inside the markers. Reasons should be short (<=140 chars).",
    ]
    if description:
        header_lines.append(f"Cartoon description (for context): {description}")

    lines = ["\n".join(header_lines), ""]
    for entry in entries:
        idx = entry["list_index"]
        lines.append(f"[CAPTION #{idx}]")
        lines.append(entry["caption"])
        lines.append(f"[END #{idx}]")
        lines.append("")

    return "\n".join(lines).strip()


def parse_favorites(
    payload: dict,
    *,
    top_k: int,
    caption_map: Dict[int, dict],
) -> List[dict]:
    favorites = payload.get("favorites")
    if not isinstance(favorites, list):
        raise RuntimeError("Response JSON missing 'favorites' list")
    if len(favorites) < top_k:
        raise RuntimeError(
            f"Model returned {len(favorites)} favorites but {top_k} are required"
        )

    normalized: List[dict] = []
    seen_indices: set[int] = set()
    for rank, entry in enumerate(favorites[:top_k], start=1):
        if not isinstance(entry, dict):
            raise RuntimeError("Each favorite must be an object with caption_index metadata")
        caption_index = entry.get("caption_index")
        if not isinstance(caption_index, int):
            raise RuntimeError("Each favorite must include an integer caption_index")
        if caption_index not in caption_map:
            raise RuntimeError(f"caption_index {caption_index} is out of range")
        if caption_index in seen_indices:
            raise RuntimeError(f"caption_index {caption_index} was selected more than once")

        reason = (entry.get("reason") or "").strip()
        normalized.append(
            {
                "rank": rank,
                "caption_index": caption_index,
                "caption": caption_map[caption_index]["caption"],
                "reason": reason,
            }
        )
        seen_indices.add(caption_index)

    return normalized


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask an LLM to pick its favorite directional captions"
    )
    parser.add_argument(
        "captions_file",
        type=Path,
        help="Path to directional_captions.csv produced by the directional workflow",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of favorites to request from the model (default: 5)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for the selection model (default: 0.4)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file containing OPENROUTER_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--app-id",
        default=os.environ.get("OPENROUTER_APP_ID"),
        help="Optional Referer/App ID for OpenRouter attribution",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.top_k <= 0:
        print("--top-k must be greater than zero", flush=True)
        return 1
    if args.temperature < 0:
        print("--temperature must be non-negative", flush=True)
        return 1

    captions_path = args.captions_file
    if not captions_path.is_absolute():
        captions_path = captions_path.resolve()
    if not captions_path.is_file():
        print(f"Captions file not found: {captions_path}", flush=True)
        return 1

    summary = load_run_summary(captions_path)
    description = resolve_cartoon_description(summary)
    elements = summary.get("elements") or []
    contest_number = summary.get("contest_number") or captions_path.parent.name

    try:
        entries = read_directional_captions(captions_path, desired_count=0)
    except Exception as exc:
        print(f"Failed to read captions: {exc}", flush=True)
        return 1

    caption_map = {entry["list_index"]: entry for entry in entries}
    winner_index = next(
        (entry["list_index"] for entry in entries if entry.get("is_winner")), None
    )
    winner_caption_text = (
        caption_map[winner_index]["caption"] if winner_index else None
    )

    prompt = build_selection_prompt(
        entries,
        args.top_k,
        contest_number=str(contest_number),
        elements=elements,
        description=description,
    )

    try:
        api_key = load_api_key(args.env_file)
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1

    try:
        payload = invoke_openrouter_json(
            api_key,
            args.model,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a discerning comedy editor who strictly follows formatting rules and "
                        "always returns valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
            app_id=args.app_id,
        )
    except Exception as exc:
        print(f"Selection request failed: {exc}", flush=True)
        return 1

    try:
        favorites = parse_favorites(payload, top_k=args.top_k, caption_map=caption_map)
    except Exception as exc:
        print(f"Failed to parse model response: {exc}", flush=True)
        return 1

    winner_selected = bool(
        winner_index and any(item["caption_index"] == winner_index for item in favorites)
    )
    if winner_index:
        if winner_selected:
            print("Ground-truth winning caption was selected by this model.")
        else:
            print("Ground-truth winning caption was NOT selected.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    captions_dir = captions_path.parent
    filename_prefix = "filtered_seeded_" if winner_index else "filtered_"
    output_name = f"{filename_prefix}{slugify_label(args.model)}_{timestamp}.json"
    output_path = captions_dir / output_name

    result_payload = {
        "contest_number": contest_number,
        "captions_file": str(captions_path),
        "captions_count": len(entries),
        "selected_count": len(favorites),
        "model": args.model,
        "temperature": args.temperature,
        "timestamp": timestamp,
        "favorites": favorites,
        "elements": elements,
        "description": description,
        "winner_caption_index": winner_index,
        "selected_winner": winner_selected,
        "winner_caption_text": winner_caption_text,
        "seeded": bool(winner_index),
    }

    Path(output_path).write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    print(f"Contest {contest_number}: {len(entries)} captions evaluated, top {len(favorites)} selected.")
    print(f"Saved favorites to {output_path}")
    for fav in favorites:
        reason_display = f" — {fav['reason']}" if fav["reason"] else ""
        print(f"  {fav['rank']}. (#{fav['caption_index']}) {fav['caption']}{reason_display}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
