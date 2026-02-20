#!/usr/bin/env python3
"""Generate captions that bridge an element-specific idea to the opposing element."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from itertools import islice
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("The openai package is required. Install it with 'pip install openai'.") from exc

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_RESULTS_PATH = Path("results")
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_PATH / "elements_1cartoons_20251016-144553.json"
DEFAULT_OUTPUT_DIR = Path("captions")
DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_EMBED_BATCH_SIZE = 64


# Prompt template used to generate diverse cross-element "premises" before writing captions.
PREMISE_PROMPT_TEMPLATE = (
    "You're writing premises for New Yorker-style caption candidates. A premise is NOT a noun-idea; it's a specific joke engine.\n\n"
    "Cartoon description:\n{description}\n\n"
    "The cartoon's two core elements are:\n"
    "- {element_a}\n"
    "- {element_b}\n\n"
    "Return strict JSON with key 'premises' as a list of exactly {count} items.\n"
    "Each item must have:\n"
    "- anchor_element: exactly one of the two element strings above (verbatim)\n"
    "- mechanism: one short label like pun, literalism, role-reversal, euphemism, analogy, bureaucratic-speak, anachronism, status-anxiety, misinterpretation, over-literal instruction\n"
    "- setup: 1 sentence describing the assumed situation\n"
    "- twist: 1 sentence describing the collision/angle that makes it funny\n"
    "- target: short phrase of what/who is being mocked (or 'none')\n"
    "- anchor_detail: OPTIONAL short concrete detail from the description to keep it grounded\n\n"
    "Diversity constraints:\n"
    "- Each premise must use a different mechanism.\n"
    "- No two premises may share the same setup or the same twist phrasing.\n"
    "- Avoid repeating the same joke template (e.g. \"X but Y\") across multiple premises.\n"
    "JSON only."
)


# Prompt template used for every caption request. Edit this string to tweak phrasing.
PROMPT_TEMPLATE = (
    "You are a super funny New Yorker-style cartoon caption writer, crafting contest-winning captions. "
    "The cartoons typically involve two distinct elements colliding, and good captions are able to somehow connect them, be it as a play on words, an interesting way they're related, a fanciful way they could interact, overlapping cultural references, combinations of any of these, etc.. "
    "You already brainstormed ideas connected to one core element in the cartoon, and now we're bridging one BRAINSTORM IDEA to the cartoon's other element, the TARGET ELEMENT, in a single witty line. "
    "Keep it clever: there needs to be a legitimate idea behind the caption, delivered saliently and crisply. Make sure it's said with intent. "
    "To give you a taste for the tone of winning captions, here's some examples (detached from their original cartoons so that you can focus on their formatting/style): "
    "\n{examples}\n\n"
    "Cartoon Description:\n{description}\n\n"
    "Anchor element: {source_element}\n"
    "BRAINSTORM IDEA: {idea}\n\n"
    "TARGET ELEMENT: {target_element}\n\n"
    "Write five distinct caption candidates that fuse the BRAINSTORM IDEA with the TARGET ELEMENT. "
    "Label them 1 through 5, each on its own line. "
    "After listing the five options, choose the strongest caption and restate it between <best_caption> and </best_caption> tags. "
    "Do not include any other commentary."
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


def format_example_captions(captions: Sequence[str]) -> str:
    """Return bullet list of example captions for the prompt."""
    cleaned = [caption.strip() for caption in captions if caption and caption.strip()]
    if not cleaned:
        return "- (no examples available)"
    return "\n".join(f"- {caption}" for caption in cleaned)


def clean_caption_text(text: str) -> str:
    """Trim whitespace and surrounding quotes from a caption string."""
    return text.strip().strip(' "\'“”')


def parse_caption_response(response: str) -> tuple[str, List[str], Optional[str]]:
    """Extract the best caption and candidate list from the model response."""
    candidates: List[str] = []
    for line in response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^\s*(\d+)[\.\)]\s*(.+)$", stripped)
        if match:
            candidate = clean_caption_text(match.group(2))
            if candidate:
                candidates.append(candidate)

    best_match = re.search(
        r"<best_caption>(.*?)</best_caption>", response, flags=re.IGNORECASE | re.DOTALL
    )
    best_caption = clean_caption_text(best_match.group(1)) if best_match else None

    error: Optional[str] = None
    if not best_caption:
        if candidates:
            best_caption = candidates[0]
            error = "best_caption tag missing; defaulted to first candidate"
        else:
            best_caption = clean_caption_text(response)
            error = "Unable to parse candidates; using raw response"

    return best_caption, candidates, error


def extract_json_payload(content: str) -> dict:
    """Parse assistant response, tolerating fenced code blocks."""
    text = (content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def normalize_premise_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def premise_fingerprint(premise: dict) -> str:
    mechanism = normalize_premise_text(premise.get("mechanism"))
    setup = normalize_premise_text(premise.get("setup"))
    twist = normalize_premise_text(premise.get("twist"))
    return f"{mechanism}: {setup} // {twist}".strip()


def generate_premises(
    api_key: str,
    *,
    model: str,
    description: str,
    element_a: str,
    element_b: str,
    count: int,
    reasoning_effort: Optional[str],
    max_attempts: int = 2,
) -> List[dict]:
    prompt = PREMISE_PROMPT_TEMPLATE.format(
        description=description,
        element_a=element_a,
        element_b=element_b,
        count=count,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You propose diverse, coherent comedic premises for New Yorker-style captions. "
                "Return strict JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    last_error: Optional[str] = None
    for _attempt in range(max_attempts + 1):
        raw = call_openrouter(
            api_key,
            model,
            messages,
            reasoning_effort=reasoning_effort,
        )
        try:
            payload = extract_json_payload(raw)
        except json.JSONDecodeError as exc:
            last_error = f"Premise JSON parse failed: {exc}"
            continue

        premises = payload.get("premises")
        if not isinstance(premises, list):
            last_error = "Premise JSON missing 'premises' list"
            continue

        cleaned: List[dict] = []
        seen_fingerprints: set[str] = set()
        seen_mechanisms: set[str] = set()
        for premise in premises:
            if not isinstance(premise, dict):
                continue
            anchor_element = normalize_premise_text(premise.get("anchor_element"))
            mechanism = normalize_premise_text(premise.get("mechanism"))
            setup = normalize_premise_text(premise.get("setup"))
            twist = normalize_premise_text(premise.get("twist"))
            target = normalize_premise_text(premise.get("target")) or "none"
            anchor_detail = normalize_premise_text(premise.get("anchor_detail"))

            if anchor_element not in {element_a, element_b}:
                continue
            if not mechanism or not setup or not twist:
                continue

            mech_key = mechanism.lower()
            fp = f"{mech_key}: {setup.lower()} // {twist.lower()}"
            if mech_key in seen_mechanisms:
                continue
            if fp in seen_fingerprints:
                continue

            cleaned.append(
                {
                    "anchor_element": anchor_element,
                    "mechanism": mechanism,
                    "setup": setup,
                    "twist": twist,
                    "target": target,
                    "anchor_detail": anchor_detail,
                }
            )
            seen_mechanisms.add(mech_key)
            seen_fingerprints.add(fp)
            if len(cleaned) >= count:
                break

        if len(cleaned) >= max(1, min(count, 3)):
            return cleaned[:count]

        last_error = (
            f"Premise JSON returned {len(cleaned)} valid unique premises (requested {count})."
        )

    raise RuntimeError(last_error or "Failed to generate premises")


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
                    if api_key:
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
    """Embed texts returning an array of shape (len(texts), embedding_dim)."""
    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=model, input=chunk)
        vectors.extend(np.array(item.embedding, dtype=float) for item in response.data)

    if not vectors:
        raise RuntimeError("No embeddings returned by OpenAI for the provided texts")

    return np.vstack(vectors)


def cosine_similarities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix between rows of a and b."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return np.matmul(a / a_norm, (b / b_norm).T)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate captions that connect each element's ideas to the other element"
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
        help="Optional limit on total directional caption prompts to run",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for caption generation (default: 1)",
    )
    parser.add_argument(
        "--use-premises",
        action="store_true",
        help="Generate diverse cross-element premises first, then write captions from those premises.",
    )
    parser.add_argument(
        "--premise-count",
        type=int,
        default=20,
        help="Number of unique premises to generate when using --use-premises (default: 20)",
    )
    parser.add_argument(
        "--premise-model",
        help="Optional OpenRouter model for premise generation (defaults to --model).",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"OpenAI embedding model for similarity scoring (default: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help=f"Batch size for embedding requests (default: {DEFAULT_EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-env-file",
        default=".env",
        help="Path to .env containing OPENAI_API_KEY for embedding (default: .env)",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip OpenAI embedding similarity scoring (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--offline-smoketest",
        action="store_true",
        help="Do not call any external APIs; generate placeholder premises/captions to exercise the code path.",
    )
    return parser.parse_args(argv)


def resolve_results_file(results_dir: Path) -> Optional[Path]:
    if not results_dir.is_dir():
        return None
    candidates = sorted(results_dir.glob("elements_*cartoons_*.json"))
    return candidates[-1] if candidates else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Get example captions as the last 10 elements of the column 'caption' in comprehensive_annotations.csv
    annotations_path = Path("comprehensive_annotations.csv")
    example_captions: List[str] = []
    captions_by_contest: Dict[str, List[str]] = {}
    if annotations_path.is_file():
        with annotations_path.open("r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                caption = (row.get("caption") or "").strip()
                contest = (row.get("contest_number") or "").strip()
                if caption:
                    example_captions.append(caption)
                if contest and caption:
                    captions_by_contest.setdefault(contest, []).append(caption)
    for contest, captions in list(captions_by_contest.items()):
        unique: List[str] = []
        seen: set[str] = set()
        for caption in captions:
            if caption not in seen:
                unique.append(caption)
                seen.add(caption)
        captions_by_contest[contest] = unique
    example_captions = example_captions[-10:]
    examples_text = format_example_captions(example_captions)

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

    if args.reasoning_effort and not args.model.startswith("openai/gpt-5"):
        print(
            "--reasoning-effort is only supported when using openai/gpt-5 via OpenRouter.",
            flush=True,
        )
        return 1

    if args.offline_smoketest:
        args.skip_similarity = True
        api_key = ""
    else:
        api_key = load_openrouter_key(args.env_file)

    contest_label = str(record.get("contest_number") or f"index_{args.index}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = contest_label.replace("/", "-").replace(" ", "_").replace(".", "_")
    run_dir = base_output_dir / f"directional_contest_{safe_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    used_premises: List[dict] = []
    if args.use_premises:
        premise_model = args.premise_model or args.model
        if args.offline_smoketest:
            used_premises = [
                {
                    "anchor_element": element_a,
                    "mechanism": "literalism",
                    "setup": f"A {element_a.lower()} is being treated like {element_b.lower()}.",
                    "twist": f"The rules and language of {element_a.lower()} are applied to {element_b.lower()}.",
                    "target": "bureaucracy",
                    "anchor_detail": "",
                },
                {
                    "anchor_element": element_b,
                    "mechanism": "status-anxiety",
                    "setup": f"Someone is trying to maintain status in a {element_b.lower()}.",
                    "twist": f"Status rituals clash with the reality of {element_a.lower()}.",
                    "target": "pretension",
                    "anchor_detail": "",
                },
            ][: max(1, args.premise_count)]
        else:
            used_premises = generate_premises(
                api_key,
                model=premise_model,
                description=description,
                element_a=element_a,
                element_b=element_b,
                count=max(1, args.premise_count),
                reasoning_effort=(
                    args.reasoning_effort
                    if premise_model.startswith("openai/gpt-5")
                    else None
                ),
            )
        for premise_index, premise in enumerate(used_premises, start=1):
            anchor = premise["anchor_element"]
            other = element_b if anchor == element_a else element_a
            tasks.append(
                {
                    "direction": "premise",
                    "direction_label": f"{anchor} -> {other}",
                    "source_element": anchor,
                    "target_element": other,
                    "idea": premise_fingerprint(premise),
                    "idea_index": premise_index,
                    "premise": premise,
                }
            )
    else:
        if args.limit_a is not None:
            ideas_a = ideas_a[: args.limit_a]
        if args.limit_b is not None:
            ideas_b = ideas_b[: args.limit_b]

        if not ideas_a and not ideas_b:
            print("No brainstorm ideas available for either element; aborting.")
            return 1

        directions = [
            {
                "slug": "forward",
                "source_element": element_a,
                "target_element": element_b,
                "ideas": ideas_a,
            },
            {
                "slug": "reverse",
                "source_element": element_b,
                "target_element": element_a,
                "ideas": ideas_b,
            },
        ]

        for direction in directions:
            ideas = direction["ideas"]
            source = direction["source_element"]
            target = direction["target_element"]
            label = f"{source} -> {target}"

            if not ideas:
                print(f"No ideas for {source}; skipping direction {label}.")
                continue

            for idea_index, idea in enumerate(ideas, start=1):
                tasks.append(
                    {
                        "direction": direction["slug"],
                        "direction_label": label,
                        "source_element": source,
                        "target_element": target,
                        "idea": idea,
                        "idea_index": idea_index,
                    }
                )

    if not tasks:
        print("No caption tasks to process; exiting.")
        return 1

    if args.limit is not None:
        tasks = list(islice(tasks, args.limit))

    for idx, task in enumerate(tasks, start=1):
        task["task_index"] = idx

    total_tasks = len(tasks)
    output_path = run_dir / "directional_captions.csv"
    error_count = 0
    error_details: List[dict] = []



    def process_task(task: dict) -> dict:
        premise: Optional[dict] = task.get("premise")
        if premise:
            premise_lines = [
                f"Premise mechanism: {premise.get('mechanism','')}",
                f"Premise setup: {premise.get('setup','')}",
                f"Premise twist: {premise.get('twist','')}",
                f"Premise target: {premise.get('target','')}",
            ]
            if premise.get("anchor_detail"):
                premise_lines.append(f"Anchor detail: {premise.get('anchor_detail')}")
            prompt = (
                "You are a super funny New Yorker-style cartoon caption writer, crafting contest-winning captions.\n"
                "Write five distinct caption candidates and then choose the best one.\n\n"
                f"Cartoon Description:\n{description}\n\n"
                f"To give you a taste for the tone of winning captions:\n{examples_text}\n\n"
                f"Anchor element: {task['source_element']}\n"
                f"Target element: {task['target_element']}\n\n"
                + "\n".join(premise_lines)
                + "\n\n"
                "Label them 1 through 5, each on its own line. "
                "After listing the five options, choose the strongest caption and restate it between <best_caption> and </best_caption> tags. "
                "Do not include any other commentary."
            )
        else:
            prompt = PROMPT_TEMPLATE.format(
                description=description,
                examples=examples_text,
                source_element=task["source_element"],
                idea=task["idea"],
                target_element=task["target_element"],
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise, clever New Yorker magazine cartoon captions. "
                    "Follow the user's instructions closely and mark the final choice with <best_caption> tags."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        candidates: List[str] = []
        raw_response = ""
        try:
            if args.offline_smoketest:
                caption = (
                    f"{task['source_element']} / {task['target_element']}: {task['idea']}"
                )
                error = None
            else:
                raw_response = call_openrouter(
                    api_key,
                    args.model,
                    messages,
                    reasoning_effort=args.reasoning_effort,
                )
                caption, candidates, parse_error = parse_caption_response(raw_response)
                error = parse_error
        except Exception as exc:  # broad to retain record
            caption = ""
            error = str(exc)

        return {
            "task_index": task["task_index"],
            "direction": task["direction"],
            "direction_label": task["direction_label"],
            "source_element": task["source_element"],
            "target_element": task["target_element"],
            "idea": task["idea"],
            "idea_index": task["idea_index"],
            "caption": caption,
            "error": error,
            "premise": premise,
            "raw": {
                "source_element": task["source_element"],
                "target_element": task["target_element"],
                "idea": task["idea"],
                "prompt": prompt,
                "response": raw_response,
                "candidates": candidates,
            },
        }

    results = []
    progress = tqdm(total=total_tasks, desc="Generating captions") if tqdm else None

    if args.n_workers <= 1:
        for task in tasks:
            results.append(process_task(task))
            if progress:
                progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
                if progress:
                    progress.update(1)

    if progress:
        progress.close()

    results.sort(key=lambda item: item["task_index"])

    winning_captions = [] if args.skip_similarity else captions_by_contest.get(contest_label, [])
    similarity_error: Optional[str] = None
    if winning_captions:
        caption_indices: List[int] = []
        generated_captions: List[str] = []
        for idx, item in enumerate(results):
            caption_text = item["caption"].strip()
            if caption_text:
                caption_indices.append(idx)
                generated_captions.append(caption_text)

        if generated_captions:
            try:
                openai_client = load_openai_client(args.embed_env_file)
                generated_vectors = embed_texts(
                    openai_client,
                    generated_captions,
                    model=args.embed_model,
                    batch_size=args.embed_batch_size,
                )
                winning_vectors = embed_texts(
                    openai_client,
                    winning_captions,
                    model=args.embed_model,
                    batch_size=args.embed_batch_size,
                )
                similarity_matrix = cosine_similarities(generated_vectors, winning_vectors)

                for local_idx, result_idx in enumerate(caption_indices):
                    sims = similarity_matrix[local_idx]
                    best_idx = int(np.argmax(sims))
                    best_score = float(sims[best_idx])
                    results[result_idx]["winning_caption_index"] = best_idx
                    results[result_idx]["winning_caption_text"] = winning_captions[best_idx]
                    results[result_idx]["winning_similarity"] = round(best_score, 4)
                    if "raw" in results[result_idx]:
                        results[result_idx]["raw"]["winning_similarities"] = [
                            round(float(score), 4) for score in sims
                        ]
                        results[result_idx]["raw"]["winning_captions"] = winning_captions
            except Exception as exc:
                similarity_error = f"Similarity scoring failed: {exc}"
        else:
            similarity_error = "No generated captions available for similarity scoring."
    else:
        similarity_error = "No winning captions found for this contest."

    results.sort(
        key=lambda item: item.get("winning_similarity", float("-inf")), reverse=True
    )

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "source_element",
                "idea",
                "target_element",
                "caption",
                "winning_caption_text",
                "winning_caption_index",
                "winning_similarity",
                "premise_mechanism",
                "premise_setup",
                "premise_twist",
                "premise_target",
                "premise_anchor_detail",
            ]
        )
        for item in results:
            premise = item.get("premise") or {}
            writer.writerow(
                [
                    item["source_element"],
                    item["idea"],
                    item["target_element"],
                    item["caption"],
                    item.get("winning_caption_text", ""),
                    item.get("winning_caption_index", ""),
                    item.get("winning_similarity", ""),
                    premise.get("mechanism", ""),
                    premise.get("setup", ""),
                    premise.get("twist", ""),
                    premise.get("target", ""),
                    premise.get("anchor_detail", ""),
                ]
            )
            if item["error"]:
                error_count += 1
                error_details.append({**item["raw"], "error": item["error"]})

    direction_counts: Dict[str, int] = {}
    for item in results:
        direction_counts[item["direction_label"]] = (
            direction_counts.get(item["direction_label"], 0) + 1
        )

    print(
        f"Processed {total_tasks} directional caption prompts for contest {contest_label}. "
        f"Errors: {error_count}."
    )
    print(f"Captions saved to {output_path.resolve()}")

    summary = {
        "contest_number": contest_label,
        "cartoon_index": args.index if args.contest is None else record.get("index"),
        "elements": [element_a, element_b],
        "ideas_a_count": len(ideas_a),
        "ideas_b_count": len(ideas_b),
        "use_premises": bool(args.use_premises),
        "premise_model": (args.premise_model or args.model) if args.use_premises else None,
        "premises_generated": len(used_premises),
        "captions_generated": total_tasks,
        "direction_counts": direction_counts,
        "errors": error_count,
        "error_details": error_details,
        "model": args.model,
        "captions_per_prompt": 5,
        "example_captions_used": len(example_captions),
        "winning_captions_used": len(winning_captions),
        "similarity_model": args.embed_model if winning_captions else None,
        "similarity_error": similarity_error,
        "prompt_template": PROMPT_TEMPLATE,
        "premise_prompt_template": PREMISE_PROMPT_TEMPLATE if args.use_premises else None,
        "results_json": str(results_path),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run metadata saved to {summary_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
