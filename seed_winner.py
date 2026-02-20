#!/usr/bin/env python3
"""Insert the ground-truth winning caption into a directional captions CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Sequence


DEFAULT_ANNOTATIONS = Path("comprehensive_annotations.csv")


def load_winning_caption(annotations_path: Path, contest_number: str) -> str:
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with annotations_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("contest_number") or "").strip() != contest_number:
                continue
            caption = (row.get("caption") or "").strip()
            if caption:
                return caption

    raise RuntimeError(
        f"No winning caption found for contest {contest_number} in {annotations_path}"
    )


def read_directional_rows(csv_path: Path) -> tuple[List[dict], List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"No rows found in {csv_path}")
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_seeded_csv(
    rows: Sequence[dict],
    fieldnames: List[str],
    output_path: Path,
    *,
    winning_caption: str,
    insertion_index: int,
) -> None:
    updated_fieldnames = list(fieldnames)
    if "is_winner" not in updated_fieldnames:
        updated_fieldnames.append("is_winner")

    seeded_rows: List[dict] = []
    for row in rows:
        new_row = dict(row)
        value = str(new_row.get("is_winner") or "").strip().lower()
        new_row["is_winner"] = "1" if value in {"1", "true", "yes"} else "0"
        seeded_rows.append(new_row)

    winner_row = {field: "" for field in updated_fieldnames}
    winner_row.update(
        {
            "source_element": "Contest Winner",
            "idea": "Ground truth caption",
            "target_element": "",
            "caption": winning_caption,
            "is_winner": "1",
        }
    )

    insertion_index = max(0, min(len(seeded_rows), insertion_index))
    seeded_rows.insert(insertion_index, winner_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=updated_fieldnames)
        writer.writeheader()
        writer.writerows(seeded_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert the actual winning caption into a directional captions CSV."
    )
    parser.add_argument(
        "captions_file",
        type=Path,
        help="Path to directional_captions.csv (input).",
    )
    parser.add_argument(
        "--contest",
        required=True,
        help="Contest number used to look up the winning caption.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f"Path to comprehensive_annotations.csv (default: {DEFAULT_ANNOTATIONS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the seeded CSV (default: <dir>/directional_captions_seeded.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for insertion position.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    captions_path = args.captions_file.resolve()
    if not captions_path.is_file():
        print(f"Captions file not found: {captions_path}")
        return 1

    annotations_path = args.annotations.resolve()
    contest_number = str(args.contest).strip()
    if not contest_number:
        print("--contest must be provided.")
        return 1

    if args.seed is not None:
        random.seed(args.seed)

    try:
        winning_caption = load_winning_caption(annotations_path, contest_number)
    except Exception as exc:
        print(f"Failed to load winning caption: {exc}")
        return 1

    try:
        rows, fieldnames = read_directional_rows(captions_path)
    except Exception as exc:
        print(f"Failed to read directional captions: {exc}")
        return 1

    insertion_index = random.randint(0, len(rows))

    output_path = args.output
    if not output_path:
        output_path = captions_path.parent / "directional_captions_seeded.csv"
    output_path = output_path.resolve()

    try:
        write_seeded_csv(
            rows,
            fieldnames,
            output_path,
            winning_caption=winning_caption,
            insertion_index=insertion_index,
        )
    except Exception as exc:
        print(f"Failed to write seeded CSV: {exc}")
        return 1

    print(
        f"Inserted winning caption for contest {contest_number} at position {insertion_index + 1}."
    )
    print(f"Seeded file saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
