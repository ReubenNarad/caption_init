#!/usr/bin/env python3
"""Visualize agreement across filtered directional caption runs."""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import List, Sequence

from PIL import Image, ImageDraw, ImageFont


def load_filtered_runs(directory: Path, seeded_only: bool) -> tuple[List[dict], dict[int, str], tuple[int, str] | None]:
    pattern = "filtered_seeded_*.json" if seeded_only else "filtered_*.json"
    files = sorted(directory.glob(pattern))
    if len(files) < 2:
        raise RuntimeError(
            f"Need at least two {pattern} files in {directory} (seeded_only={seeded_only})"
        )

    runs = []
    caption_lookup: dict[int, str] = {}
    winner_info: tuple[int, str] | None = None
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        model = str(data.get("model") or "").strip()
        favorites_set: set[int] = set()
        for entry in data.get("favorites", []):
            if not isinstance(entry, dict):
                continue
            idx = entry.get("caption_index")
            if not isinstance(idx, int):
                continue
            favorites_set.add(idx)
            caption_text = entry.get("caption")
            if isinstance(caption_text, str):
                caption_text = caption_text.strip()
                if caption_text and idx not in caption_lookup:
                    caption_lookup[idx] = caption_text
        captions_file = data.get("captions_file")
        selected_winner = bool(data.get("selected_winner"))
        winner_index = data.get("winner_caption_index")
        winner_caption = data.get("winner_caption_text")
        if winner_index is not None and winner_caption and not winner_info:
            winner_info = (int(winner_index), str(winner_caption))
        if not model or not favorites_set or not captions_file:
            raise RuntimeError(f"{path} missing required fields")
        runs.append(
            {
                "model": model,
                "favorites": favorites_set,
                "file": path,
                "captions_file": Path(captions_file).resolve(),
                "selected_winner": selected_winner,
            }
        )
    return runs, caption_lookup, winner_info


def compute_matrices(runs: Sequence[dict]):
    size = len(runs)
    counts = [[0] * size for _ in range(size)]
    overlap = [[0.0] * size for _ in range(size)]
    sets = [run["favorites"] for run in runs]
    for i in range(size):
        for j in range(size):
            shared = sets[i] & sets[j]
            counts[i][j] = len(shared)
            denom = min(len(sets[i]), len(sets[j])) or 1
            overlap[i][j] = len(shared) / denom
    return counts, overlap


def print_matrix(title: str, labels: Sequence[str], data: Sequence[Sequence[float]]) -> None:
    col_width = max(len(label) for label in labels) + 2
    print(title)
    header = " " * col_width + " ".join(label.ljust(col_width) for label in labels)
    print(header)
    for label, row in zip(labels, data):
        def fmt(value):
            return f"{value:.2f}" if isinstance(value, float) else str(value)
        values = " ".join(fmt(value).ljust(col_width) for value in row)
        print(f"{label.ljust(col_width)}{values}")
    print()


def text_size(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    lines = text.split("\n")
    width = 0
    height = 0
    line_spacing = 4
    for idx, line in enumerate(lines):
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
        else:
            line_w, line_h = font.getsize(line)
        width = max(width, line_w)
        height += line_h
        if idx < len(lines) - 1:
            height += line_spacing
    return width, height


def value_to_color(value: float) -> tuple[int, int, int]:
    """Map 0..1 to a purple palette."""
    value = max(0.0, min(1.0, value))
    # light lavender (#f4f0ff) to deep purple (#4c1d95)
    start = (244, 240, 255)
    end = (76, 29, 149)
    r = int(start[0] + (end[0] - start[0]) * value)
    g = int(start[1] + (end[1] - start[1]) * value)
    b = int(start[2] + (end[2] - start[2]) * value)
    return (r, g, b)


def draw_heatmap(labels: Sequence[str], matrix: Sequence[Sequence[float]], output_path: Path) -> None:
    n = len(labels)
    cell = 110
    margin_left = 180
    margin_top = 120
    margin_right = 140
    margin_bottom = 80
    width = margin_left + cell * n + margin_right
    height = margin_top + cell * n + margin_bottom

    img = Image.new("RGB", (width, height), "#fcfbff")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def draw_multiline_text(pos, text):
        x, y = pos
        lines = text.split("\n")
        line_spacing = 4
        for line in lines:
            draw.text((x, y), line, fill="#1f1b2c", font=font)
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(line)
                line_h = bbox[3] - bbox[1]
            else:
                line_h = font.getsize(line)[1]
            y += line_h + line_spacing

    # Title
    title = "Directional Caption Agreement (Overlap)"
    tw, th = text_size(title, font)
    draw.text(((width - tw) / 2, 20), title, fill="#1f1b2c", font=font)

    # Axis labels
    for idx, label in enumerate(labels):
        display = label.replace("/", "\n")
        w, h = text_size(display, font)
        x = margin_left + idx * cell + cell / 2 - w / 2
        draw_multiline_text((x, 60), display)

    for idx, label in enumerate(labels):
        display = label.replace("/", "\n")
        w, h = text_size(display, font)
        y = margin_top + idx * cell + cell / 2 - h / 2
        draw_multiline_text((40, y), display)

    # Grid cells
    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            color = value_to_color(value)
            x0 = margin_left + j * cell
            y0 = margin_top + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="#ccccdd")
            text = f"{value:.2f}"
            tw, th = text_size(text, font)
            brightness = sum(color) / 3
            text_color = "#ffffff" if brightness < 140 else "#1f1b2c"
            draw.text(
                (x0 + cell / 2 - tw / 2, y0 + cell / 2 - th / 2),
                text,
                fill=text_color,
                font=font,
            )

    img.save(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare filtered caption runs within a directory.")
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing filtered_*.json files (output from filter_directional.py).",
    )
    parser.add_argument(
        "--seeded",
        action="store_true",
        help="Indicate that the directional CSV contained a ground-truth winner row.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directory = args.directory.resolve()
    if not directory.is_dir():
        print(f"Directory not found: {directory}")
        return 1

    try:
        runs, caption_lookup, winner_info = load_filtered_runs(directory, args.seeded)
    except Exception as exc:
        print(f"Failed to load filtered data: {exc}")
        return 1

    if args.seeded and not winner_info:
        print(
            "No ground-truth winner metadata found in the filtered files. "
            "Ensure they were generated from the seeded CSV."
        )
        return 1

    labels = [run["model"] for run in runs]
    count_matrix, overlap_matrix = compute_matrices(runs)

    print_matrix("Shared caption counts:", labels, count_matrix)
    print_matrix("Overlap fraction:", labels, overlap_matrix)

    heatmap_path = directory / "agreement_heatmap.png"
    draw_heatmap(labels, overlap_matrix, heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")

    # Consensus picks
    vote_counts: dict[int, int] = {}
    for run in runs:
        for idx in run["favorites"]:
            vote_counts[idx] = vote_counts.get(idx, 0) + 1

    sorted_votes = sorted(
        vote_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    print("Top captions by agreement:")
    for rank, (caption_index, votes) in enumerate(sorted_votes[:5], start=1):
        print(f"  {rank}. Caption #{caption_index} — {votes} votes")

    if args.seeded and winner_info:
        winner_index, winner_caption = winner_info
        selecting_models = [
            run["model"]
            for run in runs
            if run.get("selected_winner") or winner_index in run["favorites"]
        ]
        print(
            f"\nGround-truth caption #{winner_index} "
            f"was selected by {len(selecting_models)}/{len(runs)} models."
        )
        print(f"  Caption: {winner_caption}")
        if selecting_models:
            print(f"  Models: {', '.join(selecting_models)}")
        else:
            print("  Models: (none)")

    consensus_path = directory / "consensus_captions.csv"
    with consensus_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["caption_index", "caption"])
        for caption_index, _ in sorted_votes[:5]:
            writer.writerow([caption_index, caption_lookup.get(caption_index, "")])
    print(f"Saved consensus list to {consensus_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
