#!/usr/bin/env python3
"""Initialize a review_state.csv from an existing manifest_resolved.csv.

Usage:
    python scripts/data/init_review_manifest.py \
        --dataset twist2_full \
        --manifest data/datasets/builds/twist2_full/manifest_resolved.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train_mimic.data.review_lib import ReviewRow, save_review_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize review state from manifest")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. twist2_full")
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to manifest_resolved.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/datasets/review/{dataset}/review_state.csv)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing review file")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (PROJECT_ROOT / output_path).resolve()
    else:
        output_path = PROJECT_ROOT / "data" / "datasets" / "review" / args.dataset / "review_state.csv"

    if output_path.is_file() and not args.force:
        print(
            f"ERROR: review file already exists: {output_path}\n"
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read manifest_resolved.csv
    # Columns: clip_id, source, file_rel, num_frames, fps, resolved_split, resolved_npz_path, weight
    rows: list[ReviewRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"ERROR: manifest is empty: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        required = ["clip_id", "source", "file_rel", "num_frames", "fps", "resolved_split", "weight"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            print(f"ERROR: manifest missing columns: {missing}", file=sys.stderr)
            sys.exit(1)

        for idx, raw in enumerate(reader, start=2):
            clip_id = raw["clip_id"].strip()
            source = raw["source"].strip()
            file_rel = raw["file_rel"].strip()
            num_frames = int(raw["num_frames"])
            fps = int(raw["fps"])
            resolved_split = raw["resolved_split"].strip()
            weight = float(raw["weight"])
            duration_s = num_frames / fps if fps > 0 else 0.0

            rows.append(
                ReviewRow(
                    clip_id=clip_id,
                    source=source,
                    file_rel=file_rel,
                    resolved_split=resolved_split,
                    num_frames=num_frames,
                    fps=fps,
                    duration_s=duration_s,
                    weight=weight,
                )
            )

    if not rows:
        print("ERROR: manifest has no data rows", file=sys.stderr)
        sys.exit(1)

    save_review_state(rows, output_path)

    # Print summary
    total_duration_s = sum(r.duration_s for r in rows)
    sources = {}
    for r in rows:
        sources[r.source] = sources.get(r.source, 0) + 1

    print(f"Initialized review state: {output_path}")
    print(f"  Total clips: {len(rows)}")
    print(f"  Total duration: {total_duration_s / 60:.1f} min ({total_duration_s / 3600:.1f} h)")
    print(f"  Sources:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count} clips")


if __name__ == "__main__":
    main()
