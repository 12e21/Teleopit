#!/usr/bin/env python3
"""Rebuild train.npz / val.npz from a filtered manifest (review results).

Reads filtered_manifest.csv (output of export_reviewed_manifest.py),
verifies all NPZ files exist, and merges them into cleaned train/val splits.

Usage:
    python scripts/data/build_dataset_from_review.py \
        --filtered_manifest data/datasets/review/twist2_full/filtered_manifest.csv \
        --output_dir data/datasets/builds/twist2_full_cleaned
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train_mimic.data.dataset_lib import merge_npz_files, utc_now_iso, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned dataset from filtered manifest"
    )
    parser.add_argument(
        "--filtered_manifest", type=str, required=True,
        help="Path to filtered_manifest.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory, e.g. data/datasets/builds/twist2_full_cleaned",
    )
    parser.add_argument(
        "--target_fps", type=int, default=30,
        help="Resample all clips to this FPS (default: 30). "
        "Required when source clips have mixed FPS values.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.filtered_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    if not manifest_path.is_file():
        print(f"ERROR: filtered manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Read filtered manifest
    # Columns: clip_id, source, file_rel, num_frames, fps, resolved_split, resolved_npz_path, weight
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=2):
            rows.append({
                "clip_id": raw["clip_id"].strip(),
                "source": raw["source"].strip(),
                "file_rel": raw["file_rel"].strip(),
                "num_frames": int(raw["num_frames"]),
                "fps": int(raw["fps"]),
                "resolved_split": raw["resolved_split"].strip(),
                "resolved_npz_path": raw["resolved_npz_path"].strip(),
                "weight": float(raw["weight"]),
                "line_no": idx,
            })

    if not rows:
        print("ERROR: filtered manifest has no data rows", file=sys.stderr)
        sys.exit(1)

    # Verify all NPZ files exist
    missing = []
    for row in rows:
        p = Path(row["resolved_npz_path"])
        if not p.is_file():
            # Try resolving from file_rel
            alt = Path(row["file_rel"])
            if not alt.is_absolute():
                alt = PROJECT_ROOT / alt
            if alt.is_file():
                row["resolved_npz_path"] = str(alt.resolve())
            else:
                missing.append(f"  line {row['line_no']}: {row['clip_id']} -> {row['resolved_npz_path']}")

    if missing:
        print(f"ERROR: {len(missing)} NPZ files not found:", file=sys.stderr)
        for m in missing[:20]:
            print(m, file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
        sys.exit(1)

    # Check for mixed FPS
    fps_values = sorted(set(r["fps"] for r in rows))
    if len(fps_values) > 1:
        print(f"[INFO] Mixed FPS detected: {fps_values}. Resampling all clips to {args.target_fps} FPS.")
        if args.target_fps is None:
            print(
                "ERROR: clips have mixed FPS values but --target_fps is not set.\n"
                "Please specify --target_fps (e.g. --target_fps 30).",
                file=sys.stderr,
            )
            sys.exit(1)

    # Split into train / val
    train_rows = [r for r in rows if r["resolved_split"] == "train"]
    val_rows = [r for r in rows if r["resolved_split"] == "val"]

    if not train_rows:
        print("ERROR: no train clips in filtered manifest", file=sys.stderr)
        sys.exit(1)
    if not val_rows:
        print("WARNING: no val clips in filtered manifest", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge train
    print(f"Merging {len(train_rows)} train clips...")
    train_files = [Path(r["resolved_npz_path"]) for r in train_rows]
    train_weights = [r["weight"] for r in train_rows]
    train_stats = merge_npz_files(
        train_files, output_dir / "train.npz",
        target_fps=args.target_fps, weights=train_weights,
    )
    print(f"  train.npz: {train_stats['frames']} frames, {train_stats['duration_s'] / 60:.1f} min")

    # Merge val
    val_stats = None
    if val_rows:
        print(f"Merging {len(val_rows)} val clips...")
        val_files = [Path(r["resolved_npz_path"]) for r in val_rows]
        val_weights = [r["weight"] for r in val_rows]
        val_stats = merge_npz_files(
            val_files, output_dir / "val.npz",
            target_fps=args.target_fps, weights=val_weights,
        )
        print(f"  val.npz: {val_stats['frames']} frames, {val_stats['duration_s'] / 60:.1f} min")

    # Copy manifest into output dir
    import shutil
    shutil.copy2(manifest_path, output_dir / "manifest_resolved.csv")

    # Write build info
    report = {
        "built_at_utc": utc_now_iso(),
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "target_fps": args.target_fps,
        "clip_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "splits": {
            "train": train_stats,
            "val": val_stats,
        },
    }
    write_json(output_dir / "build_info.json", report)

    print(f"\nCleaned dataset built at: {output_dir}")
    print(f"  Total clips: {len(rows)} (train={len(train_rows)}, val={len(val_rows)})")


if __name__ == "__main__":
    main()
