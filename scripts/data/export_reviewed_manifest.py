#!/usr/bin/env python3
"""Export a filtered manifest from review results.

Reads review_state.csv, keeps only clips with decision == 'keep',
and outputs filtered_manifest.csv + review_summary.json.

Usage:
    python scripts/data/export_reviewed_manifest.py \
        --review data/datasets/review/twist2_full/review_state.csv \
        --output data/datasets/review/twist2_full/filtered_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train_mimic.data.dataset_lib import write_json
from train_mimic.data.review_lib import (
    compute_review_stats,
    load_review_state,
    utc_now_iso,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export filtered manifest from review")
    parser.add_argument("--review", type=str, required=True, help="Path to review_state.csv")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for filtered_manifest.csv (default: same dir as review)",
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Path for review_summary.json (default: same dir as review)",
    )
    parser.add_argument(
        "--require_complete", action="store_true",
        help="Fail if any clips are unreviewed",
    )
    args = parser.parse_args()

    review_path = Path(args.review)
    if not review_path.is_absolute():
        review_path = (PROJECT_ROOT / review_path).resolve()

    rows = load_review_state(review_path)
    stats = compute_review_stats(rows)

    # Check completeness
    unreviewed = stats.total - stats.reviewed
    if args.require_complete and unreviewed > 0:
        print(
            f"ERROR: {unreviewed} clips are unreviewed. "
            "Complete the review or remove --require_complete.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Filter to keep only
    kept_rows = [r for r in rows if r.decision == "keep"]

    # Output paths
    review_dir = review_path.parent
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (PROJECT_ROOT / output_path).resolve()
    else:
        output_path = review_dir / "filtered_manifest.csv"

    if args.summary:
        summary_path = Path(args.summary)
        if not summary_path.is_absolute():
            summary_path = (PROJECT_ROOT / summary_path).resolve()
    else:
        summary_path = review_dir / "review_summary.json"

    # Write filtered_manifest.csv in manifest_resolved.csv format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "clip_id", "source", "file_rel", "num_frames", "fps",
            "resolved_split", "resolved_npz_path", "weight",
        ])
        for r in sorted(kept_rows, key=lambda x: x.clip_id):
            # Resolve npz path
            p = Path(r.file_rel)
            if not p.is_absolute():
                resolved = str((PROJECT_ROOT / p).resolve())
            else:
                resolved = str(p)
            writer.writerow([
                r.clip_id, r.source, r.file_rel, r.num_frames, r.fps,
                r.resolved_split, resolved, r.weight,
            ])

    # Write summary JSON
    summary = {
        "exported_at_utc": utc_now_iso(),
        "review_file": str(review_path),
        "output_file": str(output_path),
        "total_clips": stats.total,
        "reviewed_clips": stats.reviewed,
        "unreviewed_clips": unreviewed,
        "keep_count": stats.keep_count,
        "drop_count": stats.drop_count,
        "skip_count": stats.skip_count,
        "progress_pct": stats.progress_pct,
        "kept_duration_s": stats.kept_duration_s,
        "kept_duration_min": stats.kept_duration_s / 60.0,
        "kept_train_duration_s": stats.kept_train_duration_s,
        "kept_val_duration_s": stats.kept_val_duration_s,
        "kept_duration_by_source": {
            src: {"duration_s": dur, "duration_min": dur / 60.0}
            for src, dur in sorted(stats.kept_duration_by_source.items())
        },
    }
    write_json(summary_path, summary)

    print(f"Exported filtered manifest: {output_path}")
    print(f"  Kept clips: {len(kept_rows)}")
    print(f"  Kept duration: {stats.kept_duration_s / 60:.1f} min")
    print(f"  Train: {stats.kept_train_duration_s / 60:.1f} min")
    print(f"  Val: {stats.kept_val_duration_s / 60:.1f} min")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
