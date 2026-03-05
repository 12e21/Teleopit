#!/usr/bin/env python3
"""Create a manifest CSV from existing legacy NPZ clip directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy NPZ directory into dataset manifest CSV.")
    parser.add_argument(
        "--input_npz_dir",
        type=str,
        required=True,
        help="Directory containing per-clip NPZ files (recursively scanned)",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        required=True,
        help="Output manifest CSV path",
    )
    parser.add_argument(
        "--npz_root",
        type=str,
        default=".",
        help="Root path for computing file_rel (default: repo root '.')",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Source name in manifest (default: basename(input_npz_dir))",
    )
    parser.add_argument(
        "--default_split",
        type=str,
        default="",
        choices=["", "train", "val"],
        help="Default split for all migrated rows (default: empty, hash-split later)",
    )
    parser.add_argument(
        "--default_weight",
        type=float,
        default=1.0,
        help="Default weight for migrated rows (default: 1.0)",
    )
    parser.add_argument(
        "--default_quality_tag",
        type=str,
        default="legacy",
        help="Default quality_tag for migrated rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_npz_dir).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_npz_dir not found: {input_dir}")
    if args.default_weight <= 0:
        raise ValueError("--default_weight must be > 0")

    source = args.source.strip() or input_dir.name
    npz_files = sorted(p for p in input_dir.rglob("*.npz") if p.name != "merged.npz")
    if not npz_files:
        raise ValueError(f"no clip npz files found in: {input_dir}")

    rows: list[list[object]] = []
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        num_frames = int(np.asarray(data["joint_pos"]).shape[0])
        fps = int(data["fps"])

        try:
            file_rel = npz_path.relative_to(npz_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"file {npz_path} is not under npz_root {npz_root}; "
                "please adjust --npz_root"
            ) from exc

        clip_id = f"{source}:{npz_path.relative_to(input_dir).with_suffix('').as_posix()}"
        rows.append(
            [
                clip_id,
                source,
                file_rel,
                num_frames,
                fps,
                args.default_split,
                float(args.default_weight),
                1,
                args.default_quality_tag,
            ]
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "source",
                "file_rel",
                "num_frames",
                "fps",
                "split",
                "weight",
                "enabled",
                "quality_tag",
            ]
        )
        writer.writerows(rows)

    print(f"[INFO] source: {source}")
    print(f"[INFO] clips: {len(rows)}")
    print(f"[INFO] manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

