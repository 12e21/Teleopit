#!/usr/bin/env python3
"""Batch-convert BVH/PKL/NPZ inputs into standard NPZ clips.

This is a raw-to-clips utility. For the full dataset pipeline, prefer:

    python train_mimic/scripts/data/build_dataset.py --spec <yaml>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_mimic.data.dataset_builder import (
    DEFAULT_JOBS,
    SOURCE_TYPES,
    DatasetSourceSpec,
    convert_source_to_npz_clips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw motion files into standard NPZ clips.")
    parser.add_argument("--type", required=True, choices=sorted(SOURCE_TYPES), help="Input source type")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output clips directory")
    parser.add_argument("--source", default="source", help="Logical source name used in logs")
    parser.add_argument("--weight", type=float, default=1.0, help="Optional source weight metadata")
    parser.add_argument("--bvh_format", choices=["lafan1", "hc_mocap", "nokov"], default=None)
    parser.add_argument("--robot_name", default="unitree_g1", help="Robot name for BVH retargeting")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames per BVH clip (0 = all)")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="Number of file-level conversion jobs")
    parser.add_argument("--force", action="store_true", help="Delete the output clips directory before conversion")
    parser.add_argument("--json", action="store_true", help="Print the conversion report as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = DatasetSourceSpec(
        name=args.source,
        type=args.type,
        input=args.input,
        weight=float(args.weight),
        bvh_format=args.bvh_format,
        robot_name=args.robot_name,
        max_frames=int(args.max_frames),
    )
    report = convert_source_to_npz_clips(
        source,
        Path(args.output).expanduser().resolve(),
        force=args.force,
        jobs=args.jobs,
    )
    print(f"[DONE] source={report['source']}")
    print(f"[DONE] clips={report['clips']}")
    print(f"[DONE] output={report['output_dir']}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
