#!/usr/bin/env python3
"""Build train/val merged NPZ datasets from manifest."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from train_mimic.data.dataset_lib import (
    hash_split,
    load_manifest,
    merge_npz_files,
    resolve_npz_path,
    sha256_file,
    utc_now_iso,
    validate_entries,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged train/val NPZ from dataset manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest CSV")
    parser.add_argument("--dataset_version", type=str, required=True, help="Dataset version tag, e.g. v1")
    parser.add_argument(
        "--npz_root",
        type=str,
        default=".",
        help="Root directory used to resolve manifest file_rel (default: repo root '.')",
    )
    parser.add_argument(
        "--build_root",
        type=str,
        default="data/motion/builds",
        help="Output build root (default: data/motion/builds)",
    )
    parser.add_argument(
        "--val_percent",
        type=int,
        default=5,
        help="Validation split percent for rows with empty split (default: 5)",
    )
    parser.add_argument(
        "--hash_salt",
        type=str,
        default="",
        help="Optional salt for hash-based split assignment",
    )
    parser.add_argument(
        "--skip_validate",
        action="store_true",
        help="Skip pre-build validation (not recommended)",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=0,
        help=(
            "Optional target fps for merge. 0 means disabled (strict same-fps required). "
            "When >0, clips are time-resampled to this fps before merge."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.val_percent <= 0 or args.val_percent >= 100:
        raise ValueError("--val_percent must be between 1 and 99")
    if args.target_fps < 0:
        raise ValueError("--target_fps must be >= 0")
    target_fps = args.target_fps if args.target_fps > 0 else None

    manifest_path = Path(args.manifest).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()
    build_dir = Path(args.build_root).expanduser().resolve() / args.dataset_version
    build_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    if not args.skip_validate:
        report = validate_entries(entries, npz_root)
        report_path = build_dir / "validation_report.json"
        report["manifest"] = str(manifest_path)
        report["npz_root"] = str(npz_root)
        write_json(report_path, report)
        if not report["ok"]:
            print(f"[ERROR] validation failed, see: {report_path}")
            return 1

    split_rows: dict[str, list[Any]] = {"train": [], "val": []}
    split_files: dict[str, list[Path]] = {"train": [], "val": []}

    source_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"clips_train": 0, "clips_val": 0, "frames_train": 0, "frames_val": 0}
    )

    for e in entries:
        if not e.enabled:
            continue
        split = e.split if e.split else hash_split(e.clip_id, args.val_percent, args.hash_salt)
        if split not in {"train", "val"}:
            raise ValueError(f"resolved invalid split '{split}' for clip_id={e.clip_id}")

        npz_path = resolve_npz_path(e.file_rel, npz_root)
        split_rows[split].append((e, split, npz_path))
        split_files[split].append(npz_path)

        src = source_summary[e.source]
        src[f"clips_{split}"] += 1
        src[f"frames_{split}"] += e.num_frames

    if not split_files["train"]:
        raise ValueError("no train clips after split assignment")
    if not split_files["val"]:
        raise ValueError("no val clips after split assignment")

    print(
        f"[INFO] build {args.dataset_version}: train_clips={len(split_files['train'])} "
        f"val_clips={len(split_files['val'])}"
    )

    train_out = build_dir / "merged_train.npz"
    val_out = build_dir / "merged_val.npz"
    train_weights = [e.weight for e, _, _ in split_rows["train"]]
    val_weights = [e.weight for e, _, _ in split_rows["val"]]
    train_stats = merge_npz_files(
        split_files["train"], train_out, target_fps=target_fps, weights=train_weights
    )
    val_stats = merge_npz_files(
        split_files["val"], val_out, target_fps=target_fps, weights=val_weights
    )

    resolved_manifest_path = build_dir / "manifest_resolved.csv"
    with resolved_manifest_path.open("w", encoding="utf-8", newline="") as f:
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
                "resolved_split",
                "resolved_npz_path",
            ]
        )
        for split in ("train", "val"):
            for e, resolved_split, npz_path in split_rows[split]:
                writer.writerow(
                    [
                        e.clip_id,
                        e.source,
                        e.file_rel,
                        e.num_frames,
                        e.fps,
                        e.split,
                        e.weight,
                        int(e.enabled),
                        e.quality_tag,
                        resolved_split,
                        str(npz_path),
                    ]
                )

    build_info = {
        "dataset_version": args.dataset_version,
        "built_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "npz_root": str(npz_root),
        "split_strategy": {
            "name": "hash_by_clip_id_with_override",
            "val_percent": args.val_percent,
            "hash_salt": args.hash_salt,
            "explicit_split_override": True,
        },
        "merge_options": {
            "target_fps": target_fps,
        },
        "totals": {
            "rows_manifest": len(entries),
            "rows_enabled": sum(1 for e in entries if e.enabled),
            "rows_disabled": sum(1 for e in entries if not e.enabled),
        },
        "outputs": {"train": train_stats, "val": val_stats},
        "source_summary": source_summary,
        "artifacts": {
            "manifest_resolved": str(resolved_manifest_path),
            "build_info": str(build_dir / "build_info.json"),
            "validation_report": str(build_dir / "validation_report.json"),
        },
    }
    write_json(build_dir / "build_info.json", build_info)

    print(f"[INFO] train: {train_out}")
    print(f"[INFO] val:   {val_out}")
    print(f"[INFO] build info: {build_dir / 'build_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
