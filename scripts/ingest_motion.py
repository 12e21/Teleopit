#!/usr/bin/env python3
"""Ingest motion files (BVH/PKL/NPZ) into managed NPZ clips + manifest.

Pipeline per file type:
  - .bvh -> retarget .pkl -> .npz
  - .pkl -> .npz
  - .npz -> validate + copy (or reuse if already at target)

Manifest behavior:
  - Append new rows automatically.
  - Existing clip_id causes error by default (fail-fast).
  - Pass --allow_update to update existing rows.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mujoco

from train_mimic.data.dataset_lib import (
    REQUIRED_MANIFEST_COLUMNS,
    inspect_npz,
)


_SUPPORTED_SUFFIXES = {".bvh", ".pkl", ".npz"}


@dataclass(frozen=True)
class MotionInput:
    path: Path
    suffix: str
    rel_no_suffix: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_script_module(script_path: Path, module_name: str) -> Any:
    if not script_path.is_file():
        raise FileNotFoundError(f"script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_inputs(input_path: Path) -> tuple[list[MotionInput], Path]:
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported input suffix: {input_path}")
        return [MotionInput(path=input_path, suffix=suffix, rel_no_suffix=Path(input_path.stem))], input_path.parent

    if not input_path.is_dir():
        raise FileNotFoundError(f"input not found: {input_path}")

    files = sorted(
        p for p in input_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _SUPPORTED_SUFFIXES
        and not (p.suffix.lower() == ".npz" and p.name == "merged.npz")
    )
    if not files:
        raise ValueError(f"no .bvh/.pkl/.npz files found in: {input_path}")

    items: list[MotionInput] = []
    for p in files:
        items.append(
            MotionInput(
                path=p,
                suffix=p.suffix.lower(),
                rel_no_suffix=p.relative_to(input_path).with_suffix(""),
            )
        )
    return items, input_path


def _validate_manifest_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("manifest has no header")
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in fieldnames]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _validate_manifest_header(reader.fieldnames)
        return [dict(r) for r in reader]


def _write_manifest_rows(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in REQUIRED_MANIFEST_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest BVH/PKL/NPZ into managed NPZ clips + manifest.")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Source name in manifest (default: input basename/stem)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/motion/manifests/main.csv",
        help="Manifest CSV path (default: data/motion/manifests/main.csv)",
    )
    parser.add_argument(
        "--npz_root",
        type=str,
        default=".",
        help="Root path used to compute manifest file_rel (default: repo root '.')",
    )
    parser.add_argument(
        "--npz_clips_root",
        type=str,
        default="data/motion/npz_clips",
        help="Managed NPZ clips root (default: data/motion/npz_clips)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        choices=["", "train", "val"],
        help="Manifest split for ingested rows (default: empty, resolved at build)",
    )
    parser.add_argument("--weight", type=float, default=1.0, help="Manifest weight (default: 1.0)")
    parser.add_argument("--quality_tag", type=str, default="raw", help="Manifest quality_tag (default: raw)")
    parser.add_argument(
        "--enabled",
        type=int,
        default=1,
        choices=[0, 1],
        help="Manifest enabled flag (default: 1)",
    )
    parser.add_argument(
        "--allow_update",
        action="store_true",
        help="Allow updating existing manifest rows with same clip_id",
    )
    parser.add_argument(
        "--bvh_format",
        type=str,
        default=None,
        choices=["lafan1", "hc_mocap", "nokov"],
        help="BVH format for .bvh inputs (required when .bvh exists)",
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="unitree_g1",
        help="Robot name used in BVH retargeting (default: unitree_g1)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Max frames per BVH clip for conversion (0=all)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned actions; do not write output files or manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.weight <= 0.0:
        raise ValueError("--weight must be > 0")
    if args.max_frames < 0:
        raise ValueError("--max_frames must be >= 0")

    input_path = Path(args.input).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()
    npz_clips_root = Path(args.npz_clips_root).expanduser().resolve()

    items, scan_root = _collect_inputs(input_path)
    source = args.source.strip() or (input_path.stem if input_path.is_file() else input_path.name)
    if not source:
        raise ValueError("resolved empty source name")

    bvh_items = [x for x in items if x.suffix == ".bvh"]
    pkl_items = [x for x in items if x.suffix == ".pkl"]
    npz_items = [x for x in items if x.suffix == ".npz"]
    if bvh_items and args.bvh_format is None:
        raise ValueError("--bvh_format is required when ingesting .bvh files")

    # One output NPZ per relative stem under <npz_clips_root>/<source>/...
    output_map: dict[Path, MotionInput] = {}
    for item in items:
        out_rel = Path(source) / item.rel_no_suffix.with_suffix(".npz")
        if out_rel in output_map:
            prev = output_map[out_rel]
            raise ValueError(
                f"output collision detected: {prev.path} and {item.path} -> {out_rel}. "
                "Rename files or ingest one format at a time."
            )
        output_map[out_rel] = item

    # Avoid re-ingesting managed outputs when input dir accidentally overlaps npz_clips_root.
    if input_path.is_dir():
        filtered_output_map: dict[Path, MotionInput] = {}
        managed_root = npz_clips_root.resolve()
        for out_rel, item in output_map.items():
            if item.path.resolve().is_relative_to(managed_root):
                continue
            out_abs = (npz_clips_root / out_rel).resolve()
            if item.path.resolve() == out_abs:
                continue
            filtered_output_map[out_rel] = item
        output_map = filtered_output_map
        if not output_map:
            raise ValueError(
                "no ingest candidates after filtering managed outputs; "
                "check --input and --npz_clips_root"
            )

    # Recompute type counters after filtering.
    selected_items = list(output_map.values())
    bvh_items = [x for x in selected_items if x.suffix == ".bvh"]
    pkl_items = [x for x in selected_items if x.suffix == ".pkl"]
    npz_items = [x for x in selected_items if x.suffix == ".npz"]

    print(
        f"[INFO] ingest source={source} files={len(selected_items)} "
        f"(bvh={len(bvh_items)}, pkl={len(pkl_items)}, npz={len(npz_items)})"
    )
    print(f"[INFO] scan root: {scan_root}")
    print(f"[INFO] npz clips root: {npz_clips_root}")
    print(f"[INFO] manifest: {manifest_path}")

    if args.dry_run:
        for out_rel, item in sorted(output_map.items(), key=lambda x: str(x[0])):
            print(f"[DRYRUN] {item.path} -> {(npz_clips_root / out_rel)}")
        return 0

    # Fail-fast manifest conflict check before running any conversions.
    existing_rows = _read_manifest_rows(manifest_path)
    existing_by_clip = {r["clip_id"]: i for i, r in enumerate(existing_rows)}
    existing_by_file = {r["file_rel"]: r["clip_id"] for r in existing_rows}
    for out_rel in sorted(output_map.keys(), key=str):
        out_npz = (npz_clips_root / out_rel).resolve()
        rel_under_source = out_npz.relative_to(npz_clips_root / source).with_suffix("").as_posix()
        clip_id = f"{source}:{rel_under_source}"
        try:
            file_rel = out_npz.relative_to(npz_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"output npz {out_npz} is not under npz_root {npz_root}; adjust --npz_root"
            ) from exc

        if clip_id in existing_by_clip and not args.allow_update:
            raise ValueError(
                f"clip_id already exists in manifest: {clip_id}. "
                "Use --allow_update to replace it."
            )
        if file_rel in existing_by_file and existing_by_file[file_rel] != clip_id:
            raise ValueError(
                f"file_rel already used by another clip_id: {file_rel} "
                f"(existing clip_id={existing_by_file[file_rel]})."
            )

    project_root = _project_root()
    convert_pkl_module = _load_script_module(
        project_root / "train_mimic" / "scripts" / "convert_pkl_to_npz.py",
        "convert_pkl_to_npz_script",
    )
    if not hasattr(convert_pkl_module, "convert_pkl_to_npz"):
        raise RuntimeError("convert_pkl_to_npz.py missing convert_pkl_to_npz()")
    convert_pkl_to_npz_fn = convert_pkl_module.convert_pkl_to_npz

    bvh_module = None
    mj_model = None
    if bvh_items:
        bvh_module = _load_script_module(
            project_root / "scripts" / "convert_bvh_to_retarget_pkl.py",
            "convert_bvh_to_retarget_pkl_script",
        )
        if not hasattr(bvh_module, "_convert_one") or not hasattr(bvh_module, "_mocap_xml_path"):
            raise RuntimeError("convert_bvh_to_retarget_pkl.py missing required helpers")
        xml_path = Path(bvh_module._mocap_xml_path(project_root))
        if not xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))

    created_outputs: list[Path] = []
    with TemporaryDirectory(prefix="teleopit_ingest_pkl_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for out_rel, item in sorted(output_map.items(), key=lambda x: str(x[0])):
            out_npz = (npz_clips_root / out_rel).resolve()
            out_npz.parent.mkdir(parents=True, exist_ok=True)

            if item.suffix == ".npz":
                if item.path.resolve() != out_npz:
                    shutil.copy2(item.path, out_npz)
                inspect_npz(out_npz)
                created_outputs.append(out_npz)
                continue

            if item.suffix == ".pkl":
                convert_pkl_to_npz_fn(str(item.path), str(out_npz))
                inspect_npz(out_npz)
                created_outputs.append(out_npz)
                continue

            if item.suffix == ".bvh":
                assert bvh_module is not None and mj_model is not None
                tmp_pkl = (tmp_root / out_rel).with_suffix(".pkl")
                tmp_pkl.parent.mkdir(parents=True, exist_ok=True)
                bvh_module._convert_one(
                    item.path,
                    tmp_pkl,
                    args.bvh_format,
                    args.robot_name,
                    args.max_frames,
                    mj_model,
                )
                convert_pkl_to_npz_fn(str(tmp_pkl), str(out_npz))
                inspect_npz(out_npz)
                created_outputs.append(out_npz)
                continue

            raise ValueError(f"unexpected suffix: {item.suffix}")

    # Build incoming manifest rows from output NPZ metadata.
    incoming_rows: list[dict[str, str]] = []
    seen_clip_ids = set()
    seen_file_rels = set()
    for out_npz in created_outputs:
        rel_under_source = out_npz.relative_to(npz_clips_root / source).with_suffix("").as_posix()
        clip_id = f"{source}:{rel_under_source}"
        if clip_id in seen_clip_ids:
            raise ValueError(f"duplicate clip_id generated in one run: {clip_id}")
        seen_clip_ids.add(clip_id)

        try:
            file_rel = out_npz.relative_to(npz_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"output npz {out_npz} is not under npz_root {npz_root}; adjust --npz_root"
            ) from exc
        if file_rel in seen_file_rels:
            raise ValueError(f"duplicate file_rel generated in one run: {file_rel}")
        seen_file_rels.add(file_rel)

        meta = inspect_npz(out_npz)
        incoming_rows.append(
            {
                "clip_id": clip_id,
                "source": source,
                "file_rel": file_rel,
                "num_frames": str(meta.num_frames),
                "fps": str(meta.fps),
                "split": args.split,
                "weight": str(float(args.weight)),
                "enabled": str(int(args.enabled)),
                "quality_tag": args.quality_tag,
            }
        )

    existing_rows = _read_manifest_rows(manifest_path)
    existing_by_clip = {r["clip_id"]: i for i, r in enumerate(existing_rows)}
    existing_by_file = {r["file_rel"]: r["clip_id"] for r in existing_rows}

    inserted = 0
    updated = 0
    for row in incoming_rows:
        clip_id = row["clip_id"]
        file_rel = row["file_rel"]

        if clip_id in existing_by_clip:
            if not args.allow_update:
                raise ValueError(
                    f"clip_id already exists in manifest: {clip_id}. "
                    "Use --allow_update to replace it."
                )
            idx = existing_by_clip[clip_id]
            existing_rows[idx] = row
            updated += 1
            continue

        if file_rel in existing_by_file:
            raise ValueError(
                f"file_rel already used by another clip_id: {file_rel} "
                f"(existing clip_id={existing_by_file[file_rel]})."
            )

        existing_rows.append(row)
        existing_by_clip[clip_id] = len(existing_rows) - 1
        existing_by_file[file_rel] = clip_id
        inserted += 1

    existing_rows.sort(key=lambda r: r["clip_id"])
    _write_manifest_rows(manifest_path, existing_rows)

    print(
        f"[INFO] done: created_npz={len(created_outputs)} "
        f"manifest_inserted={inserted} manifest_updated={updated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
