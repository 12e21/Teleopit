#!/usr/bin/env python3
"""Shared utilities for motion-dataset manifest/validate/build scripts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_MANIFEST_COLUMNS = [
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

REQUIRED_NPZ_KEYS = [
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_names",
]

NUM_ACTIONS = 29


@dataclass(frozen=True)
class ManifestEntry:
    clip_id: str
    source: str
    file_rel: str
    num_frames: int
    fps: int
    split: str
    weight: float
    enabled: bool
    quality_tag: str
    line_no: int


@dataclass(frozen=True)
class NpzMeta:
    fps: int
    num_frames: int
    num_bodies: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def parse_enabled(raw: str, *, line_no: int) -> bool:
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"line {line_no}: enabled must be 0/1/true/false, got '{raw}'")


def normalize_split(raw: str, *, line_no: int) -> str:
    v = raw.strip().lower()
    if v in {"", "train", "val"}:
        return v
    raise ValueError(f"line {line_no}: split must be '', 'train', or 'val', got '{raw}'")


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest is empty: {manifest_path}")
        missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"manifest missing required columns: {missing_cols}")

        entries: list[ManifestEntry] = []
        for idx, row in enumerate(reader, start=2):
            try:
                clip_id = (row["clip_id"] or "").strip()
                source = (row["source"] or "").strip()
                file_rel = (row["file_rel"] or "").strip()
                if not clip_id:
                    raise ValueError(f"line {idx}: clip_id is empty")
                if not source:
                    raise ValueError(f"line {idx}: source is empty")
                if not file_rel:
                    raise ValueError(f"line {idx}: file_rel is empty")

                num_frames = int((row["num_frames"] or "").strip())
                if num_frames <= 0:
                    raise ValueError(f"line {idx}: num_frames must be > 0")
                fps = int((row["fps"] or "").strip())
                if fps <= 0:
                    raise ValueError(f"line {idx}: fps must be > 0")

                split = normalize_split(row["split"] or "", line_no=idx)
                weight = float((row["weight"] or "").strip())
                if weight <= 0.0:
                    raise ValueError(f"line {idx}: weight must be > 0")
                enabled = parse_enabled(row["enabled"] or "", line_no=idx)
                quality_tag = (row["quality_tag"] or "").strip()
            except Exception as exc:
                raise ValueError(str(exc)) from exc

            entries.append(
                ManifestEntry(
                    clip_id=clip_id,
                    source=source,
                    file_rel=file_rel,
                    num_frames=num_frames,
                    fps=fps,
                    split=split,
                    weight=weight,
                    enabled=enabled,
                    quality_tag=quality_tag,
                    line_no=idx,
                )
            )

    if not entries:
        raise ValueError(f"manifest has no rows: {manifest_path}")
    return entries


def resolve_npz_path(file_rel: str, npz_root: Path) -> Path:
    p = Path(file_rel).expanduser()
    if p.is_absolute():
        return p
    return (npz_root / p).resolve()


def inspect_npz(path: Path) -> NpzMeta:
    if not path.is_file():
        raise FileNotFoundError(f"npz not found: {path}")

    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    missing = [k for k in REQUIRED_NPZ_KEYS if k not in keys]
    if missing:
        raise ValueError(f"missing NPZ keys: {missing}")

    fps = int(data["fps"])
    if fps <= 0:
        raise ValueError(f"invalid fps={fps}")

    joint_pos = np.asarray(data["joint_pos"])
    joint_vel = np.asarray(data["joint_vel"])
    body_pos_w = np.asarray(data["body_pos_w"])
    body_quat_w = np.asarray(data["body_quat_w"])
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"])
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"])
    body_names = np.asarray(data["body_names"])

    if joint_pos.ndim != 2 or joint_pos.shape[1] != NUM_ACTIONS:
        raise ValueError(f"joint_pos must be (T,{NUM_ACTIONS}), got {joint_pos.shape}")
    if joint_vel.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise ValueError(f"joint_vel shape mismatch: {joint_vel.shape} vs {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
        raise ValueError(f"body_pos_w must be (T,nb,3), got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
        raise ValueError(f"body_quat_w must be (T,nb,4), got {body_quat_w.shape}")
    if body_lin_vel_w.ndim != 3 or body_lin_vel_w.shape[2] != 3:
        raise ValueError(f"body_lin_vel_w must be (T,nb,3), got {body_lin_vel_w.shape}")
    if body_ang_vel_w.ndim != 3 or body_ang_vel_w.shape[2] != 3:
        raise ValueError(f"body_ang_vel_w must be (T,nb,3), got {body_ang_vel_w.shape}")

    t = joint_pos.shape[0]
    nb = body_pos_w.shape[1]
    if t <= 0 or nb <= 0:
        raise ValueError("empty time/body dimension")

    for name, arr in [
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("body_pos_w", body_pos_w),
        ("body_quat_w", body_quat_w),
        ("body_lin_vel_w", body_lin_vel_w),
        ("body_ang_vel_w", body_ang_vel_w),
    ]:
        if arr.shape[0] != t:
            raise ValueError(f"{name} first dim mismatch: expected {t}, got {arr.shape[0]}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")

    if body_quat_w.shape[1] != nb or body_lin_vel_w.shape[1] != nb or body_ang_vel_w.shape[1] != nb:
        raise ValueError("body array num_bodies mismatch")
    if body_names.ndim != 1 or body_names.shape[0] != nb:
        raise ValueError(f"body_names must be (nb,), got {body_names.shape}")

    quat_norm = np.linalg.norm(body_quat_w, axis=-1)
    if np.min(quat_norm) < 1e-6:
        raise ValueError("body_quat_w contains near-zero norms")

    return NpzMeta(fps=fps, num_frames=t, num_bodies=nb)


def validate_entries(entries: list[ManifestEntry], npz_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    clip_ids = set()
    file_rels = set()
    enabled_rows = [e for e in entries if e.enabled]
    disabled_rows = [e for e in entries if not e.enabled]

    for e in entries:
        if e.clip_id in clip_ids:
            errors.append(f"line {e.line_no}: duplicate clip_id '{e.clip_id}'")
        clip_ids.add(e.clip_id)
        if e.file_rel in file_rels:
            errors.append(f"line {e.line_no}: duplicate file_rel '{e.file_rel}'")
        file_rels.add(e.file_rel)

    source_stats: dict[str, dict[str, int]] = {}
    num_bodies_expected: int | None = None
    fps_expected: int | None = None

    for e in enabled_rows:
        try:
            path = resolve_npz_path(e.file_rel, npz_root)
            meta = inspect_npz(path)
        except Exception as exc:
            errors.append(f"line {e.line_no} ({e.file_rel}): {exc}")
            continue

        if meta.num_frames != e.num_frames:
            errors.append(
                f"line {e.line_no} ({e.file_rel}): num_frames mismatch "
                f"(manifest={e.num_frames}, actual={meta.num_frames})"
            )
        if meta.fps != e.fps:
            errors.append(
                f"line {e.line_no} ({e.file_rel}): fps mismatch "
                f"(manifest={e.fps}, actual={meta.fps})"
            )

        if num_bodies_expected is None:
            num_bodies_expected = meta.num_bodies
        elif meta.num_bodies != num_bodies_expected:
            errors.append(
                f"line {e.line_no} ({e.file_rel}): num_bodies mismatch "
                f"(expected={num_bodies_expected}, actual={meta.num_bodies})"
            )

        if fps_expected is None:
            fps_expected = meta.fps
        elif meta.fps != fps_expected:
            warnings.append(
                f"line {e.line_no} ({e.file_rel}): fps differs from first clip "
                f"({meta.fps} vs {fps_expected})"
            )

        src = source_stats.setdefault(e.source, {"clips": 0, "frames": 0})
        src["clips"] += 1
        src["frames"] += meta.num_frames

    if not enabled_rows:
        errors.append("no enabled rows in manifest")

    return {
        "ok": len(errors) == 0,
        "timestamp_utc": utc_now_iso(),
        "rows_total": len(entries),
        "rows_enabled": len(enabled_rows),
        "rows_disabled": len(disabled_rows),
        "errors": errors,
        "warnings": warnings,
        "source_stats": source_stats,
        "expected_num_bodies": num_bodies_expected,
        "expected_fps": fps_expected,
    }


def hash_split(clip_id: str, val_percent: int, salt: str = "") -> str:
    payload = f"{salt}:{clip_id}".encode("utf-8")
    bucket = int(hashlib.md5(payload).hexdigest(), 16) % 100
    return "val" if bucket < val_percent else "train"


def merge_npz_files(npz_files: list[Path], output_path: Path) -> dict[str, Any]:
    if not npz_files:
        raise ValueError("no npz files to merge")

    arrays: dict[str, list[np.ndarray]] = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    fps: int | None = None
    body_names: np.ndarray[Any, Any] | None = None

    for p in npz_files:
        d = np.load(p, allow_pickle=True)
        cur_fps = int(d["fps"])
        cur_body_names = np.asarray(d["body_names"])
        if fps is None:
            fps = cur_fps
            body_names = cur_body_names
        else:
            if cur_fps != fps:
                raise ValueError(f"inconsistent fps in merge: {p} has {cur_fps}, expected {fps}")
            if body_names is None or not np.array_equal(cur_body_names, body_names):
                raise ValueError(f"inconsistent body_names in merge: {p}")
        for k in arrays:
            arrays[k].append(np.asarray(d[k]))

    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    merged["fps"] = int(fps)  # type: ignore[arg-type]
    merged["body_names"] = body_names

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **merged)

    total_frames = int(merged["joint_pos"].shape[0])
    return {
        "output": str(output_path),
        "clips": len(npz_files),
        "frames": total_frames,
        "fps": int(merged["fps"]),
        "duration_s": float(total_frames / max(int(merged["fps"]), 1)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

