#!/usr/bin/env python3
"""Shared utilities for the active NPZ-based dataset pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
class NpzMeta:
    fps: int
    num_frames: int
    num_bodies: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def hash_split(clip_id: str, val_percent: int, salt: str = "") -> str:
    payload = f"{salt}:{clip_id}".encode("utf-8")
    bucket = int(hashlib.md5(payload).hexdigest(), 16) % 100
    return "val" if bucket < val_percent else "train"


def _resample_along_time(arr: np.ndarray, new_t: int) -> np.ndarray:
    """Resample an array along time axis 0 using linear interpolation."""
    old_t = int(arr.shape[0])
    if old_t == new_t:
        return arr
    if old_t <= 0:
        raise ValueError("cannot resample empty array")
    if new_t <= 0:
        raise ValueError("resampled length must be > 0")
    if old_t == 1:
        return np.repeat(arr, new_t, axis=0)

    pos = np.linspace(0.0, float(old_t - 1), new_t, dtype=np.float64)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, old_t - 1)
    w = (pos - i0).astype(np.float32).reshape((new_t,) + (1,) * (arr.ndim - 1))
    out = arr[i0] * (1.0 - w) + arr[i1] * w
    return out.astype(arr.dtype, copy=False)


def merge_npz_files(
    npz_files: list[Path],
    output_path: Path,
    *,
    target_fps: int | None = None,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    if not npz_files:
        raise ValueError("no npz files to merge")
    if weights is not None and len(weights) != len(npz_files):
        raise ValueError(
            f"weights length ({len(weights)}) != npz_files length ({len(npz_files)})"
        )

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
    per_clip_fps: list[int] = []
    per_clip_weights: list[float] = []

    for i, p in enumerate(npz_files):
        d = np.load(p, allow_pickle=True)
        cur_fps = int(d["fps"])
        if cur_fps <= 0:
            raise ValueError(f"invalid fps in {p}: {cur_fps}")
        cur_body_names = np.asarray(d["body_names"])
        if target_fps is not None and target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")

        joint_pos = np.asarray(d["joint_pos"])
        joint_vel = np.asarray(d["joint_vel"])
        body_pos_w = np.asarray(d["body_pos_w"])
        body_quat_w = np.asarray(d["body_quat_w"])
        body_lin_vel_w = np.asarray(d["body_lin_vel_w"])
        body_ang_vel_w = np.asarray(d["body_ang_vel_w"])

        if target_fps is not None and cur_fps != target_fps:
            old_t = int(joint_pos.shape[0])
            new_t = int(round(old_t * float(target_fps) / float(cur_fps)))
            new_t = max(new_t, 1)

            joint_pos = _resample_along_time(joint_pos, new_t)
            joint_vel = _resample_along_time(joint_vel, new_t)
            body_pos_w = _resample_along_time(body_pos_w, new_t)
            body_quat_w = _resample_along_time(body_quat_w, new_t)
            body_lin_vel_w = _resample_along_time(body_lin_vel_w, new_t)
            body_ang_vel_w = _resample_along_time(body_ang_vel_w, new_t)

            quat_norm = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
            quat_norm = np.where(quat_norm < 1e-8, 1.0, quat_norm)
            body_quat_w = body_quat_w / quat_norm
            cur_fps = target_fps

        if fps is None:
            fps = cur_fps
            body_names = cur_body_names
        else:
            if cur_fps != fps:
                raise ValueError(f"inconsistent fps in merge: {p} has {cur_fps}, expected {fps}")
            if body_names is None or not np.array_equal(cur_body_names, body_names):
                raise ValueError(f"inconsistent body_names in merge: {p}")

        per_clip_fps.append(cur_fps)
        per_clip_weights.append(weights[i] if weights is not None else 1.0)

        arrays["joint_pos"].append(joint_pos)
        arrays["joint_vel"].append(joint_vel)
        arrays["body_pos_w"].append(body_pos_w)
        arrays["body_quat_w"].append(body_quat_w)
        arrays["body_lin_vel_w"].append(body_lin_vel_w)
        arrays["body_ang_vel_w"].append(body_ang_vel_w)

    # Clip boundary metadata
    clip_lengths = np.array(
        [arr.shape[0] for arr in arrays["joint_pos"]], dtype=np.int64
    )
    clip_starts = np.zeros(len(clip_lengths), dtype=np.int64)
    if len(clip_lengths) > 1:
        clip_starts[1:] = np.cumsum(clip_lengths[:-1])

    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    merged["fps"] = int(fps)  # type: ignore[arg-type]
    merged["body_names"] = body_names
    merged["clip_starts"] = clip_starts
    merged["clip_lengths"] = clip_lengths
    merged["clip_fps"] = np.array(per_clip_fps, dtype=np.int64)
    merged["clip_weights"] = np.array(per_clip_weights, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **merged)

    total_frames = int(merged["joint_pos"].shape[0])
    return {
        "output": str(output_path),
        "clips": len(npz_files),
        "num_clips": len(npz_files),
        "frames": total_frames,
        "fps": int(merged["fps"]),
        "duration_s": float(total_frames / max(int(merged["fps"]), 1)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
