"""Pre-compute per-clip quality metrics for anomaly detection during review."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from train_mimic.data.review_lib import ReviewRow


@dataclass(frozen=True)
class ClipMetrics:
    clip_id: str
    first_frame_jump: float  # max |joint_pos[1] - joint_pos[0]| (radians)
    root_height_min: float  # min pelvis z
    root_height_max: float  # max pelvis z
    root_speed_max: float  # max pelvis XY velocity (m/s)
    body_lin_vel_max: float  # max across all bodies, all frames (m/s)
    body_ang_vel_max: float  # max across all bodies, all frames (rad/s)
    duration_s: float


def compute_clip_metrics(npz_path: str | Path, clip_id: str = "") -> ClipMetrics:
    """Compute anomaly metrics for a single clip NPZ file."""
    data = np.load(str(npz_path), allow_pickle=True)
    fps = int(data["fps"])
    joint_pos = np.asarray(data["joint_pos"])  # (T, 29)
    body_pos_w = np.asarray(data["body_pos_w"])  # (T, nb, 3)
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"])  # (T, nb, 3)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"])  # (T, nb, 3)

    num_frames = joint_pos.shape[0]
    duration_s = num_frames / fps if fps > 0 else 0.0

    # First frame jump
    if num_frames > 1:
        first_frame_jump = float(np.max(np.abs(joint_pos[1] - joint_pos[0])))
    else:
        first_frame_jump = 0.0

    # Root (pelvis = body index 0) height
    pelvis_z = body_pos_w[:, 0, 2]
    root_height_min = float(np.min(pelvis_z))
    root_height_max = float(np.max(pelvis_z))

    # Root XY speed
    pelvis_xy_vel = body_lin_vel_w[:, 0, :2]  # (T, 2)
    root_speed_max = float(np.max(np.linalg.norm(pelvis_xy_vel, axis=-1)))

    # Body velocities
    body_lin_speeds = np.linalg.norm(body_lin_vel_w, axis=-1)  # (T, nb)
    body_ang_speeds = np.linalg.norm(body_ang_vel_w, axis=-1)  # (T, nb)
    body_lin_vel_max = float(np.max(body_lin_speeds))
    body_ang_vel_max = float(np.max(body_ang_speeds))

    return ClipMetrics(
        clip_id=clip_id,
        first_frame_jump=first_frame_jump,
        root_height_min=root_height_min,
        root_height_max=root_height_max,
        root_speed_max=root_speed_max,
        body_lin_vel_max=body_lin_vel_max,
        body_ang_vel_max=body_ang_vel_max,
        duration_s=duration_s,
    )


def _compute_one(args: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    """Worker function for multiprocessing."""
    npz_path, clip_id = args
    try:
        m = compute_clip_metrics(npz_path, clip_id)
        return clip_id, asdict(m)
    except Exception as exc:
        return clip_id, {"error": str(exc)}


def compute_all_metrics(
    rows: list[ReviewRow],
    project_root: Path,
    *,
    cache_path: Path | None = None,
    num_workers: int = 4,
) -> dict[str, ClipMetrics]:
    """Compute metrics for all clips, with optional JSON cache.

    If cache_path exists and covers all clip_ids, loads from cache.
    Otherwise computes (possibly in parallel) and saves to cache.
    """
    clip_ids = {r.clip_id for r in rows}

    # Try loading from cache
    if cache_path is not None and cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_ids = set(cached.keys())
        if clip_ids.issubset(cached_ids):
            result = {}
            for r in rows:
                d = cached[r.clip_id]
                if "error" not in d:
                    result[r.clip_id] = ClipMetrics(**d)
            print(f"[METRICS] Loaded {len(result)} metrics from cache: {cache_path}")
            return result

    # Build work items
    work: list[tuple[str, str]] = []
    for r in rows:
        p = Path(r.file_rel)
        if not p.is_absolute():
            p = project_root / p
        work.append((str(p), r.clip_id))

    print(f"[METRICS] Computing metrics for {len(work)} clips ({num_workers} workers)...")
    raw_results: dict[str, dict[str, Any]] = {}

    if num_workers <= 1:
        for i, (npz_path, clip_id) in enumerate(work):
            cid, data = _compute_one((npz_path, clip_id))
            raw_results[cid] = data
            if (i + 1) % 1000 == 0:
                print(f"  [{i + 1}/{len(work)}]")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i, (cid, data) in enumerate(pool.map(_compute_one, work, chunksize=64)):
                raw_results[cid] = data
                if (i + 1) % 1000 == 0:
                    print(f"  [{i + 1}/{len(work)}]")

    # Save cache
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=True)
        print(f"[METRICS] Saved cache: {cache_path}")

    # Convert to ClipMetrics
    result: dict[str, ClipMetrics] = {}
    errors = 0
    for cid, data in raw_results.items():
        if "error" in data:
            errors += 1
            continue
        result[cid] = ClipMetrics(**data)

    if errors:
        print(f"[METRICS] WARNING: {errors} clips had errors during metric computation")
    print(f"[METRICS] Done: {len(result)} clips processed")
    return result


def suspicion_score(m: ClipMetrics) -> float:
    """Heuristic anomaly score. Higher = more suspicious."""
    score = 0.0
    if m.first_frame_jump > 0.5:
        score += 3.0
    elif m.first_frame_jump > 0.2:
        score += 1.0
    if m.root_height_max > 1.5:
        score += 2.0
    if m.root_height_min < 0.2:
        score += 2.0
    if m.body_lin_vel_max > 20.0:
        score += 3.0
    elif m.body_lin_vel_max > 10.0:
        score += 1.0
    if m.body_ang_vel_max > 30.0:
        score += 2.0
    if m.duration_s < 0.5:
        score += 1.0
    return score
