#!/usr/bin/env python3
"""Validate NPZ motion labels against MuJoCo FK reconstructed from the same qpos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_mimic.data.motion_fk import compute_npz_fk_consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NPZ body pose labels against MuJoCo FK.")
    parser.add_argument("--npz", type=str, required=True, help="Path to a single motion NPZ clip")
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Optional MuJoCo XML path (default: train_mimic/assets/g1/g1_sim2sim_29dof.xml)",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=32,
        help="Number of frames to sample for FK consistency check; <=0 means all frames",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for frame sampling")
    parser.add_argument("--pos_max", type=float, default=1e-3, help="Max allowed body position error in meters")
    parser.add_argument("--quat_mean", type=float, default=0.05, help="Max allowed mean body orientation error in radians")
    parser.add_argument("--quat_p95", type=float, default=0.10, help="Max allowed p95 body orientation error in radians")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON stats in addition to the text summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = compute_npz_fk_consistency(
        args.npz,
        model_path=args.xml,
        sample_count=args.sample_count,
        seed=args.seed,
    )

    print(f"NPZ: {Path(args.npz).expanduser().resolve()}")
    print(f"frames_checked: {stats.frames_checked}")
    print(f"bodies_checked: {stats.bodies_checked}")
    print(f"pos_mean:  {stats.pos_mean:.6e} m")
    print(f"pos_p95:   {stats.pos_p95:.6e} m")
    print(f"pos_max:   {stats.pos_max:.6e} m")
    print(f"quat_mean: {stats.quat_mean:.6e} rad")
    print(f"quat_p95:  {stats.quat_p95:.6e} rad")
    print(f"quat_max:  {stats.quat_max:.6e} rad")

    if args.json:
        print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))

    if stats.pos_max > args.pos_max:
        print(
            f"[FAIL] pos_max={stats.pos_max:.6e} exceeds threshold {args.pos_max:.6e}",
        )
        return 1
    if stats.quat_mean > args.quat_mean:
        print(
            f"[FAIL] quat_mean={stats.quat_mean:.6e} exceeds threshold {args.quat_mean:.6e}",
        )
        return 1
    if stats.quat_p95 > args.quat_p95:
        print(
            f"[FAIL] quat_p95={stats.quat_p95:.6e} exceeds threshold {args.quat_p95:.6e}",
        )
        return 1

    print("[PASS] FK consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
