#!/usr/bin/env python3
"""Convert PKL motion files to NPZ format for mjlab MotionCommand.

Reads retargeted PKL files (IsaacGym convention) and converts them to the NPZ
format expected by mjlab's MotionLoader.

PKL fields:
    fps          : int scalar
    root_pos     : (T, 3) float64
    root_rot     : (T, 4) float64  xyzw quaternion
    dof_pos      : (T, 29) float64
    local_body_pos : (T, 38, 3) float32  body positions in root's LOCAL frame
    link_body_list : list[str] of 38 body names

NPZ fields (30 bodies in mjlab G1 robot body order):
    fps          : int scalar
    joint_pos    : (T, 29) float32
    joint_vel    : (T, 29) float32  (finite-difference)
    body_pos_w   : (T, 30, 3) float32  world-frame body positions
    body_quat_w  : (T, 30, 4) float32  wxyz quaternion
    body_lin_vel_w : (T, 30, 3) float32
    body_ang_vel_w : (T, 30, 3) float32
    body_names   : list[str]  30 body names in mjlab G1 robot order

IMPORTANT: NPZ body ordering must match mjlab G1 robot body ordering because
mjlab's MotionLoader uses robot body indices to index into body_pos_w.

Usage:
    # Convert a single file
    python convert_pkl_to_npz.py --input path/to/file.pkl --output path/to/file.npz

    # Batch convert a directory
    python convert_pkl_to_npz.py --input data/twist2_retarget_pkl/OMOMO_g1_GMR \\
                                  --output data/twist2_retarget_npz/OMOMO_g1_GMR

    # Batch convert + merge into single file (recommended for training)
    python convert_pkl_to_npz.py --input data/twist2_retarget_pkl/OMOMO_g1_GMR \\
                                  --output data/twist2_retarget_npz/OMOMO_g1_GMR \\
                                  --merge
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from train_mimic.data.motion_fk import (
    MotionFkExtractor,
    compute_body_velocities,
    normalize_quaternion,
    quat_xyzw_to_wxyz,
)


# mjlab G1 robot body ordering (matches robot.body_names from G1_ROBOT_CFG).
# mjlab's MotionLoader uses robot body indices to index into NPZ body_pos_w,
# so NPZ body ordering MUST match this list exactly.
_MJLAB_G1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]
def _validate_required_bodies(body_names: list[str], pkl_path: str) -> None:
    missing = sorted(set(_MJLAB_G1_BODY_NAMES).difference(body_names))
    if missing:
        raise ValueError(
            f"PKL body metadata missing required G1 bodies for conversion: {missing}. "
            f"File: {pkl_path}"
        )


def _validate_fk_outputs(body_quat_w: np.ndarray, body_ang_vel_w: np.ndarray, pkl_path: str) -> None:
    quat_spread = np.max(np.linalg.norm(body_quat_w - body_quat_w[:, :1, :], axis=-1))
    if quat_spread < 1e-5:
        raise ValueError(
            "FK-derived body_quat_w collapsed to a rigid-body solution across all tracked bodies. "
            f"This indicates invalid conversion output for {pkl_path}."
        )

    if not np.isfinite(body_ang_vel_w).all():
        raise ValueError(f"FK-derived body_ang_vel_w contains NaN/Inf for {pkl_path}")


def convert_pkl_to_npz(
    pkl_path: str,
    npz_path: str,
    *,
    extractor: MotionFkExtractor | None = None,
) -> None:
    """Convert a single PKL motion file to NPZ format."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    required_keys = {"fps", "root_pos", "root_rot", "dof_pos", "link_body_list"}
    missing = sorted(required_keys.difference(data.keys()))
    if missing:
        raise ValueError(f"Missing PKL keys {missing} in {pkl_path}")

    fps: int = int(data["fps"])
    dt = 1.0 / fps

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)  # (T, 3)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)  # (T, 4) xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)  # (T, 29)
    body_names: list[str] = list(data["link_body_list"])
    _validate_required_bodies(body_names, pkl_path)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (T,3), got {root_pos.shape} in {pkl_path}")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"root_rot must be (T,4), got {root_rot_xyzw.shape} in {pkl_path}")
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be 2D, got {dof_pos.shape} in {pkl_path}")
    if not (root_pos.shape[0] == root_rot_xyzw.shape[0] == dof_pos.shape[0]):
        raise ValueError(
            f"root_pos/root_rot/dof_pos time dimensions mismatch in {pkl_path}: "
            f"{root_pos.shape[0]}/{root_rot_xyzw.shape[0]}/{dof_pos.shape[0]}"
        )

    # Joint velocity via finite difference
    joint_vel = np.gradient(dof_pos, dt, axis=0).astype(np.float32)

    # Convert root quaternion: xyzw -> wxyz
    root_rot_wxyz = normalize_quaternion(quat_xyzw_to_wxyz(root_rot_xyzw))

    fk_extractor = extractor or MotionFkExtractor()
    body_pos_w, body_quat_w = fk_extractor.extract(
        root_pos,
        root_rot_wxyz,
        dof_pos,
        _MJLAB_G1_BODY_NAMES,
    )
    body_lin_vel_w, body_ang_vel_w = compute_body_velocities(body_pos_w, body_quat_w, dt)
    _validate_fk_outputs(body_quat_w, body_ang_vel_w, pkl_path)

    # Save
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(
        npz_path,
        fps=fps,
        joint_pos=dof_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        body_names=np.array(_MJLAB_G1_BODY_NAMES, dtype=str),
    )


def merge_npz_dir(npz_dir: str, output_path: str) -> None:
    """Merge all NPZ files in a directory into a single NPZ by concatenating on time axis.

    mjlab's MotionLoader requires a single NPZ file. Use this after batch-converting
    a dataset directory to concatenate all clips into one training file.

    Args:
        npz_dir: Directory containing .npz files (searched recursively)
        output_path: Output merged .npz file path
    """
    npz_files = sorted(f for f in Path(npz_dir).rglob("*.npz") if f.name != "merged.npz")
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        raise SystemExit(1)

    print(f"Merging {len(npz_files)} NPZ files from {npz_dir} -> {output_path}")

    arrays: dict[str, list[np.ndarray]] = {
        "joint_pos": [], "joint_vel": [],
        "body_pos_w": [], "body_quat_w": [],
        "body_lin_vel_w": [], "body_ang_vel_w": [],
    }
    fps = None
    body_names = None

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        if fps is None:
            fps = int(data["fps"])
            body_names = data["body_names"]
        for key in arrays:
            arrays[key].append(data[key])

    merged = {key: np.concatenate(vals, axis=0) for key, vals in arrays.items()}
    merged["fps"] = fps
    merged["body_names"] = body_names

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **merged)

    T = merged["joint_pos"].shape[0]
    print(f"Done. Merged {len(npz_files)} clips, total {T} frames ({T / fps:.1f} s at {fps} fps).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PKL motion data to NPZ for mjlab.")
    parser.add_argument("--input", type=str, required=True, help="Input PKL file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output NPZ file or directory")
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "After converting, merge all NPZ files in the output directory "
            "into a single merged.npz. Required because mjlab MotionLoader "
            "accepts only a single NPZ file."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single file conversion
        out = output_path if output_path.suffix == ".npz" else output_path / input_path.with_suffix(".npz").name
        print(f"Converting {input_path} -> {out}")
        convert_pkl_to_npz(str(input_path), str(out))
        print("Done.")
    elif input_path.is_dir():
        # Batch directory conversion
        pkl_files = sorted(input_path.rglob("*.pkl"))
        if not pkl_files:
            print(f"No PKL files found in {input_path}")
            return
        print(f"Found {len(pkl_files)} PKL files in {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        extractor = MotionFkExtractor()
        for i, pkl_file in enumerate(pkl_files):
            rel = pkl_file.relative_to(input_path)
            npz_file = output_path / rel.with_suffix(".npz")
            npz_file.parent.mkdir(parents=True, exist_ok=True)
            convert_pkl_to_npz(str(pkl_file), str(npz_file), extractor=extractor)
            if (i + 1) % 100 == 0 or (i + 1) == len(pkl_files):
                print(f"  [{i + 1}/{len(pkl_files)}] {rel}")
        print(f"Done. Converted {len(pkl_files)} files to {output_path}")

        if args.merge:
            merged_out = str(output_path / "merged.npz")
            merge_npz_dir(str(output_path), merged_out)
    else:
        print(f"Error: {input_path} not found")
        raise SystemExit(1)

    # Merge-only mode: input is already an NPZ directory
    if args.merge and input_path.is_dir() and not list(input_path.rglob("*.pkl")):
        merged_out = str(output_path) if output_path.suffix == ".npz" else str(output_path / "merged.npz")
        merge_npz_dir(str(input_path), merged_out)


if __name__ == "__main__":
    main()
