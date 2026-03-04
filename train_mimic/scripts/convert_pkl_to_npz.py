#!/usr/bin/env python3
"""Convert PKL motion files to NPZ format for mjlab MotionCommand.

Reads retargeted PKL files (IsaacGym convention) and converts them to the NPZ
format expected by mjlab's MotionLoader.

PKL fields:
    fps          : int scalar
    root_pos     : (T, 3) float64
    root_rot     : (T, 4) float64  xyzw quaternion
    dof_pos      : (T, 29) float64
    local_body_pos : (T, 38, 3) float32  body positions in root frame
    link_body_list : list[str] of 38 body names

NPZ fields:
    fps          : int scalar
    joint_pos    : (T, 29) float32
    joint_vel    : (T, 29) float32  (finite-difference)
    body_pos_w   : (T, nb, 3) float32  world-frame body positions
    body_quat_w  : (T, nb, 4) float32  wxyz quaternion
    body_lin_vel_w : (T, nb, 3) float32
    body_ang_vel_w : (T, nb, 3) float32
    body_names   : list[str]  names of the nb bodies

Usage:
    # Convert a single file
    python convert_pkl_to_npz.py --input path/to/file.pkl --output path/to/file.npz

    # Batch convert a directory
    python convert_pkl_to_npz.py --input data/twist2_retarget_pkl/OMOMO_g1_GMR \
                                  --output data/twist2_retarget_npz/OMOMO_g1_GMR
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from xyzw (IsaacGym) to wxyz (MuJoCo) convention."""
    return q[..., [3, 0, 1, 2]]


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions: q1 * q2.

    Args:
        q1: (..., 4) wxyz
        q2: (..., 4) wxyz
    Returns:
        (..., 4) wxyz
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of wxyz quaternion."""
    conj = q.copy()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def quat_to_angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from quaternion sequence.

    Uses finite differences: omega = 2 * q_dot * q_conj (imaginary part).

    Args:
        q: (T, ..., 4) wxyz quaternion sequence
        dt: time step

    Returns:
        (T, ..., 3) angular velocity in world frame
    """
    q_dot = np.gradient(q, dt, axis=0)
    # omega = 2 * q_dot * conj(q), take imaginary (xyz) part
    product = quat_multiply(q_dot, quat_conjugate(q))
    return 2.0 * product[..., 1:4]


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz convention).

    Args:
        q: (..., 4) wxyz quaternion
        v: (..., 3) vector

    Returns:
        (..., 3) rotated vector
    """
    # q * [0, v] * q_conj
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=v.dtype)
    v_quat[..., 1:4] = v
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[..., 1:4]


def convert_pkl_to_npz(pkl_path: str, npz_path: str) -> None:
    """Convert a single PKL motion file to NPZ format."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fps: int = int(data["fps"])
    dt = 1.0 / fps

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)  # (T, 3)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)  # (T, 4) xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)  # (T, 29)
    local_body_pos = np.asarray(data["local_body_pos"], dtype=np.float32)  # (T, 38, 3)
    body_names: list[str] = list(data["link_body_list"])

    T = root_pos.shape[0]
    nb = local_body_pos.shape[1]

    # Joint velocity via finite difference
    joint_vel = np.gradient(dof_pos, dt, axis=0).astype(np.float32)

    # Convert root quaternion: xyzw -> wxyz
    root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw)  # (T, 4)

    # Body positions in world frame: local_body_pos is relative to root
    body_pos_w = local_body_pos + root_pos[:, None, :]  # (T, nb, 3)

    # Body quaternions in world frame
    # For simplicity, assign root orientation to all bodies since PKL
    # doesn't store per-body orientations. The MotionCommand will use
    # body positions as primary tracking targets.
    body_quat_w = np.tile(root_rot_wxyz[:, None, :], (1, nb, 1))  # (T, nb, 4)

    # Body linear velocities via finite difference
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)

    # Body angular velocities via quaternion differentiation
    body_ang_vel_w = quat_to_angular_velocity(body_quat_w, dt).astype(np.float32)

    # Normalize quaternions
    norms = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    body_quat_w = body_quat_w / norms

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
        body_names=np.array(body_names, dtype=str),
    )


def merge_npz_dir(npz_dir: str, output_path: str) -> None:
    """Merge all NPZ files in a directory into a single NPZ by concatenating on time axis.

    mjlab's MotionLoader requires a single NPZ file. Use this after batch-converting
    a dataset directory to concatenate all clips into one training file.

    Args:
        npz_dir: Directory containing .npz files (searched recursively)
        output_path: Output merged .npz file path
    """
    npz_files = sorted(Path(npz_dir).rglob("*.npz"))
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
        for i, pkl_file in enumerate(pkl_files):
            rel = pkl_file.relative_to(input_path)
            npz_file = output_path / rel.with_suffix(".npz")
            npz_file.parent.mkdir(parents=True, exist_ok=True)
            convert_pkl_to_npz(str(pkl_file), str(npz_file))
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
