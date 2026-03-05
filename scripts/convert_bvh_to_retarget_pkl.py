#!/usr/bin/env python3
"""Convert BVH motions to retargeted PKL format.

Output PKL schema is compatible with existing TWIST-style retarget files:
  - fps: int
  - root_pos: (T, 3) float32
  - root_rot: (T, 4) float32, xyzw
  - dof_pos: (T, 29) float32
  - local_body_pos: (T, 38, 3) float32, body positions in root local frame
  - link_body_list: list[str], length 38
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.bvh_provider import BVHInputProvider
from teleopit.retargeting.core import RetargetingModule


def _find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mocap_xml_path(project_root: Path) -> Path:
    return (
        project_root
        / "teleopit"
        / "retargeting"
        / "gmr"
        / "assets"
        / "unitree_g1"
        / "g1_mocap_29dof.xml"
    )


def _resolve_output(input_path: Path, output_path: Path, base_input_dir: Path | None) -> Path:
    if base_input_dir is None:
        if output_path.suffix.lower() == ".pkl":
            return output_path
        return output_path / input_path.with_suffix(".pkl").name

    rel = input_path.relative_to(base_input_dir).with_suffix(".pkl")
    if output_path.suffix.lower() == ".pkl":
        raise ValueError("--output must be a directory when --input is a directory")
    return output_path / rel


def _convert_one(
    bvh_path: Path,
    output_pkl: Path,
    bvh_format: str,
    robot_name: str,
    max_frames: int,
    model: mujoco.MjModel,
) -> None:
    provider = BVHInputProvider(bvh_path=str(bvh_path), human_format=bvh_format)
    retargeter = RetargetingModule(
        robot_name=robot_name,
        human_format=f"bvh_{provider.human_format}",
        actual_human_height=provider.human_height,
    )

    n_total = len(provider)
    n_frames = n_total if max_frames <= 0 else min(n_total, max_frames)
    if n_frames <= 0:
        raise ValueError(f"No frames in BVH: {bvh_path}")

    expected_qpos_dim = model.nq
    num_actions = expected_qpos_dim - 7
    data = mujoco.MjData(model)
    link_body_list = [model.body(i).name for i in range(1, model.nbody)]
    n_bodies = len(link_body_list)

    root_pos = np.zeros((n_frames, 3), dtype=np.float32)
    root_rot_xyzw = np.zeros((n_frames, 4), dtype=np.float32)
    dof_pos = np.zeros((n_frames, num_actions), dtype=np.float32)
    body_pos_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)

    for i in range(n_frames):
        human_frame = provider.get_frame()
        qpos = np.asarray(retargeter.retarget(human_frame), dtype=np.float64).reshape(-1)
        if qpos.shape[0] != expected_qpos_dim:
            raise ValueError(
                f"Retargeted qpos dim mismatch at frame {i}: got {qpos.shape[0]}, expected {expected_qpos_dim}"
            )

        root_pos[i] = qpos[0:3].astype(np.float32)
        root_quat_wxyz = qpos[3:7]
        root_rot_xyzw[i] = np.array(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
            dtype=np.float32,
        )
        dof_pos[i] = qpos[7 : 7 + num_actions].astype(np.float32)

        data.qpos[:] = 0.0
        data.qpos[: 7 + num_actions] = qpos[: 7 + num_actions]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos_w[i] = np.asarray(data.xpos[1 : 1 + n_bodies], dtype=np.float32)

    # World -> root local for each frame/body.
    root_rot_local = R.from_quat(root_rot_xyzw).inv().as_matrix()  # (T, 3, 3)
    delta = body_pos_w - root_pos[:, None, :]  # (T, B, 3)
    local_body_pos = np.einsum("tij,tbj->tbi", root_rot_local, delta)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": int(provider.fps),
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,  # xyzw
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.astype(np.float32),
        "link_body_list": link_body_list,
    }
    with output_pkl.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    duration = n_frames / max(int(provider.fps), 1)
    print(f"[OK] {bvh_path} -> {output_pkl} | {n_frames} frames @ {provider.fps}fps ({duration:.1f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BVH to retargeted PKL format.")
    parser.add_argument("--input", type=str, required=True, help="Input BVH file or directory")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PKL path (if --input is file) or output directory (if --input is dir)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="lafan1",
        choices=["lafan1", "hc_mocap", "nokov"],
        help="BVH format (default: lafan1)",
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="unitree_g1",
        help="Robot name for retargeting (default: unitree_g1)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Process at most N frames per file (0 = all frames)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"--input not found: {input_path}")
    if args.max_frames < 0:
        raise ValueError("--max_frames must be >= 0")

    project_root = _find_project_root()
    mocap_xml = _mocap_xml_path(project_root)
    if not mocap_xml.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {mocap_xml}")
    model = mujoco.MjModel.from_xml_path(str(mocap_xml))

    if input_path.is_file():
        if input_path.suffix.lower() != ".bvh":
            raise ValueError(f"--input must be .bvh file, got: {input_path}")
        output_pkl = _resolve_output(input_path, output_path, base_input_dir=None)
        _convert_one(input_path, output_pkl, args.format, args.robot_name, args.max_frames, model)
        return 0

    if not input_path.is_dir():
        raise ValueError(f"--input must be file or directory: {input_path}")

    bvh_files = sorted(p for p in input_path.rglob("*.bvh"))
    if not bvh_files:
        raise ValueError(f"No .bvh files found in directory: {input_path}")

    print(f"[INFO] Found {len(bvh_files)} BVH files in {input_path}")
    for idx, bvh_file in enumerate(bvh_files, start=1):
        out_pkl = _resolve_output(bvh_file, output_path, base_input_dir=input_path)
        _convert_one(bvh_file, out_pkl, args.format, args.robot_name, args.max_frames, model)
        if idx % 50 == 0 or idx == len(bvh_files):
            print(f"[INFO] Progress: {idx}/{len(bvh_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
