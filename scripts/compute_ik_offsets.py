#!/usr/bin/env python3
"""Compute IK quaternion offsets for hc_mocap → G1 retargeting.

Formula: R_offset = R_human_tpose^{-1} * R_robot_tpose

Where:
- R_human_tpose: bone orientation from BVH frame 0 after Y-up→Z-up rotation
- R_robot_tpose: robot body orientation with correct root yaw and arms horizontal
  (left_shoulder_roll=+1.57, right_shoulder_roll=-1.57)

The robot root must be yaw-rotated to face the same direction as the BVH human.
The human's facing direction is determined from the skeleton geometry (shoulder line
cross up vector). Without this root yaw alignment, all offsets will have a
systematic yaw error.

Prints quaternion offsets (w, x, y, z) for each (robot_body, human_bone) pair
ready to paste into bvh_hc_mocap_to_g1.json.

Usage:
    python scripts/compute_ik_offsets.py
    python scripts/compute_ik_offsets.py --bvh data/hc_mocap/tpose.bvh
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_human_facing_yaw(bvh_path: str) -> float:
    """Determine the human's facing yaw angle from skeleton geometry.

    Uses the shoulder line (left→right) crossed with up (+Z) to find the
    forward direction, then returns the yaw angle from +X axis.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
    from teleopit.retargeting.gmr.utils.lafan_vendor import utils

    data = read_bvh(bvh_path)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    bones = list(data.bones)

    def get_pos(bone_name: str) -> np.ndarray:
        idx = bones.index(bone_name)
        return global_data[1][0, idx] @ rotation_matrix.T

    # Shoulder line: right shoulder - left shoulder
    shoulder_vec = get_pos("hc_Shoulder_R") - get_pos("hc_Shoulder_L")
    shoulder_xy = shoulder_vec[:2]
    shoulder_xy = shoulder_xy / np.linalg.norm(shoulder_xy)

    # Forward = up × right_direction (right-hand rule with Z-up)
    # up × shoulder_xy = (0,0,1) × (sx,sy,0) = (-sy, sx, 0)
    forward_xy = np.array([-shoulder_xy[1], shoulder_xy[0]])
    yaw = math.atan2(forward_xy[1], forward_xy[0])
    return yaw


def get_robot_tpose_orientations(
    xml_path: str, root_yaw: float
) -> dict[str, np.ndarray]:
    """Load G1 model, set T-pose with correct root yaw, return body xquat dict.

    Args:
        xml_path: Path to MuJoCo XML model.
        root_yaw: Yaw angle (radians) to rotate the robot root to match the
            human's facing direction.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Set root quaternion to face the human's direction
    root_quat = R.from_euler("z", root_yaw).as_quat(scalar_first=True)
    data.qpos[3:7] = root_quat

    # Set shoulder roll joints for T-pose (arms horizontal)
    for joint_name, angle in [
        ("left_shoulder_roll_joint", 1.5707963),
        ("right_shoulder_roll_joint", -1.5707963),
    ]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            print(f"WARNING: Joint '{joint_name}' not found in model")
            continue
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = angle

    mujoco.mj_forward(model, data)

    orientations = {}
    for i in range(model.nbody):
        name = model.body(i).name
        if name:
            orientations[name] = data.xquat[i].copy()
    return orientations


def get_human_tpose_orientations(
    bvh_path: str, force_no_toe: bool = False
) -> dict[str, np.ndarray]:
    """Load hc_mocap BVH, get frame 0 bone orientations after Y-up→Z-up rotation.

    Replicates the same transform as bvh_provider.py _load_bvh_file().

    Args:
        bvh_path: Path to BVH file (should be T-pose frame 0).
        force_no_toe: If True, always use hc_Foot_L/R for FootMod orientation
            even when LeftToeBase/RightToeBase exist. Used to generate IK offsets
            for BVH files that lack toe joints.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
    from teleopit.retargeting.gmr.utils.lafan_vendor import utils

    data = read_bvh(bvh_path)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    orientations = {}
    for i, bone in enumerate(data.bones):
        bone_quat = utils.quat_mul(rotation_quat, global_data[0][0, i])
        orientations[bone] = bone_quat

    # Synthesize FootMod entries (same as bvh_provider.py hc_mocap branch)
    if force_no_toe:
        left_toe_key = "hc_Foot_L"
        right_toe_key = "hc_Foot_R"
    else:
        left_toe_key = "LeftToeBase" if "LeftToeBase" in orientations else "hc_Foot_L"
        right_toe_key = "RightToeBase" if "RightToeBase" in orientations else "hc_Foot_R"
    orientations["LeftFootMod"] = orientations[left_toe_key].copy()
    orientations["RightFootMod"] = orientations[right_toe_key].copy()

    return orientations


def compute_offsets(
    robot_oris: dict[str, np.ndarray],
    human_oris: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
) -> dict[str, np.ndarray]:
    """Compute R_offset = R_human_tpose^{-1} * R_robot_tpose for each pair."""
    offsets = {}
    for robot_body, human_bone in pairs:
        if robot_body not in robot_oris:
            print(f"WARNING: Robot body '{robot_body}' not found in model")
            continue
        if human_bone not in human_oris:
            print(f"WARNING: Human bone '{human_bone}' not found in BVH")
            continue

        R_human = R.from_quat(human_oris[human_bone], scalar_first=True)
        R_robot = R.from_quat(robot_oris[robot_body], scalar_first=True)
        R_offset = R_human.inv() * R_robot
        offsets[(robot_body, human_bone)] = R_offset.as_quat(scalar_first=True)

    return offsets


def main():
    parser = argparse.ArgumentParser(description="Compute hc_mocap IK quaternion offsets")
    parser.add_argument(
        "--bvh",
        default="data/hc_mocap/tpose.bvh",
        help="Path to hc_mocap T-pose BVH file",
    )
    parser.add_argument(
        "--config",
        default="teleopit/retargeting/gmr/ik_configs/bvh_hc_mocap_to_g1.json",
        help="Path to current IK config (to read robot-human pairs and weights)",
    )
    parser.add_argument(
        "--no-toe",
        action="store_true",
        help="Force using hc_Foot_L/R for FootMod (for BVH without toe joints)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Auto-update the config file in-place",
    )
    args = parser.parse_args()

    bvh_path = Path(args.bvh)
    if not bvh_path.is_absolute():
        bvh_path = PROJECT_ROOT / bvh_path

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    xml_path = str(
        PROJECT_ROOT
        / "teleopit"
        / "retargeting"
        / "gmr"
        / "assets"
        / "unitree_g1"
        / "g1_mocap_29dof.xml"
    )

    with open(config_path) as f:
        config = json.load(f)

    pairs = []
    for robot_body, entry in config["ik_match_table1"].items():
        pairs.append((robot_body, entry[0]))

    print(f"BVH file: {bvh_path}")
    print(f"Robot XML: {xml_path}")
    print(f"Config: {config_path}")
    print(f"Pairs: {len(pairs)}")
    print()

    # Step 1: Determine human facing direction
    yaw = get_human_facing_yaw(str(bvh_path))
    print(f"Human facing yaw: {math.degrees(yaw):.1f}° from +X")
    print()

    # Step 2: Robot T-pose with matching root yaw
    print("=== Robot T-pose (yaw-aligned, arms horizontal) ===")
    robot_oris = get_robot_tpose_orientations(xml_path, yaw)
    for robot_body, _ in pairs:
        q = robot_oris.get(robot_body)
        if q is not None:
            print(f"  {robot_body:30s}: [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}]")
    print()

    # Step 3: Human T-pose orientations
    print("=== Human T-pose (BVH frame 0, Z-up) ===")
    human_oris = get_human_tpose_orientations(str(bvh_path), force_no_toe=args.no_toe)
    for _, human_bone in pairs:
        q = human_oris.get(human_bone)
        if q is not None:
            print(f"  {human_bone:30s}: [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}]")
    print()

    # Step 4: Compute offsets
    print("=== Computed offsets (R_human_tpose^{-1} * R_robot_tpose) ===")
    offsets = compute_offsets(robot_oris, human_oris, pairs)
    for robot_body, human_bone in pairs:
        key = (robot_body, human_bone)
        if key in offsets:
            q = offsets[key]
            print(f"  {robot_body:30s} <- {human_bone:20s}: [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}]")
    print()

    if args.write:
        print("=== Updating config in-place ===")
        for table_name in ["ik_match_table1", "ik_match_table2"]:
            if table_name not in config:
                continue
            for robot_body, entry in config[table_name].items():
                human_bone = entry[0]
                key = (robot_body, human_bone)
                if key in offsets:
                    q = offsets[key]
                    entry[4] = [round(float(q[i]), 6) for i in range(4)]

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")
        print(f"Written updated config to: {config_path}")
    else:
        print("(Use --write to auto-update the config file)")


if __name__ == "__main__":
    main()
