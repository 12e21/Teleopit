#!/usr/bin/env python3
"""Render sim2sim videos directly from TWIST2 .pkl motion files.

Usage:
    MUJOCO_GL=egl python scripts/render_pkl_sim2sim.py --pkl data/twist2/OMOMO_g1_GMR/sub1_clothesstand_000.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio  # noqa: E402
import mujoco  # noqa: E402


def _find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _make_camera() -> Any:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.8]
    cam.distance = 3.0
    cam.azimuth = 135.0
    cam.elevation = -20.0
    return cam


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    size_mb = path.stat().st_size / (1024 * 1024)
    duration = len(frames) / fps
    print(f"  Saved: {path} ({size_mb:.1f} MB, {len(frames)} frames, {duration:.1f}s @ {fps}fps)")


def _load_pkl(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    with open(pkl_path, "rb") as f:
        motion = pickle.load(f)

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_raw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    motion_fps = int(round(float(motion["fps"])))
    n_frames = root_pos.shape[0]

    # PKL quaternions are xyzw (IsaacGym), convert to wxyz (MuJoCo)
    root_rot = np.zeros_like(root_rot_raw)
    root_rot[:, 0] = root_rot_raw[:, 3]
    root_rot[:, 1] = root_rot_raw[:, 0]
    root_rot[:, 2] = root_rot_raw[:, 1]
    root_rot[:, 3] = root_rot_raw[:, 2]

    return root_pos, root_rot, dof_pos, motion_fps, n_frames


def render_pkl_retarget(
    pkl_path: Path,
    output_path: Path,
    width: int = 640,
    height: int = 360,
    max_seconds: float = 0,
) -> None:
    project_root = _find_project_root()
    root_pos, root_rot, dof_pos, motion_fps, n_frames = _load_pkl(pkl_path)

    motion_duration = n_frames / motion_fps
    if max_seconds > 0:
        motion_duration = min(motion_duration, max_seconds)
    num_frames = min(int(motion_duration * motion_fps), n_frames)

    print(f"  [retarget] {num_frames} frames @ {motion_fps}fps = {motion_duration:.1f}s")

    from omegaconf import OmegaConf
    from teleopit.robots.mujoco_robot import MuJoCoRobot

    robot_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "robot" / "g1.yaml")
    mocap_xml = project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    robot_cfg.xml_path = str(mocap_xml)

    robot = MuJoCoRobot(robot_cfg)
    model = robot.model
    data = robot.data
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = _make_camera()

    frames: list[np.ndarray] = []
    t0 = time.time()

    for i in range(num_frames):
        qpos = np.concatenate([root_pos[i], root_rot[i], dof_pos[i]])
        data.qpos[: len(qpos)] = qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        frames.append(frame.copy())

        if (i + 1) % 100 == 0:
            print(f"    Frame {i + 1}/{num_frames} ({time.time() - t0:.1f}s)")

    renderer.close()

    if not frames:
        print("  WARNING: No frames rendered!")
        return

    print(f"  [retarget] Done: {len(frames)} frames in {time.time() - t0:.1f}s")
    _write_video(frames, output_path, motion_fps)


def render_pkl_sim2sim(
    pkl_path: Path,
    output_path: Path,
    width: int = 640,
    height: int = 360,
    max_seconds: float = 0,
) -> None:
    project_root = _find_project_root()
    root_pos, root_rot, dof_pos, motion_fps, n_frames = _load_pkl(pkl_path)

    motion_duration = n_frames / motion_fps
    if max_seconds > 0:
        motion_duration = min(motion_duration, max_seconds)

    print(f"  PKL: {n_frames} frames @ {motion_fps}fps = {motion_duration:.1f}s, {dof_pos.shape[1]} DOF")

    from omegaconf import OmegaConf
    from teleopit.controllers.rl_policy import RLPolicyController
    from teleopit.controllers.observation import TWIST2ObservationBuilder
    from teleopit.retargeting.core import extract_mimic_obs
    from teleopit.robots.mujoco_robot import MuJoCoRobot

    robot_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "robot" / "g1.yaml")
    controller_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "controller" / "rl_policy.yaml")
    default_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "default.yaml")

    sim2sim_xml = project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_sim2sim_29dof.xml"
    robot_cfg.xml_path = str(sim2sim_xml)

    policy_path = project_root.parent / "TWIST2" / "assets" / "ckpts" / "twist2_1017_20k.onnx"
    if not policy_path.exists():
        print(f"ERROR: ONNX policy not found at {policy_path}")
        sys.exit(1)
    controller_cfg.policy_path = str(policy_path)
    controller_cfg.default_dof_pos = list(robot_cfg.default_angles)

    POLICY_HZ = int(OmegaConf.select(default_cfg, "policy_hz", default=50))
    PD_HZ = int(OmegaConf.select(default_cfg, "pd_hz", default=1000))
    DECIMATION = PD_HZ // POLICY_HZ

    robot = MuJoCoRobot(robot_cfg)
    controller = RLPolicyController(controller_cfg)

    obs_cfg = {
        "num_actions": int(robot_cfg.num_actions),
        "ang_vel_scale": float(OmegaConf.select(robot_cfg, "ang_vel_scale", default=0.25)),
        "dof_pos_scale": float(OmegaConf.select(robot_cfg, "dof_pos_scale", default=1.0)),
        "dof_vel_scale": float(OmegaConf.select(robot_cfg, "dof_vel_scale", default=0.05)),
        "ankle_idx": list(OmegaConf.select(robot_cfg, "ankle_idx", default=[4, 5, 10, 11])),
        "default_dof_pos": list(robot_cfg.default_angles),
    }
    obs_builder = TWIST2ObservationBuilder(obs_cfg)

    num_actions = int(robot_cfg.num_actions)
    kps = np.asarray(list(robot_cfg.kps), dtype=np.float32)
    kds = np.asarray(list(robot_cfg.kds), dtype=np.float32)
    torque_limits = np.asarray(list(robot_cfg.torque_limits), dtype=np.float32)

    model = robot.model
    data = robot.data
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = _make_camera()

    num_policy_steps = int(motion_duration * POLICY_HZ)
    print(f"  Running {num_policy_steps} policy steps @ {POLICY_HZ}Hz, PD @ {PD_HZ}Hz (decimation={DECIMATION})")

    video_frames: list[np.ndarray] = []
    t0 = time.time()
    last_action = np.zeros(num_actions, dtype=np.float32)
    last_retarget_qpos = None
    last_motion_idx = -1

    for step in range(num_policy_steps):
        policy_time = step / POLICY_HZ
        motion_idx = min(int(policy_time * motion_fps), n_frames - 1)

        if motion_idx != last_motion_idx:
            qpos = np.concatenate([
                root_pos[motion_idx],
                root_rot[motion_idx],
                dof_pos[motion_idx],
            ])
            last_motion_idx = motion_idx

        mimic_obs = extract_mimic_obs(
            qpos=qpos, last_qpos=last_retarget_qpos, dt=1.0 / POLICY_HZ
        )

        state = robot.get_state()
        obs = obs_builder.build(state, mimic_obs, last_action)

        expected_dim = getattr(controller, "_expected_obs_dim", None)
        if isinstance(expected_dim, int) and expected_dim > 0 and obs.shape[0] != expected_dim:
            if obs.shape[0] > expected_dim:
                obs = obs[:expected_dim]
            else:
                obs = np.pad(obs, (0, expected_dim - obs.shape[0]))

        action = np.asarray(controller.compute_action(obs), dtype=np.float32).reshape(-1)
        target_dof_pos = controller.get_target_dof_pos(action)

        for _ in range(DECIMATION):
            pd_state = robot.get_state()
            cur_pos = np.asarray(pd_state.qpos, dtype=np.float32)[:num_actions]
            cur_vel = np.asarray(pd_state.qvel, dtype=np.float32)[:num_actions]
            torque = (target_dof_pos - cur_pos) * kps - cur_vel * kds
            torque = np.clip(torque, -torque_limits, torque_limits).astype(np.float32)
            robot.set_action(torque)
            robot.step()

        last_action = action
        last_retarget_qpos = qpos.copy()

        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        video_frames.append(frame.copy())

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}/{num_policy_steps} ({time.time() - t0:.1f}s)")

    renderer.close()

    if not video_frames:
        print("  WARNING: No frames rendered!")
        return

    print(f"  Done: {len(video_frames)} frames in {time.time() - t0:.1f}s")
    _write_video(video_frames, output_path, POLICY_HZ)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render retarget + sim2sim from TWIST2 .pkl motion files")
    parser.add_argument("--pkl", required=True, help="Path to .pkl motion file")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: outputs/pkl_sim2sim/{stem}/)")
    parser.add_argument("--max_seconds", type=float, default=0, help="Max video duration (0=full)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    project_root = _find_project_root()
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = (project_root / pkl_path).resolve()

    if not pkl_path.exists():
        print(f"PKL file not found: {pkl_path}")
        sys.exit(1)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "outputs" / "pkl_sim2sim" / pkl_path.stem

    print(f"Processing: {pkl_path.name}")

    print("\n--- Retarget ---")
    render_pkl_retarget(pkl_path, out_dir / "retarget.mp4", args.width, args.height, args.max_seconds)

    print("\n--- Sim2Sim ---")
    render_pkl_sim2sim(pkl_path, out_dir / "sim2sim.mp4", args.width, args.height, args.max_seconds)

    print(f"\nDone! Videos in: {out_dir}")


if __name__ == "__main__":
    main()
