#!/usr/bin/env python3
"""Render MuJoCo verification videos from BVH input.

Produces THREE videos per BVH file, all at the BVH native frame rate
with ALL frames rendered, so they have identical duration:
  1. *_bvh.mp4      — Raw BVH skeleton (matplotlib 3D)
  2. *_retarget.mp4  — GMR kinematic retargeting (qpos set directly, no physics)
  3. *_sim2sim.mp4   — Full RL policy pipeline (BVH → GMR → obs → ONNX → PD → MuJoCo)

Usage:
    MUJOCO_GL=egl python scripts/render_sim.py --bvh data/lafan1/dance1_subject2.bvh
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio  # noqa: E402
import mujoco  # noqa: E402


def _find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_bvh_fps(bvh_path: Path) -> int:
    with open(bvh_path, "r") as f:
        for line in f:
            m = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
            if m:
                return round(1.0 / float(m.group(1)))
    return 30


def _make_camera() -> mujoco.MjvCamera:
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
    print(
        f"  Saved: {path} ({size_mb:.1f} MB, {len(frames)} frames, {duration:.1f}s @ {fps}fps)"
    )


def _load_configs(
    bvh_path: str, project_root: Path, bvh_format: str = "lafan1"
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    default_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "default.yaml")
    robot_cfg = OmegaConf.load(
        project_root / "teleopit" / "configs" / "robot" / "g1.yaml"
    )
    controller_cfg = OmegaConf.load(
        project_root / "teleopit" / "configs" / "controller" / "rl_policy.yaml"
    )
    input_cfg = OmegaConf.load(
        project_root / "teleopit" / "configs" / "input" / "bvh.yaml"
    )

    xml_path = (
        project_root
        / "teleopit"
        / "retargeting"
        / "gmr"
        / "assets"
        / "unitree_g1"
        / "g1_mocap_29dof.xml"
    )
    robot_cfg.xml_path = str(xml_path)

    policy_path = (
        project_root.parent / "TWIST2" / "assets" / "ckpts" / "twist2_1017_20k.onnx"
    )
    if not policy_path.exists():
        print(f"ERROR: ONNX policy not found at {policy_path}")
        sys.exit(1)
    controller_cfg.policy_path = str(policy_path)
    controller_cfg.default_dof_pos = list(robot_cfg.default_angles)

    input_cfg.bvh_file = str(bvh_path)
    input_cfg.provider = "bvh"
    input_cfg.bvh_format = bvh_format
    input_cfg.human_format = f"bvh_{bvh_format}"
    input_cfg.robot_name = "unitree_g1"

    return {
        "robot": robot_cfg,
        "controller": controller_cfg,
        "input": input_cfg,
        "policy_hz": float(OmegaConf.select(default_cfg, "policy_hz", default=50.0)),
        "pd_hz": float(OmegaConf.select(default_cfg, "pd_hz", default=1000.0)),
    }


def render_bvh(
    bvh_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    max_frames: int = 0,
    bvh_format: str = "lafan1",
) -> None:
    """Render raw BVH skeleton as 3D matplotlib animation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
    from teleopit.retargeting.gmr.utils.lafan_vendor import utils

    data = read_bvh(str(bvh_path))
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    global_quats, global_pos = utils.quat_fk(data.quats, data.pos, data.parents)
    scale_divisor = 1.0 if bvh_format == "hc_mocap" else 100.0
    positions = np.einsum("fbi,ji->fbj", global_pos, rotation_matrix) / scale_divisor
    # hc_mocap 60fps is downsampled to 30fps by BVHInputProvider — match here
    if bvh_format == "hc_mocap" and positions.shape[0] > 0:
        raw_fps = round(1.0 / data.frametime) if data.frametime else 30
        if raw_fps == 60:
            positions = positions[::2]
            global_quats = global_quats[::2]
    parents = data.parents

    num_frames = positions.shape[0]
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)
    duration = num_frames / fps
    print(f"  [bvh] Rendering {num_frames} frames @ {fps}fps -> {duration:.1f}s video")

    dpi = 100
    fig_w, fig_h = width / dpi, height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    frames: list[np.ndarray] = []
    t0 = time.time()

    for step in range(num_frames):
        ax.cla()
        pos = positions[step]
        root = pos[0]

        ax.set_xlim(root[0] - 1.0, root[0] + 1.0)
        ax.set_ylim(root[1] - 1.0, root[1] + 1.0)
        ax.set_zlim(0.0, 2.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"BVH Skeleton — frame {step}")
        ax.view_init(elev=20, azim=135)

        for j in range(len(parents)):
            p = parents[j]
            if p < 0:
                continue
            xs = [pos[p, 0], pos[j, 0]]
            ys = [pos[p, 1], pos[j, 1]]
            zs = [pos[p, 2], pos[j, 2]]
            ax.plot(xs, ys, zs, "b-", linewidth=2)

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="red", s=15, depthshade=True)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        frames.append(img)

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}/{num_frames} ({time.time() - t0:.1f}s)")

    plt.close(fig)

    if not frames:
        print("  WARNING: No frames rendered!")
        return

    print(f"  [bvh] Done: {len(frames)} frames in {time.time() - t0:.1f}s")
    _write_video(frames, output_path, fps)


def render_retarget(
    bvh_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    max_frames: int = 0,
    bvh_format: str = "lafan1",
) -> None:
    """Render GMR retargeting result — set qpos directly, no physics."""
    project_root = _find_project_root()
    cfgs = _load_configs(str(bvh_path), project_root, bvh_format)

    from teleopit.inputs import BVHInputProvider
    from teleopit.retargeting.core import RetargetingModule
    from teleopit.robots.mujoco_robot import MuJoCoRobot

    robot = MuJoCoRobot(cfgs["robot"])
    input_prov = BVHInputProvider(bvh_path=str(bvh_path), human_format=bvh_format)
    retargeter = RetargetingModule(
        robot_name="unitree_g1",
        human_format=f"bvh_{bvh_format}",
        actual_human_height=input_prov.human_height,
    )

    model = robot.model
    data = robot.data
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = _make_camera()

    num_steps = len(input_prov)
    if max_frames > 0:
        num_steps = min(num_steps, max_frames)
    duration = num_steps / fps
    print(
        f"  [retarget] Rendering {num_steps} frames @ {fps}fps -> {duration:.1f}s video"
    )

    frames: list[np.ndarray] = []
    t0 = time.time()

    for step in range(num_steps):
        try:
            human_frame = input_prov.get_frame()
        except StopIteration:
            break

        retargeted = retargeter.retarget(human_frame)
        qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)

        # Set full qpos (7D root + 29D joints) directly — no physics
        data.qpos[: len(qpos)] = qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        # Post-process: lift root Z so feet don't sink below ground
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        lowest_foot_z = min(data.xpos[left_foot_id][2], data.xpos[right_foot_id][2])
        if lowest_foot_z < 0.0:
            data.qpos[2] -= lowest_foot_z  # lift root by the penetration depth
            mujoco.mj_forward(model, data)

        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]

        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        frames.append(frame.copy())

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}/{num_steps} ({time.time() - t0:.1f}s)")

    renderer.close()

    if not frames:
        print("  WARNING: No frames rendered!")
        return

    print(f"  [retarget] Done: {len(frames)} frames in {time.time() - t0:.1f}s")
    _write_video(frames, output_path, fps)


def render_sim2sim(
    bvh_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    max_frames: int = 0,
    bvh_format: str = "lafan1",
) -> None:
    """Render full sim2sim: BVH → GMR → obs → ONNX policy → PD control → MuJoCo."""
    project_root = _find_project_root()
    cfgs = _load_configs(str(bvh_path), project_root, bvh_format)

    POLICY_HZ = int(cfgs["policy_hz"])
    PD_HZ = int(cfgs["pd_hz"])

    from omegaconf import OmegaConf

    from teleopit.pipeline import TeleopPipeline
    from teleopit.retargeting.core import extract_mimic_obs

    sim2sim_xml = (
        project_root
        / "teleopit"
        / "retargeting"
        / "gmr"
        / "assets"
        / "unitree_g1"
        / "g1_sim2sim_29dof.xml"
    )
    cfgs["robot"].xml_path = str(sim2sim_xml)

    cfg = OmegaConf.create(
        {
            "robot": cfgs["robot"],
            "controller": cfgs["controller"],
            "input": cfgs["input"],
            "policy_hz": POLICY_HZ,
            "pd_hz": PD_HZ,
            "recording": {"output_path": "/tmp/teleopit_render.h5"},
        }
    )
    pipeline = TeleopPipeline(cfg)

    model = pipeline.robot.model
    data = pipeline.robot.data
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = _make_camera()

    input_prov = cast(Any, pipeline.input_provider)
    retargeter = cast(Any, pipeline.retargeter)
    loop = pipeline.loop

    n_bvh = len(input_prov)
    bvh_duration = n_bvh / fps
    if max_frames > 0:
        bvh_duration = min(bvh_duration, max_frames / fps)
    num_policy_steps = int(bvh_duration * POLICY_HZ)
    print(f"  [sim2sim] {n_bvh} BVH frames @ {fps}fps = {bvh_duration:.1f}s")
    print(
        f"  [sim2sim] Running {num_policy_steps} policy steps @ {POLICY_HZ}Hz, PD @ {PD_HZ}Hz (decimation={loop.decimation})"
    )

    bvh_frames_all = input_prov._frames

    frames: list[np.ndarray] = []
    t0 = time.time()
    last_bvh_idx = -1

    for step in range(num_policy_steps):
        policy_time = step / POLICY_HZ
        bvh_idx = min(int(policy_time * fps), n_bvh - 1)

        if bvh_idx != last_bvh_idx:
            raw_frame = bvh_frames_all[bvh_idx]
            human_frame = {
                body_name: (np.array(d[0]), np.array(d[1]))
                for body_name, d in raw_frame.items()
            }
            retargeted = retargeter.retarget(human_frame)
            qpos = loop._retarget_to_qpos(retargeted)
            last_bvh_idx = bvh_idx

        mimic_obs = extract_mimic_obs(
            qpos=qpos, last_qpos=loop._last_retarget_qpos, dt=1.0 / POLICY_HZ
        )

        state = pipeline.robot.get_state()
        obs = loop._build_observation(
            state=state, mimic_obs=mimic_obs, last_action=loop._last_action
        )
        policy_obs = loop._adapt_observation_for_policy(obs)
        action = np.asarray(
            pipeline.controller.compute_action(policy_obs), dtype=np.float32
        ).reshape(-1)

        target_dof_pos = loop._compute_target_dof_pos(action)

        for _ in range(loop.decimation):
            pd_state = pipeline.robot.get_state()
            dof_pos = np.asarray(pd_state.qpos, dtype=np.float32)[: loop._num_actions]
            dof_vel = np.asarray(pd_state.qvel, dtype=np.float32)[: loop._num_actions]
            torque = (target_dof_pos - dof_pos) * loop._kps - dof_vel * loop._kds
            torque = np.clip(torque, -loop._torque_limits, loop._torque_limits).astype(
                np.float32
            )
            pipeline.robot.set_action(torque)
            pipeline.robot.step()

        loop._last_action = action
        loop._last_retarget_qpos = qpos.copy()

        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]

        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        frames.append(frame.copy())

        if (step + 1) % 100 == 0:
            print(f"    Step {step + 1}/{num_policy_steps} ({time.time() - t0:.1f}s)")

    renderer.close()

    if not frames:
        print("  WARNING: No frames rendered!")
        return

    print(f"  [sim2sim] Done: {len(frames)} frames in {time.time() - t0:.1f}s")
    _write_video(frames, output_path, POLICY_HZ)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render BVH + retarget + sim2sim verification videos"
    )
    parser.add_argument("--bvh", required=True, help="Path to a single BVH file")
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=0,
        help="Max video duration in seconds (0=full BVH)",
    )
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=360, help="Video height")
    parser.add_argument(
        "--format", type=str, default="lafan1", help="BVH format (lafan1 or hc_mocap)"
    )
    args = parser.parse_args()

    project_root = _find_project_root()
    bvh_path = Path(args.bvh)
    if not bvh_path.is_absolute():
        bvh_path = (project_root / bvh_path).resolve()

    if not bvh_path.exists():
        print(f"BVH file not found: {bvh_path}")
        sys.exit(1)

    fps = _read_bvh_fps(bvh_path)
    # hc_mocap 60fps is downsampled to 30fps by BVHInputProvider — match here
    if args.format == "hc_mocap" and fps == 60:
        fps = 30
    max_frames = int(fps * args.max_seconds) if args.max_seconds > 0 else 0

    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    stem = bvh_path.stem

    print(
        f"Processing: {bvh_path.name} (native {fps}fps"
        + (f", cap {args.max_seconds:.0f}s)" if max_frames else ", full)")
    )

    bvh_out = output_dir / f"{stem}_bvh.mp4"
    print(f"\n=== Pass 1: BVH Skeleton ===")
    render_bvh(
        bvh_path=bvh_path,
        output_path=bvh_out,
        width=args.width,
        height=args.height,
        fps=fps,
        max_frames=max_frames,
        bvh_format=args.format,
    )

    retarget_out = output_dir / f"{stem}_retarget.mp4"
    print(f"\n=== Pass 2: GMR Retarget ===")
    render_retarget(
        bvh_path=bvh_path,
        output_path=retarget_out,
        width=args.width,
        height=args.height,
        fps=fps,
        max_frames=max_frames,
        bvh_format=args.format,
    )

    sim2sim_out = output_dir / f"{stem}_sim2sim.mp4"
    print(f"\n=== Pass 3: Sim2Sim ===")
    render_sim2sim(
        bvh_path=bvh_path,
        output_path=sim2sim_out,
        width=args.width,
        height=args.height,
        fps=fps,
        max_frames=max_frames,
        bvh_format=args.format,
    )

    print(f"\nDone! Videos saved to {output_dir}/")
    print(f"  BVH:      {bvh_out.name}")
    print(f"  Retarget: {retarget_out.name}")
    print(f"  Sim2Sim:  {sim2sim_out.name}")


if __name__ == "__main__":
    main()
