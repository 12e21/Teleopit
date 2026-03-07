#!/usr/bin/env python3
"""Benchmark a trained tracking policy on motion clips.

Runs policy rollout for a fixed number of evaluation steps and reports
distribution statistics for motion-tracking errors.

Can optionally render and save a benchmark video for qualitative inspection.

Usage:
    python train_mimic/scripts/benchmark.py --task Tracking-Flat-G1-v0 \
        --checkpoint logs/rsl_rl/g1_tracking/.../model_30000.pt \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/sub10_clothesstand_000.npz \
        --num_envs 1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _make_tracking_camera(mujoco_module: object, target_xyz: np.ndarray):
    cam = mujoco_module.MjvCamera()
    cam.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(target_xyz, dtype=np.float64)
    cam.distance = 3.0
    cam.azimuth = 135.0
    cam.elevation = -15.0
    return cam


def _render_split_frame(unwrapped: object, command: object, mujoco_module: object) -> np.ndarray:
    sim = getattr(unwrapped, "sim")
    sim.update_render()
    renderer = sim.renderer

    ref_target = np.asarray(command.anchor_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
    robot_target = np.asarray(command.robot_anchor_pos_w[0].detach().cpu().numpy(), dtype=np.float64)

    ref_cam = _make_tracking_camera(mujoco_module, ref_target)
    renderer.update_scene(data=sim.mj_data, camera=ref_cam)
    unwrapped.update_visualizers(renderer.scene)
    ref_frame = renderer.render().copy()

    robot_cam = _make_tracking_camera(mujoco_module, robot_target)
    renderer.update_scene(data=sim.mj_data, camera=robot_cam)
    unwrapped.update_visualizers(renderer.scene)
    robot_frame = renderer.render().copy()

    return np.concatenate((ref_frame, robot_frame), axis=1)


def _to_float(value: object, torch_module: object) -> float:
    if isinstance(value, torch_module.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.float().mean().item())
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"Unsupported value type: {type(value)}")


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark G1 tracking policy.")
    parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_eval_steps", type=int, default=2000,
                        help="Number of rollout steps for evaluation (default: 2000)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps ignored from metric aggregation (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true",
                        help="Record split-screen benchmark video (left=reference, right=robot)")
    parser.add_argument("--video_length", type=int, default=600,
                        help="Recorded video length in steps (default: 600)")
    parser.add_argument("--video_folder", type=str, default=None,
                        help="Output folder for benchmark video")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be >= 0")
    if args.num_eval_steps <= args.warmup_steps:
        raise ValueError("--num_eval_steps must be greater than --warmup_steps")
    if args.video and args.num_envs != 1:
        raise ValueError("--video currently requires --num_envs 1")

    # Set render backend before importing modules that may initialize MuJoCo/GL.
    if args.video and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("[INFO] --video enabled, MUJOCO_GL not set. Defaulting to MUJOCO_GL=egl.")
    if args.video and "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("[INFO] --video enabled, PYOPENGL_PLATFORM not set. Defaulting to PYOPENGL_PLATFORM=egl.")

    import gymnasium as gym
    import torch

    import train_mimic.tasks  # noqa: F401 -- triggers gym.register()
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
        load_cfg_from_registry,
    )
    from mjlab.utils.torch import configure_torch_backends
    from rsl_rl.runners import OnPolicyRunner
    from train_mimic.scripts.train import _to_rsl_rl5_cfg

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    configure_torch_backends()

    # Load configs
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rl_cfg_entry_point")

    # Configure for benchmark
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = args.motion_file
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create env
    render_mode = "rgb_array" if args.video else None
    try:
        env = gym.make(args.task, cfg=env_cfg, device=device, render_mode=render_mode)
    except Exception as exc:
        if args.video:
            raise RuntimeError(
                "Video renderer initialization failed. "
                "Try setting MUJOCO_GL=egl (or osmesa) and make sure the corresponding "
                "OpenGL backend libraries are available on this machine."
            ) from exc
        else:
            raise

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    video_writer = None
    video_path: Path | None = None
    video_steps = 0
    mujoco = None
    if args.video:
        import imageio.v2 as imageio
        import mujoco as mujoco_module

        video_folder = args.video_folder or "benchmark_results/videos"
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        video_path = Path(video_folder) / "benchmark_split.mp4"
        video_steps = min(args.video_length, args.num_eval_steps)
        video_fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))
        video_writer = imageio.get_writer(str(video_path), fps=video_fps, quality=8)
        mujoco = mujoco_module
        print(f"[INFO] Recording split benchmark video to: {video_path}")
        print("[INFO] Split layout: left=reference, right=robot")

    log_dir = os.path.dirname(args.checkpoint)
    runner = OnPolicyRunner(env, _to_rsl_rl5_cfg(asdict(agent_cfg)), log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Benchmark
    obs = env.get_observations()
    unwrapped = env.unwrapped

    cmd = unwrapped.command_manager.get_term("motion")
    metric_keys = sorted(cmd.metrics.keys())
    metric_series: dict[str, list[float]] = {k: [] for k in metric_keys}
    reward_series: list[float] = []
    reset_log_series: dict[str, list[float]] = {}

    done_events = 0
    timeout_events = 0
    ep_len_buf = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    completed_episode_lengths: list[int] = []

    try:
        for step in range(args.num_eval_steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            ep_len_buf += 1

            if video_writer is not None and mujoco is not None and step < video_steps:
                frame = _render_split_frame(env.unwrapped, cmd, mujoco)
                video_writer.append_data(frame)

            done_mask = dones > 0
            num_done = int(done_mask.sum().item())
            if num_done > 0:
                done_events += num_done
                completed_episode_lengths.extend(ep_len_buf[done_mask].detach().cpu().tolist())
                ep_len_buf[done_mask] = 0
                # Pull reset-time logs (episode reward breakdown, termination stats, etc.)
                # from the underlying environment.
                extras_log = extras.get("log", {}) if isinstance(extras, dict) else {}
                if isinstance(extras_log, dict):
                    for key, value in extras_log.items():
                        if key.startswith(("Episode_Reward/", "Episode_Termination/", "Metrics/motion/")):
                            reset_log_series.setdefault(key, []).append(_to_float(value, torch))

            if step < args.warmup_steps:
                continue

            reward_series.append(_to_float(rewards, torch))
            for key in metric_keys:
                metric_series[key].append(_to_float(cmd.metrics[key], torch))

            # Wrapper may expose timeout flags in extras for infinite-horizon settings.
            if isinstance(extras, dict) and "time_outs" in extras and isinstance(extras["time_outs"], torch.Tensor):
                timeout_events += int(extras["time_outs"].sum().item())
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    effective_steps = args.num_eval_steps - args.warmup_steps
    if effective_steps <= 0:
        raise RuntimeError("No effective evaluation steps. Increase --num_eval_steps or decrease --warmup_steps.")

    metric_stats = {key: _stats(vals) for key, vals in metric_series.items()}
    reward_stats = _stats(reward_series)
    reset_log_stats = {key: _stats(vals) for key, vals in reset_log_series.items()}

    anchor_pos = metric_stats.get("error_anchor_pos", {}).get("mean", float("nan"))
    anchor_rot = metric_stats.get("error_anchor_rot", {}).get("mean", float("nan"))
    body_pos = metric_stats.get("error_body_pos", {}).get("mean", float("nan"))
    total = anchor_pos + anchor_rot + body_pos

    eval_transitions = args.num_eval_steps * args.num_envs
    done_rate = done_events / max(eval_transitions, 1)
    timeout_rate = timeout_events / max(eval_transitions, 1)
    ep_len_stats = _stats([float(v) for v in completed_episode_lengths])

    print(f"\nBenchmark Results ({effective_steps} effective steps, warmup {args.warmup_steps}):")
    print(f"  total_error(anchor_pos+anchor_rot+body_pos): {total:.4f}")
    print(f"  error_anchor_pos: {anchor_pos:.4f}")
    print(f"  error_anchor_rot: {anchor_rot:.4f}")
    print(f"  error_body_pos: {body_pos:.4f}")
    print(f"  mean_step_reward: {reward_stats['mean']:.4f}")
    print(f"  done_rate: {done_rate:.4f}")
    print(f"  timeout_rate: {timeout_rate:.4f}")
    print(f"  completed_episodes: {len(completed_episode_lengths)}")
    print(f"  mean_episode_length: {ep_len_stats['mean']:.2f}")

    print("\nMetric distributions (mean / p50 / p95):")
    for key in (
        "error_anchor_pos",
        "error_anchor_rot",
        "error_anchor_lin_vel",
        "error_anchor_ang_vel",
        "error_body_pos",
        "error_body_rot",
        "error_body_lin_vel",
        "error_body_ang_vel",
        "error_joint_pos",
        "error_joint_vel",
        "sampling_entropy",
        "sampling_top1_prob",
        "sampling_top1_bin",
    ):
        if key not in metric_stats:
            continue
        stats = metric_stats[key]
        print(f"  {key}: {stats['mean']:.4f} / {stats['p50']:.4f} / {stats['p95']:.4f}")

    Path("benchmark_results").mkdir(exist_ok=True)
    output_path = Path("benchmark_results") / f"{args.task}-{Path(args.checkpoint).stem}.txt"
    json_path = Path("benchmark_results") / f"{args.task}-{Path(args.checkpoint).stem}.json"

    lines = [
        f"checkpoint: {args.checkpoint}",
        f"motion_file: {args.motion_file}",
        f"num_envs: {args.num_envs}",
        f"num_eval_steps: {args.num_eval_steps}",
        f"warmup_steps: {args.warmup_steps}",
        f"effective_steps: {effective_steps}",
        "",
        f"total_error(anchor_pos+anchor_rot+body_pos): {total:.6f}",
        f"error_anchor_pos_mean: {anchor_pos:.6f}",
        f"error_anchor_rot_mean: {anchor_rot:.6f}",
        f"error_body_pos_mean: {body_pos:.6f}",
        f"mean_step_reward: {reward_stats['mean']:.6f}",
        f"done_rate: {done_rate:.6f}",
        f"timeout_rate: {timeout_rate:.6f}",
        f"completed_episodes: {len(completed_episode_lengths)}",
        f"mean_episode_length: {ep_len_stats['mean']:.6f}",
        "",
        "metric_stats(mean,std,p50,p95,min,max):",
    ]
    for key in sorted(metric_stats.keys()):
        s = metric_stats[key]
        lines.append(
            f"{key}: {s['mean']:.6f}, {s['std']:.6f}, {s['p50']:.6f}, {s['p95']:.6f}, {s['min']:.6f}, {s['max']:.6f}"
        )
    if reset_log_stats:
        lines.append("")
        lines.append("reset_log_stats(mean,std,p50,p95,min,max):")
        for key in sorted(reset_log_stats.keys()):
            s = reset_log_stats[key]
            lines.append(
                f"{key}: {s['mean']:.6f}, {s['std']:.6f}, {s['p50']:.6f}, {s['p95']:.6f}, {s['min']:.6f}, {s['max']:.6f}"
            )
    output_path.write_text("\n".join(lines) + "\n")

    report = {
        "checkpoint": args.checkpoint,
        "motion_file": args.motion_file,
        "num_envs": args.num_envs,
        "num_eval_steps": args.num_eval_steps,
        "warmup_steps": args.warmup_steps,
        "effective_steps": effective_steps,
        "total_error": total,
        "mean_step_reward": reward_stats["mean"],
        "done_rate": done_rate,
        "timeout_rate": timeout_rate,
        "completed_episodes": len(completed_episode_lengths),
        "mean_episode_length": ep_len_stats["mean"],
        "metric_stats": metric_stats,
        "reset_log_stats": reset_log_stats,
    }
    json_path.write_text(json.dumps(report, indent=2))

    print(f"\nSaved summary to: {output_path}")
    print(f"Saved detailed json to: {json_path}")
    if video_path is not None:
        print(f"Saved split video to: {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
