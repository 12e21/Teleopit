#!/usr/bin/env python3
"""Benchmark a trained tracking policy on motion clips.

Runs policy rollout for a fixed number of evaluation steps and reports
distribution statistics for motion-tracking errors.

Can optionally render and save benchmark videos for qualitative inspection.

Usage:
    # Benchmark only (no video)
    python train_mimic/scripts/benchmark.py --task Tracking-Flat-G1-v0 \
        --checkpoint logs/rsl_rl/g1_tracking/.../model_30000.pt \
        --motion_file data/datasets/builds/twist2_full/val.npz \
        --num_envs 1

    # Single video (one continuous clip)
    python train_mimic/scripts/benchmark.py ... --video --video_length 500

    # Multiple separate clip videos
    python train_mimic/scripts/benchmark.py ... --video --num_clips 10 --video_length 250
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from train_mimic.scripts.train import _validate_motion_file


def _render_frame(unwrapped: object, split: bool = False, _cmd: object = None) -> np.ndarray:
    """Render a frame using the environment's offline renderer (ghost included).

    Args:
        unwrapped: The unwrapped ManagerBasedRlEnv.
        split: If True, render split-screen with camera lookat on ref (left)
               and robot (right). Same scene, different camera targets.
        _cmd: MotionCommand term (required when split=True).
    """
    if not split:
        frame = unwrapped.render()
        if frame is None:
            raise RuntimeError("render() returned None; ensure render_mode='rgb_array'")
        return frame

    import mujoco

    renderer = unwrapped._offline_renderer
    cam = renderer._cam
    env_idx = max(0, min(int(renderer._cfg.env_idx), int(unwrapped.sim.data.nworld) - 1))

    # Full scene update (robot + ghost via debug vis).
    debug_callback = (
        unwrapped.update_visualizers if hasattr(unwrapped, "update_visualizers") else None
    )
    renderer.update(unwrapped.sim.data, debug_vis_callback=debug_callback)

    # Save original camera state.
    orig_type = cam.type
    orig_trackbodyid = cam.trackbodyid
    orig_lookat = cam.lookat.copy()

    # --- Left: camera follows ref pose ---
    ref_pos = _cmd.body_pos_w[env_idx, 0].cpu().numpy()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    cam.trackbodyid = -1
    cam.lookat[:] = [ref_pos[0], ref_pos[1], 0.8]
    renderer._renderer.update_scene(renderer._data, camera=cam)
    # Re-apply ghost geoms after update_scene reset.
    if debug_callback is not None:
        from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
        vis = MujocoNativeDebugVisualizer(
            renderer._renderer.scene, renderer._model, env_idx=renderer._cfg.env_idx
        )
        debug_callback(vis)
    frame_ref = renderer._renderer.render()

    # --- Right: camera follows robot ---
    robot_pos = unwrapped.sim.data.qpos[env_idx, :3].cpu().numpy()
    cam.lookat[:] = [robot_pos[0], robot_pos[1], 0.8]
    renderer._renderer.update_scene(renderer._data, camera=cam)
    if debug_callback is not None:
        vis = MujocoNativeDebugVisualizer(
            renderer._renderer.scene, renderer._model, env_idx=renderer._cfg.env_idx
        )
        debug_callback(vis)
    frame_robot = renderer._renderer.render()

    # Restore camera.
    cam.type = orig_type
    cam.trackbodyid = orig_trackbodyid
    cam.lookat[:] = orig_lookat

    return np.concatenate([frame_ref, frame_robot], axis=1)


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
    parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0-NoStateEst")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_eval_steps", type=int, default=2000,
                        help="Number of rollout steps for evaluation (default: 2000)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps ignored from metric aggregation (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true",
                        help="Record benchmark video(s)")
    parser.add_argument("--num_clips", type=int, default=1,
                        help="Number of separate video clips to render (default: 1)")
    parser.add_argument("--video_length", type=int, default=600,
                        help="Steps per video clip (default: 600)")
    parser.add_argument("--video_folder", type=str, default=None,
                        help="Output folder for benchmark video(s)")
    parser.add_argument("--split", action="store_true",
                        help="Render split-screen video with two camera angles")
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
    _validate_motion_file(args.motion_file)

    # Set render backend before importing modules that may initialize MuJoCo/GL.
    if args.video and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("[INFO] --video enabled, MUJOCO_GL not set. Defaulting to MUJOCO_GL=egl.")
    if args.video and "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("[INFO] --video enabled, PYOPENGL_PLATFORM not set. Defaulting to PYOPENGL_PLATFORM=egl.")

    import torch

    import mjlab.tasks  # noqa: F401 -- populates mjlab built-in tasks
    import train_mimic.tasks  # noqa: F401 -- registers our custom tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    configure_torch_backends()

    # Load configs (play=True disables corruption, push_robot, etc.)
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    # Configure for benchmark
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands["motion"].motion_file = args.motion_file
    env_cfg.commands["motion"].pose_range = {}
    env_cfg.commands["motion"].velocity_range = {}

    # Use uniform sampling so each reset picks a different motion segment.
    if args.video and args.num_clips > 1:
        env_cfg.commands["motion"].sampling_mode = "uniform"

    step_dt = float(env_cfg.decimation) * float(env_cfg.sim.mujoco.timestep)
    required_episode_s = args.num_eval_steps * step_dt + 1.0
    if float(env_cfg.episode_length_s) < required_episode_s:
        env_cfg.episode_length_s = required_episode_s
    if args.video:
        env_cfg.terminations.pop("time_out", None)
        env_cfg.terminations.pop("anchor_pos", None)
        env_cfg.terminations.pop("anchor_ori", None)
        env_cfg.terminations.pop("ee_body_pos", None)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create env
    render_mode = "rgb_array" if args.video else None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
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

    log_dir = os.path.dirname(args.checkpoint)
    agent_dict = asdict(agent_cfg)
    agent_dict["logger"] = "tensorboard"  # Never init wandb for evaluation.
    RunnerCls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = RunnerCls(env, agent_dict, log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # --- Video recording setup ---
    video_writer = None
    video_folder: Path | None = None
    video_paths: list[Path] = []
    clip_step_counter = 0
    clip_idx = 0

    if args.video:
        import imageio.v2 as imageio

        video_folder = Path(args.video_folder or "benchmark_results/videos")
        video_folder.mkdir(parents=True, exist_ok=True)
        video_fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))

        def _open_clip_writer(idx: int):
            nonlocal video_writer, clip_step_counter
            if args.num_clips == 1:
                path = video_folder / "benchmark.mp4"
            else:
                path = video_folder / f"clip_{idx:03d}.mp4"
            video_paths.append(path)
            video_writer = imageio.get_writer(str(path), fps=video_fps, quality=8)
            clip_step_counter = 0
            print(f"[INFO] Recording clip {idx + 1}/{args.num_clips}: {path}")

        _open_clip_writer(0)

    # --- Benchmark loop ---
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

            # Record video frame.
            if video_writer is not None and clip_idx < args.num_clips:
                frame = _render_frame(env.unwrapped, split=args.split, _cmd=cmd)
                video_writer.append_data(frame)
                clip_step_counter += 1

                # Close current clip and open next one.
                if clip_step_counter >= args.video_length:
                    video_writer.close()
                    video_writer = None
                    clip_idx += 1
                    if clip_idx < args.num_clips:
                        # Reset env to sample a new motion segment.
                        obs, _ = env.reset()
                        ep_len_buf[:] = 0
                        _open_clip_writer(clip_idx)

            done_mask = dones > 0
            num_done = int(done_mask.sum().item())
            if num_done > 0:
                done_events += num_done
                completed_episode_lengths.extend(ep_len_buf[done_mask].detach().cpu().tolist())
                ep_len_buf[done_mask] = 0
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

            if isinstance(extras, dict) and "time_outs" in extras and isinstance(extras["time_outs"], torch.Tensor):
                timeout_events += int(extras["time_outs"].sum().item())
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    # --- Report ---
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
    for vp in video_paths:
        print(f"Saved video: {vp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
