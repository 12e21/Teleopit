#!/usr/bin/env python3
"""Play back a trained tracking policy in simulation.

Viewer options:
  native  -- MuJoCo native window (default, requires display)
  viser   -- browser-based 3D viewer at http://localhost:8012

Usage:
    # Native window
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets/builds/twist2_full/val.npz

    # Browser viewer (no display required)
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets/builds/twist2_full/val.npz \
        --viewer viser

    # Record video instead of interactive viewer
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets/builds/twist2_full/val.npz \
        --video
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401 -- populates mjlab built-in tasks
import train_mimic.tasks  # noqa: F401 -- registers our custom tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from train_mimic.scripts.train import _validate_motion_file
from train_mimic.tasks.tracking.config.g1.flat_env_cfg import DEFAULT_TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play trained G1 tracking policy.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to NPZ motion file")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--viewer", type=str, default="native", choices=["native", "viser"],
        help="native: MuJoCo window (requires display); viser: browser at localhost:8012",
    )
    parser.add_argument("--video", action="store_true", help="Record video instead of interactive viewer")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        raise SystemExit(1)
    _validate_motion_file(args.motion_file)

    configure_torch_backends()

    # Load configs (play=True disables corruption, push_robot, etc.)
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    # Override for playback
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands["motion"].motion_file = args.motion_file

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # render_mode only needed for video recording
    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if args.video:
        from mjlab.utils.wrappers import VideoRecorder
        log_dir = os.path.dirname(args.checkpoint)
        env = VideoRecorder(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=500,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy (force tensorboard to avoid wandb init during playback).
    log_dir = os.path.dirname(args.checkpoint)
    agent_dict = asdict(agent_cfg)
    agent_dict["logger"] = "tensorboard"
    RunnerCls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = RunnerCls(env, agent_dict, log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    if args.video:
        # Run a fixed number of steps then close
        obs = env.get_observations()
        for _ in range(500):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    elif args.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
