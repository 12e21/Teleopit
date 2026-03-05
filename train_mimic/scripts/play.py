#!/usr/bin/env python3
"""Play back a trained tracking policy in simulation.

Viewer options:
  native  -- MuJoCo native window (default, requires display)
  viser   -- browser-based 3D viewer at http://localhost:8012

Usage:
    # Native window
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz

    # Browser viewer (no display required)
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
        --viewer viser

    # Record video instead of interactive viewer
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
        --video
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import gymnasium as gym
import torch

import train_mimic.tasks  # noqa: F401 -- triggers gym.register()
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserViewer
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner
from train_mimic.scripts.train import _to_rsl_rl5_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play trained G1 tracking policy.")
    parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
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

    configure_torch_backends()

    # Load configs
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rl_cfg_entry_point")

    # Override for playback
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = args.motion_file
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = int(1e9)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # render_mode only needed for video recording
    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, device=device, render_mode=render_mode)

    if args.video:
        log_dir = os.path.dirname(args.checkpoint)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=500,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    log_dir = os.path.dirname(args.checkpoint)
    runner = OnPolicyRunner(env, _to_rsl_rl5_cfg(asdict(agent_cfg)), log_dir=log_dir, device=device)
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
        ViserViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
