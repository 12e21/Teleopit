#!/usr/bin/env python3
"""Train G1 whole-body tracking policy with mjlab + rsl_rl PPO.

Usage:
    python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
        --num_envs 4096 --max_iterations 30000 \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz

    # Quick verification
    python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
        --num_envs 64 --max_iterations 100 \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz

    # With wandb logging
    python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
        --num_envs 4096 --max_iterations 30000 \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
        --wandb_project teleopit
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime

import gymnasium as gym
import torch

import train_mimic.tasks  # noqa: F401 -- triggers gym.register()
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train G1 tracking policy (mjlab).")
    parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Enable wandb and set project name (default: tensorboard)")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--motion_file", type=str, default=None,
                        help="NPZ motion file path (single file, use --merge to create one)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--video", action="store_true",
                        help="Record periodic videos during training")
    parser.add_argument("--video_interval", type=int, default=2000,
                        help="Record a video every N iterations (default: 2000)")
    parser.add_argument("--video_length", type=int, default=200,
                        help="Number of steps per video clip (default: 200)")
    return parser.parse_args()


def _to_rsl_rl5_cfg(cfg: dict) -> dict:
    """Convert mjlab's RslRlOnPolicyRunnerCfg dict to rsl_rl 5.x format.

    mjlab uses a single "policy" key (ActorCritic style), but rsl_rl 5.x
    expects separate "actor" and "critic" keys with MLPModel config.
    """
    policy = cfg.pop("policy")
    cfg["actor"] = {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": policy["actor_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": policy["actor_obs_normalization"],
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": policy["init_noise_std"],
            "std_type": policy.get("noise_std_type", "scalar"),
        },
    }
    cfg["critic"] = {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": policy["critic_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": policy["critic_obs_normalization"],
    }
    return cfg


def main() -> None:
    args = parse_args()

    configure_torch_backends()

    # Load configs from registry
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rl_cfg_entry_point")

    # CLI overrides
    env_cfg.seed = args.seed
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.motion_file is not None:
        env_cfg.commands.motion.motion_file = args.motion_file
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.wandb_project is not None:
        agent_cfg.logger = "wandb"
        agent_cfg.wandb_project = args.wandb_project

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Log directory (defined before env creation so video path is available)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # render_mode only needed for video recording
    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, device=device, render_mode=render_mode)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % (args.video_interval * env_cfg.decimation) == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner
    runner = OnPolicyRunner(env, _to_rsl_rl5_cfg(asdict(agent_cfg)), log_dir=log_dir, device=device)

    if args.resume is not None:
        print(f"[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
