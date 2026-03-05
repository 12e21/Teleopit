#!/usr/bin/env python3
"""Benchmark a trained tracking policy on motion clips.

Reports per-clip and aggregate tracking errors for anchor position,
anchor orientation, and per-body key-body positions.

Usage:
    python train_mimic/scripts/benchmark.py --task Tracking-Flat-G1-v0 \
        --checkpoint logs/rsl_rl/g1_tracking/.../model_30000.pt \
        --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/sub10_clothesstand_000.npz \
        --num_envs 1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

from train_mimic.scripts.train import _to_rsl_rl5_cfg
from pathlib import Path

import gymnasium as gym
import torch

import train_mimic.tasks  # noqa: F401 -- triggers gym.register()
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.mdp import quat_error_magnitude, quat_inv, quat_mul
from rsl_rl.runners import OnPolicyRunner
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark G1 tracking policy.")
    parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    configure_torch_backends()

    # Load configs
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rl_cfg_entry_point")

    # Configure for benchmark
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = args.motion_file
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create env
    env = gym.make(args.task, cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = os.path.dirname(args.checkpoint)
    runner = OnPolicyRunner(env, _to_rsl_rl5_cfg(asdict(agent_cfg)), log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Benchmark
    obs = env.get_observations()
    unwrapped = env.unwrapped

    cmd = unwrapped.command_manager.get_term("motion")
    max_steps = int(env_cfg.episode_length_s / unwrapped.step_dt)

    anchor_pos_errors = []
    anchor_ori_errors = []
    body_pos_errors = []

    for step in range(max_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        # Compute errors
        pos_err = torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1).mean().item()
        anchor_pos_errors.append(pos_err)

        quat_diff = quat_mul(quat_inv(cmd.robot_anchor_quat_w), cmd.anchor_quat_w)
        ori_err = (2.0 * torch.acos(torch.clamp(quat_diff[:, 0].abs(), -1.0, 1.0))).mean().item()
        anchor_ori_errors.append(ori_err)

        body_err = torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1).mean().item()
        body_pos_errors.append(body_err)

        if torch.any(dones).item():
            break

    env.close()

    # Report
    n = len(anchor_pos_errors)
    avg_pos = sum(anchor_pos_errors) / n
    avg_ori = sum(anchor_ori_errors) / n
    avg_body = sum(body_pos_errors) / n
    total = avg_pos + avg_ori + avg_body

    print(f"\nBenchmark Results ({n} steps):")
    print(f"  anchor_position_error: {avg_pos:.4f}")
    print(f"  anchor_orientation_error: {avg_ori:.4f}")
    print(f"  body_position_error: {avg_body:.4f}")
    print(f"  total_error: {total:.4f}")

    # Save results
    Path("benchmark_results").mkdir(exist_ok=True)
    output_path = Path("benchmark_results") / f"{args.task}-{Path(args.checkpoint).stem}.txt"
    lines = [
        f"total_error: {total:.4f}",
        f"anchor_position_error: {avg_pos:.4f}",
        f"anchor_orientation_error: {avg_ori:.4f}",
        f"body_position_error: {avg_body:.4f}",
        f"steps: {n}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
