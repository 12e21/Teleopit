#!/usr/bin/env python3

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportUnusedImport=false, reportAny=false, reportExplicitAny=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportArgumentType=false

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Benchmark trained G1 motion tracking policy in Isaac Lab.")
parser.add_argument("--task", type=str, default="Isaac-G1-Mimic-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--motion_file", type=str, default=None, help="Override motion file")
AppLauncher.add_app_launcher_args(parser)
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser.print_help()
    raise SystemExit(0)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import train_mimic.envs  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from train_mimic.rsl_rl.modules.actor_critic_mimic import ActorCriticMimic  # noqa: E402
from train_mimic.rsl_rl.runners.on_policy_runner_mimic import OnPolicyRunnerMimic  # noqa: F401, E402


def _to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Unsupported cfg type: {type(cfg)}")


def _policy_obs(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict):
        return obs["policy"]
    if hasattr(obs, "keys") and "policy" in obs.keys():
        return obs["policy"]
    if isinstance(obs, torch.Tensor):
        return obs
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _configure_env_cfg(env_cfg: Any, cli_args: argparse.Namespace) -> None:
    env_cfg.scene.num_envs = cli_args.num_envs

    if hasattr(env_cfg, "init_state") and hasattr(env_cfg.init_state, "pos"):
        env_cfg.init_state.pos = [0.0, 0.0, 0.75]

    _safe_setattr(env_cfg, "rand_reset", False)

    if hasattr(env_cfg, "motion"):
        _safe_setattr(env_cfg.motion, "motion_curriculum", False)
        if cli_args.motion_file:
            _safe_setattr(env_cfg.motion, "motion_file", cli_args.motion_file)

    if hasattr(env_cfg, "noise"):
        _safe_setattr(env_cfg.noise, "add_noise", False)

    if hasattr(env_cfg, "domain_rand"):
        for attr in (
            "randomize_gravity",
            "randomize_friction",
            "randomize_base_mass",
            "randomize_base_com",
            "push_robots",
            "randomize_motor",
            "action_delay",
            "domain_rand_general",
            "randomize_start_pos",
            "randomize_start_yaw",
        ):
            _safe_setattr(env_cfg.domain_rand, attr, False)

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        _safe_setattr(env_cfg.observations.policy, "enable_corruption", False)


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "ac_state_dict" in checkpoint:
        return checkpoint["ac_state_dict"]
    return checkpoint


def _load_policy(
    checkpoint_path: str,
    env_cfg: Any,
    runner_cfg: Any,
    device: torch.device,
) -> tuple[ActorCriticMimic, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_state_dict = _extract_state_dict(checkpoint)

    policy_cfg = _to_dict(runner_cfg.policy)
    policy = ActorCriticMimic(
        num_observations=int(env_cfg.num_observations),
        num_critic_observations=int(env_cfg.num_observations),
        num_motion_observations=int(env_cfg.n_priv_mimic_obs),
        num_motion_steps=int(policy_cfg.get("num_motion_steps", 10)),
        num_actions=int(env_cfg.num_actions),
        actor_hidden_dims=policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
        critic_hidden_dims=policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
        motion_latent_dim=int(policy_cfg.get("motion_latent_dim", 64)),
        activation=policy_cfg.get("activation", "elu"),
        init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
        fix_action_std=bool(policy_cfg.get("fix_action_std", False)),
        action_std=policy_cfg.get("action_std", None),
        layer_norm=bool(policy_cfg.get("layer_norm", False)),
    ).to(device)
    result = policy.load_state_dict(policy_state_dict, strict=False)
    if result.missing_keys:
        print(f"[WARNING] Missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARNING] Unexpected keys in checkpoint: {result.unexpected_keys}")
    policy.eval()

    normalizer = checkpoint.get("normalizer", None)
    if normalizer is not None and hasattr(normalizer, "to"):
        normalizer.to(device)

    return policy, normalizer


def _reset_to_motion(unwrapped_env: Any, motion_id: int) -> torch.Tensor:
    motion_lib = unwrapped_env._motion_lib
    env_ids = torch.arange(unwrapped_env.num_envs, device=unwrapped_env.device, dtype=torch.long)
    fixed_motion_ids = torch.full((unwrapped_env.num_envs,), motion_id, device=unwrapped_env.device, dtype=torch.long)

    def _fixed_sampler(n: int, max_difficulty: Any = None) -> torch.Tensor:  # noqa: ARG001
        _ = max_difficulty
        return fixed_motion_ids[:n]

    original_sampler = motion_lib.sample_motions
    motion_lib.sample_motions = _fixed_sampler
    try:
        unwrapped_env._reset_idx(env_ids)
    finally:
        motion_lib.sample_motions = original_sampler

    obs_dict = unwrapped_env._get_observations()
    return _policy_obs(obs_dict)


def _keybody_groups(num_bodies: int) -> dict[str, list[int]]:
    if num_bodies >= 10:
        return {
            "hand": [3, 4],
            "feet": [1, 2],
            "knee": [5, 6],
            "elbow": [7, 8],
            "head": [9],
        }
    if num_bodies == 9:
        return {
            "hand": [0, 1],
            "feet": [2, 3],
            "knee": [4, 5],
            "elbow": [6, 7],
            "head": [8],
        }
    return {
        "hand": list(range(min(2, num_bodies))),
        "feet": list(range(2, min(4, num_bodies))),
        "knee": list(range(4, min(6, num_bodies))),
        "elbow": list(range(6, min(8, num_bodies))),
        "head": [num_bodies - 1] if num_bodies > 0 else [],
    }


def _group_mean(per_body: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return 0.0
    valid_indices = [idx for idx in indices if 0 <= idx < per_body.shape[0]]
    if not valid_indices:
        return 0.0
    return float(per_body[valid_indices].mean().item())


def _benchmark(args_cli: argparse.Namespace) -> tuple[str, str]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    _configure_env_cfg(env_cfg, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped_env = env.unwrapped

    policy, normalizer = _load_policy(args_cli.checkpoint, env_cfg, runner_cfg, device)

    motion_lib = getattr(unwrapped_env, "_motion_lib", None)
    if motion_lib is None:
        raise RuntimeError("Environment motion library is not initialized.")

    metric_order = [
        "error_tracking_joint_dof",
        "error_tracking_joint_vel",
        "error_tracking_root_translation",
        "error_tracking_root_rotation",
        "error_tracking_root_vel",
        "error_tracking_keybody_pos",
        "error_feet_slip",
        "error_tracking_root_ang_vel",
    ]
    metric_totals = {name: 0.0 for name in metric_order}
    group_totals = {name: 0.0 for name in ("hand", "feet", "knee", "elbow", "head")}
    total_steps = 0

    try:
        _ = env.reset()
        num_motions = int(motion_lib.num_motions())
        for motion_idx in range(num_motions):
            obs = _reset_to_motion(unwrapped_env, motion_idx)
            motion_id = torch.tensor([motion_idx], device=unwrapped_env.device, dtype=torch.long)
            motion_length = float(motion_lib.get_motion_length(motion_id)[0].item())
            max_steps = max(1, int(motion_length / float(unwrapped_env.step_dt)))

            for _step in range(max_steps):
                with torch.no_grad():
                    policy_input = normalizer.normalize(obs) if normalizer is not None else obs
                    actions = policy.act_inference(policy_input)

                step_out = env.step(actions)
                if len(step_out) == 5:
                    next_obs, _, terminated, truncated, _ = step_out
                    dones = terminated | truncated
                else:
                    next_obs, _, dones, _ = step_out
                obs = _policy_obs(next_obs)

                metric_totals["error_tracking_joint_dof"] += float(unwrapped_env._error_tracking_joint_dof().mean().item())
                metric_totals["error_tracking_joint_vel"] += float(unwrapped_env._error_tracking_joint_vel().mean().item())
                metric_totals["error_tracking_root_translation"] += float(
                    unwrapped_env._error_tracking_root_translation().mean().item()
                )
                metric_totals["error_tracking_root_rotation"] += float(unwrapped_env._error_tracking_root_rotation().mean().item())
                metric_totals["error_tracking_root_vel"] += float(unwrapped_env._error_tracking_root_vel().mean().item())
                metric_totals["error_tracking_root_ang_vel"] += float(unwrapped_env._error_tracking_root_ang_vel().mean().item())
                keybody_scalar, keybody_per_body = unwrapped_env._error_tracking_keybody_pos()
                metric_totals["error_tracking_keybody_pos"] += float(keybody_scalar.mean().item())
                metric_totals["error_feet_slip"] += float(unwrapped_env._error_feet_slip().mean().item())

                per_body_mean = keybody_per_body.mean(dim=0)
                groups = _keybody_groups(int(per_body_mean.shape[0]))
                for group_name, indices in groups.items():
                    group_totals[group_name] += _group_mean(per_body_mean, indices)

                total_steps += 1
                if bool(torch.any(dones).item()):
                    break
    finally:
        env.close()

    if total_steps == 0:
        raise RuntimeError("No benchmark steps were collected.")

    avg_errors = {name: metric_totals[name] / total_steps for name in metric_order}
    avg_groups = {name: group_totals[name] / total_steps for name in group_totals}
    total_error = sum(avg_errors.values())

    lines = [f"total_error: {total_error:.4f}"]
    lines.extend(f"avg {name}: {avg_errors[name]:.4f}" for name in metric_order)
    lines.extend(
        [
            f"avg error_tracking_keybody_pos_hand: {avg_groups['hand']:.4f}",
            f"avg error_tracking_keybody_pos_feet: {avg_groups['feet']:.4f}",
            f"avg error_tracking_keybody_pos_knee: {avg_groups['knee']:.4f}",
            f"avg error_tracking_keybody_pos_elbow: {avg_groups['elbow']:.4f}",
            f"avg error_tracking_keybody_pos_head: {avg_groups['head']:.4f}",
        ]
    )

    print("Average Tracking Errors:")
    for name in metric_order:
        print(f"  {name}: {avg_errors[name]:.4f}")
    print("\nPer-Body Keybody Errors:")
    print(f"  hand: {avg_groups['hand']:.4f}")
    print(f"  feet: {avg_groups['feet']:.4f}")
    print(f"  knee: {avg_groups['knee']:.4f}")
    print(f"  elbow: {avg_groups['elbow']:.4f}")
    print(f"  head: {avg_groups['head']:.4f}")
    print(f"\ntotal_error: {total_error:.4f}")

    Path("benchmark_results").mkdir(exist_ok=True)
    output_path = Path("benchmark_results") / f"{args_cli.task}-{Path(args_cli.checkpoint).stem}.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved benchmark report to: {output_path}")

    return str(output_path), "\n".join(lines)


def main() -> int:
    if args.checkpoint is None:
        parser.error("the following arguments are required: --checkpoint")
    _benchmark(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
