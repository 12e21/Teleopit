#!/usr/bin/env python3

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAny=false, reportExplicitAny=false, reportUnusedImport=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportArgumentType=false, reportMissingParameterType=false

import argparse
import asyncio
import importlib
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train G1 mimic policy with custom rsl_rl runner.")
    parser.add_argument("--task", type=str, default="Isaac-G1-Mimic-v0")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--motion_file", type=str, default=None,
                        help="Override motion file (pkl, yaml, or directory of pkl files)")
    # Intercept --help before SimulationApp swallows it
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        parser.print_help()
        raise SystemExit(0)
    return parser.parse_args()


# Save original asyncio.run BEFORE Isaac Sim patches it.
# Isaac Sim's omni.kit.async_engine monkey-patches asyncio.run() globally,
# which conflicts with wandb's background AsyncioManager thread and causes a segfault.
# We restore the original after SimulationApp creation so that wandb (and other libraries)
# use the standard asyncio.run() in their threads.
_original_asyncio_run = asyncio.run

# Bootstrap Omniverse Kit runtime — MUST instantiate SimulationApp before any omni / Isaac Lab imports
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": "--headless" in sys.argv})

# Restore original asyncio.run — Isaac Sim uses its own internal event loop for
# simulation and does not need the global asyncio.run to be patched.
asyncio.run = _original_asyncio_run

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import train_mimic.envs  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from train_mimic.rsl_rl.runners.on_policy_runner_mimic import OnPolicyRunnerMimic as OnPolicyRunner  # noqa: E402


def _to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Unsupported cfg type: {type(cfg)}")


def _resolve_policy_class_name(policy_class_name: str) -> str:
    if ":" not in policy_class_name:
        return policy_class_name
    module_path, class_name = policy_class_name.split(":", maxsplit=1)
    module = importlib.import_module(module_path)
    globals()[class_name] = getattr(module, class_name)
    return class_name


def _build_custom_runner_cfg(agent_cfg: Any, cli_args: argparse.Namespace) -> dict[str, Any]:
    cfg = _to_dict(agent_cfg)

    policy_cfg = dict(cfg["policy"])
    policy_cfg.pop("class_name", None)

    algorithm_cfg = dict(cfg["algorithm"])
    algorithm_cfg.pop("class_name", None)
    algorithm_cfg.setdefault("dagger_update_freq", 20)
    algorithm_cfg.setdefault("normalizer_update_iterations", 3000)

    policy_class_name = cfg.get("policy_class_name", "ActorCriticMimic")
    policy_class_name = _resolve_policy_class_name(policy_class_name)

    runner_cfg = {
        "policy_class_name": policy_class_name,
        "algorithm_class_name": cfg.get("algorithm_class_name", "PPO"),
        "runner_class_name": cfg.get("runner_class_name", "OnPolicyRunnerMimic"),
        "num_steps_per_env": cfg["num_steps_per_env"],
        "max_iterations": cfg["max_iterations"],
        "save_interval": cfg["save_interval"],
        "experiment_name": cfg["experiment_name"],
        "run_name": cfg.get("run_name", ""),
        "logger": "console",
        "wandb_project": cfg.get("wandb_project", "teleopit_isaaclab"),
        "seed": cfg.get("seed", 42),
    }

    runner_cfg["seed"] = cli_args.seed
    if cli_args.max_iterations is not None:
        runner_cfg["max_iterations"] = cli_args.max_iterations
    if cli_args.experiment_name is not None:
        runner_cfg["experiment_name"] = cli_args.experiment_name
    if cli_args.wandb_project is not None:
        runner_cfg["wandb_project"] = cli_args.wandb_project
        runner_cfg["logger"] = "wandb"

    return {
        "runner": runner_cfg,
        "policy": policy_cfg,
        "algorithm": algorithm_cfg,
    }


class _LegacyCfg:
    def __init__(self, env_cfg: Any):
        self.env = env_cfg
        self.rewards = env_cfg.rewards


class _LegacyRslRlEnvAdapter:
    def __init__(self, env: RslRlVecEnvWrapper):
        self._env = env
        self.cfg = _LegacyCfg(env.cfg)
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_obs = int(self.cfg.env.num_observations)
        self.num_privileged_obs = int(self.cfg.env.num_observations)
        self.obs_buf = self._policy_obs(self._env.get_observations())

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._env.episode_length_buf = value

    @property
    def episode_length(self):
        return getattr(self._env.unwrapped, "episode_length", None)

    @property
    def mean_motion_difficulty(self):
        return getattr(self._env.unwrapped, "mean_motion_difficulty", 0.0)

    @property
    def global_counter(self):
        return getattr(self._env.unwrapped, "global_counter", 0)

    @global_counter.setter
    def global_counter(self, value):
        setattr(self._env.unwrapped, "global_counter", value)

    @property
    def total_env_steps_counter(self):
        return getattr(self._env.unwrapped, "total_env_steps_counter", 0)

    @total_env_steps_counter.setter
    def total_env_steps_counter(self, value):
        setattr(self._env.unwrapped, "total_env_steps_counter", value)

    def _policy_obs(self, obs_dict: Any) -> torch.Tensor:
        if isinstance(obs_dict, dict):
            return obs_dict["policy"]
        if hasattr(obs_dict, "keys") and "policy" in obs_dict.keys():
            return obs_dict["policy"]
        raise KeyError("Expected observation group 'policy' in wrapped environment output")

    def get_observations(self) -> torch.Tensor:
        self.obs_buf = self._policy_obs(self._env.get_observations())
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor:
        self.obs_buf = self._policy_obs(self._env.get_observations())
        return self.obs_buf

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, dones, infos = self._env.step(actions)
        obs = self._policy_obs(obs_dict)
        self.obs_buf = obs
        return obs, obs, rewards, dones, infos

    def close(self):
        return self._env.close()


def main() -> None:
    args = parse_args()

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    env_cfg.seed = args.seed
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.motion_file is not None:
        env_cfg.motion.motion_file = args.motion_file

    train_cfg = _build_custom_runner_cfg(agent_cfg, args)

    render_mode = None if args.headless else "rgb_array"
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    env = RslRlVecEnvWrapper(env)
    env = _LegacyRslRlEnvAdapter(env)

    log_root = os.path.join("logs", "rsl_rl", train_cfg["runner"]["experiment_name"])
    log_root = os.path.abspath(log_root)
    os.makedirs(log_root, exist_ok=True)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if train_cfg["runner"]["run_name"]:
        run_name = f"{run_name}_{train_cfg['runner']['run_name']}"
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    runner.learn(
        num_learning_iterations=train_cfg["runner"]["max_iterations"],
        init_at_random_ep_len=True,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
