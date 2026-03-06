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
import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Single-node multi-GPU launch helper. Example: --gpu_ids 0 1 2 3. "
            "When multiple IDs are provided, this script relaunches itself via torchrun."
        ),
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for internal torchrun launch when using --gpu_ids (default: 29500)",
    )
    parser.add_argument("--video", action="store_true",
                        help="Record periodic videos during training")
    parser.add_argument("--video_interval", type=int, default=2000,
                        help="Record a video every N iterations (default: 2000)")
    parser.add_argument("--video_length", type=int, default=200,
                        help="Number of steps per video clip (default: 200)")
    return parser.parse_args(argv)


def _to_rsl_rl5_cfg(cfg: dict) -> dict:
    """Convert mjlab's RslRlOnPolicyRunnerCfg dict to rsl_rl 5.x format.

    mjlab uses a single "policy" key (ActorCritic style), but rsl_rl 5.x
    expects separate "actor" and "critic" keys with MLPModel config.
    """
    policy = cfg.pop("policy")
    if "policy" in cfg.get("obs_groups", {}):
        cfg["obs_groups"]["actor"] = cfg["obs_groups"].pop("policy")

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


def _is_distributed_env(env: dict[str, str] | None = None) -> bool:
    runtime_env = os.environ if env is None else env
    return int(runtime_env.get("WORLD_SIZE", "1")) > 1


def _should_launch_multi_gpu(args: argparse.Namespace, env: dict[str, str] | None = None) -> bool:
    gpu_ids = list(args.gpu_ids or [])
    return len(gpu_ids) > 1 and not _is_distributed_env(env)


def _validate_multi_gpu_args(args: argparse.Namespace) -> None:
    gpu_ids = list(args.gpu_ids or [])
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--gpu_ids contains duplicates: {gpu_ids}")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"--gpu_ids must be non-negative, got {gpu_ids}")
    if args.master_port <= 0:
        raise ValueError(f"--master_port must be positive, got {args.master_port}")


def _filtered_argv_for_worker(argv: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    skip_gpu_id_values = False
    skip_next_value = False
    for token in argv:
        if skip_gpu_id_values:
            if token.startswith("-"):
                skip_gpu_id_values = False
            else:
                continue
        if skip_next_value:
            skip_next_value = False
            continue

        if token == "--gpu_ids":
            skip_gpu_id_values = True
            continue
        if token.startswith("--gpu_ids="):
            continue
        if token == "--master_port":
            skip_next_value = True
            continue
        if token.startswith("--master_port="):
            continue

        filtered.append(token)
    return filtered


def _build_torchrun_command(args: argparse.Namespace, argv: Sequence[str]) -> list[str]:
    gpu_ids = list(args.gpu_ids or [])
    if len(gpu_ids) <= 1:
        raise ValueError("multi-GPU launch requires at least two gpu ids")

    worker_argv = _filtered_argv_for_worker(argv)
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        f"--master_port={args.master_port}",
        *worker_argv,
    ]


def _build_launcher_env(args: argparse.Namespace, env: dict[str, str] | None = None) -> dict[str, str]:
    runtime_env = dict(os.environ if env is None else env)
    gpu_ids = list(args.gpu_ids or [])
    runtime_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    return runtime_env


def _wait_process(proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)
    return proc.poll() is not None


def _terminate_worker_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGINT)
    if _wait_process(proc, timeout_s=5.0):
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    if _wait_process(proc, timeout_s=5.0):
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    _wait_process(proc, timeout_s=2.0)


def _resolve_device(args: argparse.Namespace, torch_module: object) -> str:
    if _is_distributed_env():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        expected_device = f"cuda:{local_rank}"
        if args.device is not None and args.device != expected_device:
            raise ValueError(
                f"Distributed worker must use device '{expected_device}' for LOCAL_RANK={local_rank}, "
                f"got '{args.device}'. Remove --device or set it to the local-rank device."
            )
        return expected_device

    if args.device is not None:
        return args.device
    return "cuda:0" if torch_module.cuda.is_available() else "cpu"


def _launch_multi_gpu(args: argparse.Namespace, argv: Sequence[str]) -> None:
    _validate_multi_gpu_args(args)
    command = _build_torchrun_command(args, argv)
    env = _build_launcher_env(args)
    print(
        f"[INFO] Launching multi-GPU training on GPUs {list(args.gpu_ids or [])} "
        f"with {args.num_envs} envs/GPU"
    )
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        return_code = proc.wait()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, stopping distributed workers...")
        _terminate_worker_group(proc)
        raise SystemExit(130)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _import_training_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    import gymnasium as gym
    import torch

    import train_mimic.tasks  # noqa: F401 -- triggers gym.register()
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
        load_cfg_from_registry,
    )
    from mjlab.utils.torch import configure_torch_backends
    from rsl_rl.runners import OnPolicyRunner

    return gym, torch, RslRlVecEnvWrapper, load_cfg_from_registry, configure_torch_backends, OnPolicyRunner


def _destroy_process_group(torch_module: Any) -> None:
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None or not distributed.is_available():
        return
    if not distributed.is_initialized():
        return
    with contextlib.suppress(Exception):
        distributed.destroy_process_group()


def _run_worker(args: argparse.Namespace) -> None:
    gym, torch, RslRlVecEnvWrapper, load_cfg_from_registry, configure_torch_backends, OnPolicyRunner = _import_training_stack()
    env: Any | None = None
    rank = os.environ.get("RANK", "0")

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        print(f"[INFO] Rank {rank} received signal {signum}, shutting down...")
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        _destroy_process_group(torch)
        raise KeyboardInterrupt

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

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

    device = _resolve_device(args, torch)

    # Log directory (defined before env creation so video path is available)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # render_mode only needed for video recording
    render_mode = "rgb_array" if args.video else None
    try:
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

        runner = OnPolicyRunner(env, _to_rsl_rl5_cfg(asdict(agent_cfg)), log_dir=log_dir, device=device)

        if args.resume is not None:
            print(f"[INFO] Resuming from: {args.resume}")
            runner.load(args.resume)

        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print(f"[INFO] Rank {rank} interrupted; exiting gracefully.")
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        _destroy_process_group(torch)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main(argv: Sequence[str] | None = None) -> None:
    cli_argv = list(sys.argv if argv is None else argv)
    parse_argv = cli_argv[1:]
    args = parse_args(parse_argv)

    if _should_launch_multi_gpu(args):
        _launch_multi_gpu(args, cli_argv)
        return

    _run_worker(args)


if __name__ == "__main__":
    main()
