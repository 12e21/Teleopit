"""Shared application helpers for train/eval/export entry points."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE, OFFICIAL_TASK

DEFAULT_TASK = OFFICIAL_TASK
LEGACY_TASK_ALIASES = {
    "Tracking-Flat-G1-v2-NoStateEst": DEFAULT_TASK,
}


def resolve_task_name(task_name: str) -> str:
    """Normalize a task name to the canonical registered task."""
    normalized = LEGACY_TASK_ALIASES.get(task_name, task_name)
    if normalized != task_name:
        print(f"[WARN] Task '{task_name}' is deprecated. Use '{normalized}' instead.")
    return normalized


def validate_motion_file(motion_file: str) -> None:
    if Path(motion_file).is_file():
        return
    raise FileNotFoundError(
        f"Motion file not found: {motion_file}. Provide --motion_file explicitly or build a dataset "
        f"such as {DEFAULT_TRAIN_MOTION_FILE}."
    )


def validate_checkpoint_path(checkpoint_path: str) -> None:
    if Path(checkpoint_path).is_file():
        return
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def import_training_stack() -> tuple[Any, ...]:
    import torch

    import mjlab.tasks  # noqa: F401 -- populates mjlab built-in tasks
    import train_mimic.tasks  # noqa: F401 -- registers our custom tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    return (
        torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
        configure_torch_backends,
    )


def load_task_components(
    task_name: str,
    *,
    play: bool = False,
    load_env_cfg: Any | None = None,
    load_rl_cfg: Any | None = None,
    load_runner_cls: Any | None = None,
) -> tuple[str, Any, Any, Any]:
    if load_env_cfg is None or load_rl_cfg is None or load_runner_cls is None:
        (
            _torch,
            _ManagerBasedRlEnv,
            _RslRlVecEnvWrapper,
            _MjlabOnPolicyRunner,
            load_env_cfg,
            load_rl_cfg,
            load_runner_cls,
            _configure_torch_backends,
        ) = import_training_stack()
    canonical_task = resolve_task_name(task_name)
    env_cfg = load_env_cfg(canonical_task, play=play)
    agent_cfg = load_rl_cfg(canonical_task)
    runner_cls = load_runner_cls(canonical_task)
    return canonical_task, env_cfg, agent_cfg, runner_cls


def build_runner_cfg_dict(agent_cfg: Any, *, force_tensorboard: bool = False) -> dict[str, Any]:
    agent_dict = asdict(agent_cfg)
    if force_tensorboard:
        agent_dict["logger"] = "tensorboard"
    return agent_dict


def resolve_device(requested_device: str | None, torch_module: Any) -> str:
    if requested_device is not None:
        return requested_device
    return "cuda:0" if torch_module.cuda.is_available() else "cpu"
