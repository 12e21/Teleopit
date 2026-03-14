from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def validate_policy_path(cfg: Any, script_name: str) -> Path:
    controller_cfg = getattr(cfg, "controller", None)
    policy_raw = ""
    if controller_cfg is not None and hasattr(controller_cfg, "get"):
        policy_raw = str(controller_cfg.get("policy_path", "")).strip()
    elif controller_cfg is not None:
        policy_raw = str(getattr(controller_cfg, "policy_path", "")).strip()

    if not policy_raw:
        raise ValueError(
            "controller.policy_path is required and must point to ONNX exported from "
            f"train_mimic checkpoint.\nExample: python scripts/{script_name} "
            "controller.policy_path=policy.onnx"
        )

    policy_path = Path(policy_raw).expanduser()
    if not policy_path.is_absolute():
        policy_path = (Path.cwd() / policy_path).resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"ONNX policy file not found: {policy_path}")
    return policy_path


def add_unitree_sdk_submodule(repo_root: Path) -> None:
    sdk_path = repo_root / "third_party" / "unitree_sdk2_python"
    if sdk_path.exists():
        sys.path.insert(0, str(sdk_path))
