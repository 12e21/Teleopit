"""G1 sim2real control entry point with Pico4 full-body tracking — standing/mocap dual mode."""

from __future__ import annotations

import sys
from pathlib import Path

# Add Unitree SDK2 Python to sys.path (git submodule in third_party/)
_SDK_PATH = Path(__file__).resolve().parent.parent / "third_party" / "unitree_sdk2_python"
if _SDK_PATH.exists():
    sys.path.insert(0, str(_SDK_PATH))

import hydra
from omegaconf import DictConfig

from teleopit.sim2real.controller import Sim2RealController


def _validate_policy_path(cfg: DictConfig, script_name: str) -> None:
    policy_path = str(cfg.controller.get("policy_path", "")).strip()
    if not policy_path:
        raise ValueError(
            "controller.policy_path is required and must point to ONNX exported from train_mimic checkpoint.\n"
            f"Example: python scripts/{script_name} controller.policy_path=policy.onnx"
        )
    resolved = Path(policy_path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"ONNX policy file not found: {resolved}")


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="pico4_sim2real")
def main(cfg: DictConfig) -> None:
    _validate_policy_path(cfg, "run_pico4_sim2real.py")
    controller = Sim2RealController(cfg)
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
