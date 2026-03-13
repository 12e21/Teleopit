"""Online sim2sim entry point — receives real-time Pico4 full-body tracking data."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline


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


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="pico4_sim")
def main(cfg: DictConfig) -> None:
    _validate_policy_path(cfg, "run_pico4_sim.py")
    pipeline = TeleopPipeline(cfg)
    num_steps = int(cfg.get("num_steps", 0))
    print("Waiting for Pico4 body tracking data...")
    try:
        result = pipeline.run(num_steps=num_steps)
        print(result)
    finally:
        if hasattr(pipeline.input_provider, "close"):
            pipeline.input_provider.close()


if __name__ == "__main__":
    main()
