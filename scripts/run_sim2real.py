"""G1 sim2real control entry point — standing/mocap dual mode."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from teleopit.runtime.cli import add_unitree_sdk_submodule, validate_policy_path

add_unitree_sdk_submodule(Path(__file__).resolve().parent.parent)
from teleopit.sim2real.controller import Sim2RealController


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="sim2real")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "run_sim2real.py")
    controller = Sim2RealController(cfg)
    if cfg.input.get("provider") == "pico4":
        print("Waiting for Pico4 body tracking data...")
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
