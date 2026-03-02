"""G1 sim2real control entry point — gamepad/mocap dual mode."""

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


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="sim2real")
def main(cfg: DictConfig) -> None:
    controller = Sim2RealController(cfg)
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
