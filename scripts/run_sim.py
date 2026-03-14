from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline
from teleopit.runtime.cli import validate_policy_path


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="default")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "run_sim.py")
    pipeline = TeleopPipeline(cfg)
    num_steps = int(cfg.get("num_steps", 0))
    record = bool(cfg.get("record", False))
    if cfg.input.get("provider") == "udp_bvh":
        udp_port = cfg.input.get("udp_port", 1118)
        print(f"Waiting for UDP data on port {udp_port}...")
    elif cfg.input.get("provider") == "pico4":
        print("Waiting for Pico4 body tracking data...")
    result = pipeline.run(num_steps=num_steps, record=record)
    print(result)


if __name__ == "__main__":
    main()
