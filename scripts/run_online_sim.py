"""Online sim2sim entry point — receives real-time BVH data over UDP."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="online")
def main(cfg: DictConfig) -> None:
    pipeline = TeleopPipeline(cfg)
    num_steps = int(cfg.get("num_steps", 0))
    udp_port = cfg.input.get("udp_port", 1118)
    print(f"Waiting for UDP data on port {udp_port}...")
    try:
        result = pipeline.run(num_steps=num_steps)
        print(result)
    finally:
        if hasattr(pipeline.input_provider, "close"):
            pipeline.input_provider.close()


if __name__ == "__main__":
    main()
