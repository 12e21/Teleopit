from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="default")
def main(cfg: DictConfig) -> None:
    pipeline = TeleopPipeline(cfg)
    num_steps = int(cfg.get("num_steps", 1000))
    record = bool(cfg.get("record", False))
    result = pipeline.run(num_steps=num_steps, record=record)
    print(result)


if __name__ == "__main__":
    main()
