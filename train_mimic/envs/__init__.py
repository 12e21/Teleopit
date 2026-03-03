"""Isaac Lab environment configurations and task registration for train_mimic."""

# pyright: reportUnknownMemberType=false

import gymnasium as gym

from .g1_mimic_cfg import G1MimicEnvCfg, G1MimicPPORunnerCfg
from .g1_mimic_env import G1MimicEnv

gym.register(
    id="Isaac-G1-Mimic-v0",
    entry_point="train_mimic.envs.g1_mimic_env:G1MimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "train_mimic.envs.g1_mimic_cfg:G1MimicEnvCfg",
        "rsl_rl_cfg_entry_point": "train_mimic.envs.g1_mimic_cfg:G1MimicPPORunnerCfg",
    },
)

__all__ = ["G1MimicEnvCfg", "G1MimicPPORunnerCfg", "G1MimicEnv"]
