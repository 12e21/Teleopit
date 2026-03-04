"""G1 tracking task registration via gymnasium."""

import gymnasium as gym

from .flat_env_cfg import G1FlatTrackingEnvCfg

gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatTrackingEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1TrackingPPORunnerCfg",
    },
)
