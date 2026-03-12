"""PPO runner configuration for G1 motion tracking (v2)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from train_mimic.tasks.tracking.config.g1.rl_cfg import (
    make_g1_tracking_ppo_runner_cfg,
)


def make_g1_tracking_ppo_runner_cfg_v2() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for G1 tracking (v2)."""
    cfg = make_g1_tracking_ppo_runner_cfg()
    cfg.experiment_name = "g1_tracking_v2"
    return cfg
