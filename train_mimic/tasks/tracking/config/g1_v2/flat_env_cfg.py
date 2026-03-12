"""G1 flat-terrain tracking environment configuration (v2).

v2 is behaviorally identical to v0 except that motion sampling is forced to
uniform. This preserves the original reward, termination, and episode settings
while avoiding adaptive resampling of hard or invalid clips.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from train_mimic.tasks.tracking.config.g1.flat_env_cfg import (
    make_g1_flat_tracking_env_cfg,
)

DEFAULT_TASK_V2 = "Tracking-Flat-G1-v2"


def make_g1_flat_tracking_env_cfg_v2(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 flat terrain tracking configuration (v2)."""
    cfg = make_g1_flat_tracking_env_cfg(
        has_state_estimation=has_state_estimation,
        play=play,
    )
    if not play:
        cfg.commands["motion"].sampling_mode = "uniform"
    return cfg
