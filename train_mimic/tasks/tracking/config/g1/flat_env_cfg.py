"""G1 flat-terrain tracking environment configuration.

Extends mjlab's built-in G1FlatEnvCfg with any custom overrides.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg


DEFAULT_TRAIN_MOTION_FILE = "data/datasets/builds/twist2_full/train.npz"


@dataclass
class G1FlatTrackingEnvCfg(G1FlatEnvCfg):
    """G1 on flat terrain for whole-body motion tracking.

    Inherits from mjlab's G1FlatEnvCfg. Override __post_init__ to
    customize motion file path, tracking bodies, reward weights, etc.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Default motion file: must be a single NPZ (not a directory).
        # Prefer dataset build outputs over legacy twist2_retarget_npz assets.
        self.commands.motion.motion_file = DEFAULT_TRAIN_MOTION_FILE

        # Increase the global simulation constraint buffer as well. mjlab's
        # tracking base config sets sim.njmax=250, which is what triggers the
        # runtime `nefc overflow` warnings during training when many contacts
        # are active at once.
        self.sim.njmax = 500
        if self.sim.nconmax is not None and self.sim.nconmax < 150_000:
            self.sim.nconmax = 150_000
