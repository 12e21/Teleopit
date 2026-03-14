"""Internal tracking task profiles.

Only the uniform no-state-estimation profile is registered publicly. The
adaptive variants stay here as implementation references for future sampling
work, but they are not exposed as task IDs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingTaskProfile:
    name: str
    experiment_name: str
    sampling_mode: str
    episode_length_s: float = 10.0
    anchor_pos_threshold: float = 0.4
    anchor_ori_threshold: float = 1.0
    ee_body_pos_threshold: float = 0.4
    adaptive_kernel_size: int | None = None
    adaptive_uniform_ratio: float | None = None
    adaptive_alpha: float | None = None


OFFICIAL_UNIFORM_PROFILE = TrackingTaskProfile(
    name="official_uniform_no_state_est",
    experiment_name="g1_tracking",
    sampling_mode="uniform",
)

ADAPTIVE_REFERENCE_PROFILE = TrackingTaskProfile(
    name="adaptive_reference",
    experiment_name="g1_tracking_adaptive_reference",
    sampling_mode="adaptive",
)

GENERAL_MOTION_REFERENCE_PROFILE = TrackingTaskProfile(
    name="general_motion_reference",
    experiment_name="g1_tracking_general_motion_reference",
    sampling_mode="uniform",
    episode_length_s=5.0,
    anchor_pos_threshold=0.5,
    anchor_ori_threshold=1.2,
    ee_body_pos_threshold=0.5,
    adaptive_kernel_size=64,
    adaptive_uniform_ratio=0.4,
    adaptive_alpha=0.01,
)
