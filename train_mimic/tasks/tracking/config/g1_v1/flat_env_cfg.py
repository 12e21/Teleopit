"""G1 flat-terrain tracking environment configuration (v1 – general motion).

Optimised for large-scale general motion training:
- Uniform sampling (avoids adaptive death spiral on diverse datasets)
- Relaxed early termination thresholds
- Shorter episode length for faster curriculum signal
- Tuned adaptive params for future switch back to adaptive mode
"""

from __future__ import annotations

from mjlab.asset_zoo.robots import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from train_mimic.tasks.tracking.mdp import MotionCommandCfg
from train_mimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

DEFAULT_TASK_V1 = "Tracking-Flat-G1-v1"
DEFAULT_TRAIN_MOTION_FILE = "data/datasets/builds/twist2_full/train.npz"


def make_g1_flat_tracking_env_cfg_v1(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 flat terrain tracking configuration (v1 – general motion)."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.body_names = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    cfg.viewer.body_name = "torso_link"

    # ------------------------------------------------------------------
    # Project-specific overrides
    # ------------------------------------------------------------------
    motion_cmd.motion_file = DEFAULT_TRAIN_MOTION_FILE
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    # ------------------------------------------------------------------
    # v1: General motion optimisations
    # ------------------------------------------------------------------

    # Sampling: uniform to avoid adaptive death spiral on diverse datasets.
    motion_cmd.sampling_mode = "uniform"
    # Pre-tuned adaptive params for future switch back to adaptive mode.
    motion_cmd.adaptive_kernel_size = 64
    motion_cmd.adaptive_uniform_ratio = 0.4
    motion_cmd.adaptive_alpha = 0.01

    # Relaxed early termination thresholds.
    cfg.terminations["anchor_pos"].params["threshold"] = 0.5
    cfg.terminations["anchor_ori"].params["threshold"] = 1.2
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.5

    # Shorter episodes for faster curriculum signal.
    cfg.episode_length_s = 5.0

    # Modify observations if we don't have state estimation.
    if not has_state_estimation:
        new_actor_terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=new_actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

    return cfg
