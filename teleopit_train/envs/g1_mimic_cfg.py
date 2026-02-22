"""Flattened G1 Mimic environment configuration for Isaac Lab.

This file consolidates the 6-layer IsaacGym config inheritance chain
(BaseTaskCfg → LeggedRobotCfg → HumanoidCfg → HumanoidCharCfg →
 HumanoidMimicCfg → G1MimicCfg) into a single @configclass that
inherits from Isaac Lab's DirectRLEnvCfg.

All numerical values are preserved exactly from the original G1MimicCfg
(with overrides applied in inheritance order).
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUntypedClassDecorator=false, reportUntypedBaseClass=false, reportUnusedImport=false

import os
from dataclasses import field

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


# ---------------------------------------------------------------------------
# Sub-configclasses for domain-specific parameter groups
# (These have no Isaac Lab equivalent and are consumed by the env directly.)
# ---------------------------------------------------------------------------

@configclass
class G1InitStateCfg:
    """Initial state of the robot."""
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # xyzw
    lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    default_joint_angles: dict[str, float] = field(default_factory=lambda: {
        'left_hip_pitch_joint': -0.2,
        'left_hip_roll_joint': 0.0,
        'left_hip_yaw_joint': 0.0,
        'left_knee_joint': 0.4,
        'left_ankle_pitch_joint': -0.2,
        'left_ankle_roll_joint': 0.0,
        'right_hip_pitch_joint': -0.2,
        'right_hip_roll_joint': 0.0,
        'right_hip_yaw_joint': 0.0,
        'right_knee_joint': 0.4,
        'right_ankle_pitch_joint': -0.2,
        'right_ankle_roll_joint': 0.0,
        'waist_yaw_joint': 0.0,
        'waist_roll_joint': 0.0,
        'waist_pitch_joint': 0.0,
        'left_shoulder_pitch_joint': 0.0,
        'left_shoulder_roll_joint': 0.4,
        'left_shoulder_yaw_joint': 0.0,
        'left_elbow_joint': 1.2,
        'left_wrist_roll_joint': 0.0,
        'left_wrist_pitch_joint': 0.0,
        'left_wrist_yaw_joint': 0.0,
        'right_shoulder_pitch_joint': 0.0,
        'right_shoulder_roll_joint': -0.4,
        'right_shoulder_yaw_joint': 0.0,
        'right_elbow_joint': 1.2,
        'right_wrist_roll_joint': 0.0,
        'right_wrist_pitch_joint': 0.0,
        'right_wrist_yaw_joint': 0.0,
    })


@configclass
class G1ControlCfg:
    """PD control parameters."""
    control_type: str = 'P'
    stiffness: dict[str, float] = field(default_factory=lambda: {
        'hip_yaw': 100.0,
        'hip_roll': 100.0,
        'hip_pitch': 100.0,
        'knee': 150.0,
        'ankle': 40.0,
        'waist': 150.0,
        'shoulder': 40.0,
        'elbow': 40.0,
        'wrist': 40.0,
    })
    damping: dict[str, float] = field(default_factory=lambda: {
        'hip_yaw': 2.0,
        'hip_roll': 2.0,
        'hip_pitch': 2.0,
        'knee': 4.0,
        'ankle': 2.0,
        'waist': 4.0,
        'shoulder': 5.0,
        'elbow': 5.0,
        'wrist': 5.0,
    })
    action_scale: float = 0.5
    decimation: int = 10


@configclass
class G1AssetCfg:
    """Robot asset parameters."""
    file: str = ""
    xml_file: str = ""
    foot_name: str = 'ankle_roll_link'
    torso_name: str = 'pelvis'
    chest_name: str = 'imu_in_torso'
    thigh_name: str = 'hip'
    shank_name: str = 'knee'
    waist_name: list[str] = field(default_factory=lambda: ['torso_link', 'waist_roll_link', 'waist_yaw_link'])
    upper_arm_name: str = 'shoulder_roll_link'
    lower_arm_name: str = 'elbow_link'
    hand_name: str = 'hand'
    feet_bodies: list[str] = field(default_factory=lambda: ['left_ankle_roll_link', 'right_ankle_roll_link'])
    n_lower_body_dofs: int = 12
    penalize_contacts_on: list[str] = field(default_factory=lambda: ["shoulder", "elbow", "hip", "knee"])
    terminate_after_contacts_on: list[str] = field(default_factory=lambda: ['torso_link'])
    disable_gravity: bool = False
    collapse_fixed_joints: bool = False
    fix_base_link: bool = False
    default_dof_drive_mode: int = 3
    self_collisions: int = 0
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = False
    density: float = 0.001
    angular_damping: float = 0.0
    linear_damping: float = 0.0
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0
    thickness: float = 0.01
    dof_armature: list[float] = field(default_factory=lambda: (
        [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2
        + [0.0103] * 3
        + [0.003597] * 14
    ))


@configclass
class G1TerrainCfg:
    """Terrain parameters."""
    mesh_type: str = 'trimesh'
    hf2mesh_method: str = 'grid'
    max_error: float = 0.1
    max_error_camera: float = 2.0
    y_range: list[float] = field(default_factory=lambda: [-0.4, 0.4])
    edge_width_thresh: float = 0.05
    horizontal_scale: float = 0.1  # G1 override
    horizontal_scale_camera: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 5.0
    height: list[float] = field(default_factory=lambda: [0.0, 0.0])  # G1 override
    simplify_grid: bool = False
    gap_size: list[float] = field(default_factory=lambda: [0.02, 0.1])
    stepping_stone_distance: list[float] = field(default_factory=lambda: [0.02, 0.08])
    downsampled_scale: float = 0.075
    curriculum: bool = False
    all_vertical: bool = False
    no_flat: bool = True
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    measure_heights: bool = False
    measured_points_x: list[float] = field(default_factory=lambda: [
        -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2
    ])
    measured_points_y: list[float] = field(default_factory=lambda: [
        -0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75
    ])
    measure_horizontal_noise: float = 0.0
    selected: bool = False
    terrain_kwargs: dict[str, object] | None = None
    max_init_terrain_level: int = 5
    terrain_length: float = 18.0
    terrain_width: float = 4.0
    num_rows: int = 10
    num_cols: int = 40
    terrain_dict: dict[str, float] = field(default_factory=lambda: {
        "smooth slope": 0.0,
        "rough slope up": 0.0,
        "rough slope down": 0.0,
        "rough stairs up": 0.0,
        "rough stairs down": 0.0,
        "discrete": 0.0,
        "stepping stones": 0.0,
        "gaps": 0.0,
        "smooth flat": 0.0,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.0,
        "large stairs up": 0.0,
        "large stairs down": 0.0,
        "parkour": 0.0,
        "parkour_hurdle": 0.0,
        "parkour_flat": 0.05,
        "parkour_step": 0.0,
        "parkour_gap": 0.0,
        "demo": 0.0,
    })
    terrain_proportions: list[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0,
    ])
    slope_treshold: float = 1.5
    origin_zero_z: bool = True
    num_goals: int = 8


@configclass
class G1RewardScalesCfg:
    """Reward function scales."""
    tracking_joint_dof: float = 0.6
    tracking_joint_vel: float = 0.2
    tracking_root_translation: float = 0.6
    tracking_root_rotation: float = 0.6
    tracking_root_vel: float = 1.0
    tracking_keybody_pos: float = 2.0
    feet_slip: float = -0.1
    feet_contact_forces: float = -5e-4
    feet_stumble: float = -1.25
    dof_pos_limits: float = -5.0
    dof_torque_limits: float = -1.0
    dof_vel: float = -1e-4
    dof_acc: float = -5e-8
    action_rate: float = -0.01
    feet_air_time: float = 5.0
    ang_vel_xy: float = -0.01
    ankle_dof_acc: float = -5e-8 * 2  # = -1e-7
    ankle_dof_vel: float = -1e-4 * 2  # = -2e-4


@configclass
class G1RewardsCfg:
    """Reward parameters."""
    regularization_names: list[str] = field(default_factory=lambda: [
        "feet_stumble",
        "feet_contact_forces",
        "lin_vel_z",
        "ang_vel_xy",
        "orientation",
        "dof_pos_limits",
        "dof_torque_limits",
        "collision",
        "torque_penalty",
        "thigh_torque_roll_yaw",
        "thigh_roll_yaw_acc",
        "dof_acc",
        "dof_vel",
        "action_rate",
    ])
    regularization_scale: float = 1.0
    regularization_scale_range: list[float] = field(default_factory=lambda: [0.8, 2.0])
    regularization_scale_curriculum: bool = False
    regularization_scale_gamma: float = 0.0001
    scales: G1RewardScalesCfg = G1RewardScalesCfg()
    only_positive_rewards: bool = False
    clip_rewards: bool = False
    tracking_sigma: float = 0.2
    tracking_sigma_ang: float = 0.125
    soft_dof_pos_limit: float = 0.95
    soft_dof_vel_limit: float = 0.95
    soft_torque_limit: float = 0.95
    max_contact_force: float = 350.0  # G1 override
    feet_height_target: float = 0.2
    feet_air_time_target: float = 0.5
    cycle_time: float = 0.64
    termination_height: float = 0.5
    termination_roll: float = 1.5  # G1 override
    termination_pitch: float = 1.5  # G1 override
    num_lower_body: int = 0
    target_feet_height: float = 0.07  # G1 override
    min_dist: float = 0.1
    max_dist: float = 0.4  # G1 override
    max_knee_dist: float = 0.4  # G1 override
    torque_safety_limit: float = 0.9  # G1 override
    root_height_diff_threshold: float = 0.3  # G1 override


@configclass
class G1DomainRandCfg:
    """Domain randomization parameters."""
    domain_rand_general: bool = True
    randomize_gravity: bool = True
    gravity_rand_interval_s: float = 4.0  # G1 override
    gravity_range: tuple[float, float] = (-0.1, 0.1)
    randomize_friction: bool = True
    friction_range: list[float] = field(default_factory=lambda: [0.1, 2.0])  # G1 override
    randomize_base_mass: bool = True
    added_mass_range: list[float] = field(default_factory=lambda: [-3.0, 3.0])  # G1 override
    randomize_base_com: bool = True
    added_com_range: list[float] = field(default_factory=lambda: [-0.05, 0.05])  # G1 override
    push_robots: bool = True
    push_interval_s: float = 4.0  # G1 override
    max_push_vel_xy: float = 1.0
    push_end_effector: bool = True
    push_end_effector_interval_s: float = 2.0  # G1 override
    max_push_force_end_effector: float = 20.0  # G1 override
    randomize_motor: bool = True
    motor_strength_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    action_delay: bool = True
    action_buf_len: int = 8


@configclass
class G1ObsScalesCfg:
    """Observation normalization scales."""
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    imu: float = 0.5


@configclass
class G1NormalizationCfg:
    """Normalization parameters."""
    obs_scales: G1ObsScalesCfg = G1ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 5.0  # G1 override


@configclass
class G1NoiseScalesCfg:
    """Noise scales for observations."""
    dof_pos: float = 0.01
    dof_vel: float = 0.1  # G1 override
    lin_vel: float = 0.1
    ang_vel: float = 0.1  # G1 override
    gravity: float = 0.05
    imu: float = 0.1  # G1 override


@configclass
class G1NoiseCfg:
    """Noise parameters."""
    add_noise: bool = True  # G1 override
    noise_level: float = 1.0
    noise_increasing_steps: int = 3000  # G1 override
    noise_scales: G1NoiseScalesCfg = G1NoiseScalesCfg()


@configclass
class G1CommandsCfg:
    """Command parameters."""
    curriculum: bool = False
    num_commands: int = 3  # HumanoidChar override
    resampling_time: float = 3.0  # HumanoidChar override
    ang_vel_clip: float = 0.1  # HumanoidChar override
    lin_vel_clip: float = 0.1  # HumanoidChar override
    stand_prob: float = 0.3  # HumanoidChar override

    class ranges:
        lin_vel_x: list[float] = field(default_factory=lambda: [-0.3, 1.5])
        lin_vel_y: list[float] = field(default_factory=lambda: [-0.3, 0.3])
        ang_vel_yaw: list[float] = field(default_factory=lambda: [-0.6, 0.6])


@configclass
class G1CommandRangesCfg:
    """Command ranges."""
    lin_vel_x: list[float] = field(default_factory=lambda: [-0.3, 1.5])
    lin_vel_y: list[float] = field(default_factory=lambda: [-0.3, 0.3])
    ang_vel_yaw: list[float] = field(default_factory=lambda: [-0.6, 0.6])


@configclass
class G1ViewerCfg:
    """Viewer camera settings."""
    ref_env: int = 0
    pos: list[float] = field(default_factory=lambda: [10.0, 0.0, 6.0])
    lookat: list[float] = field(default_factory=lambda: [11.0, 5.0, 3.0])


@configclass
class G1MotionCfg:
    """Motion reference parameters."""
    motion_curriculum: bool = True  # G1 override
    motion_curriculum_gamma: float = 0.01  # G1 override
    key_bodies: list[str] = field(default_factory=lambda: [
        "left_rubber_hand", "right_rubber_hand",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_knee_link", "right_knee_link",
        "left_elbow_link", "right_elbow_link",
        "head_mocap",
    ])
    upper_key_bodies: list[str] = field(default_factory=list)
    motion_file: str = "../../../../motion_data/LAFAN1_g1_gmr/dance1_subject2.pkl"
    height_offset: float = 0.0
    reset_consec_frames: int = 30  # G1 override
    sample_ratio: float = 1.0
    motion_decompose: bool = False
    motion_smooth: bool = True
    use_adaptive_pose_termination: bool = False
    # Motion Domain Randomization
    motion_dr_enabled: bool = False
    root_position_noise: list[float] = field(default_factory=lambda: [0.01, 0.05])
    root_orientation_noise: list[float] = field(default_factory=lambda: [0.1, 0.2])
    root_velocity_noise: list[float] = field(default_factory=lambda: [0.05, 0.1])
    joint_position_noise: list[float] = field(default_factory=lambda: [0.05, 0.1])
    motion_dr_resampling: bool = True
    # Error Aware Sampling
    use_error_aware_sampling: bool = False
    error_sampling_power: float = 5.0
    error_sampling_threshold: float = 0.15


@configclass
class G1EvaluationsCfg:
    """Evaluation tracking flags."""
    tracking_joint_dof: bool = True
    tracking_joint_vel: bool = True
    tracking_root_translation: bool = True
    tracking_root_rotation: bool = True
    tracking_root_vel: bool = True
    tracking_root_ang_vel: bool = True
    tracking_keybody_pos: bool = True
    tracking_root_pose_delta_local: bool = True
    tracking_root_rotation_delta_local: bool = True


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

# Resolve USD asset path relative to this file's package
_TELEOPIT_TRAIN_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
_USD_ASSET_PATH = os.path.join(_TELEOPIT_TRAIN_DIR, "assets", "g1", "usd", "g1_29dof.usd")


@configclass
class G1MimicEnvCfg(DirectRLEnvCfg):
    """Flattened G1 Mimic environment configuration.

    Consolidates the 6-layer IsaacGym config chain into a single
    @configclass inheriting from Isaac Lab's DirectRLEnvCfg.
    """

    # ── DirectRLEnvCfg required fields ──────────────────────────────────
    decimation: int = 10  # from G1MimicCfg.control.decimation
    episode_length_s: float = 10.0  # from G1MimicCfg.env.episode_length_s

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,  # G1 override
        render_interval=10,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.5,
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**24,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
    )

    # ── Observation / Action spaces ─────────────────────────────────────
    # Computed from G1MimicCfg.env:
    #   num_actions = 29
    #   n_mimic_obs = 3*4 + 29 = 41
    #   tar_motion_steps_priv has 20 entries
    #   n_proprio = 20*41 + 3 + 2 + 3*29 = 912
    #   n_priv_latent = 4 + 1 + 2*29 = 63
    #   n_priv = 0, extra_critic_obs = 3, history_len = 10
    #   num_observations = 912 + 63 + 10*912 + 0 + 3 = 10098
    num_actions: int = 29
    num_observations: int = 10098
    observation_space: int = 10098
    action_space: int = 29
    state_space: int | None = None

    # ── Env-level parameters (from G1MimicCfg.env) ─────────────────────
    num_envs: int = 4096
    n_proprio: int = 912
    n_priv_mimic_obs: int = 9120
    n_priv_latent: int = 63
    n_priv: int = 0
    n_mimic_obs: int = 41
    extra_critic_obs: int = 3
    history_len: int = 10
    tar_motion_steps_priv: list[int] = field(default_factory=lambda: [
        1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    ])
    tar_motion_steps: list[int] = field(default_factory=lambda: [1])
    env_spacing: float = 3.0
    send_timeouts: bool = True

    # Early termination (from HumanoidMimicCfg + G1 overrides)
    enable_early_termination: bool = True
    pose_termination: bool = True  # G1 override (was False in HumanoidMimicCfg)
    pose_termination_dist: float = 0.7  # G1 override (was 1.0)
    root_tracking_termination_dist: float = 0.8  # G1 override (was 0.7)

    # Observation flags
    enable_tar_obs: bool = False
    rand_reset: bool = True
    ref_char_offset: float = 0.0
    global_obs: bool = False  # G1 override (was True in HumanoidMimicCfg)
    track_root: bool = False  # G1 override (was True in HumanoidMimicCfg)

    # DOF error weights (G1-specific, 29 values)
    dof_err_w: list[float] = field(default_factory=lambda: [
        1.0, 1.0, 1.0, 1.0, 0.1, 0.1,  # Left Leg
        1.0, 1.0, 1.0, 1.0, 0.1, 0.1,  # Right Leg
        1.0, 1.0, 1.0,                  # waist yaw, roll, pitch
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Left Arm
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Right Arm
    ])

    # Start randomization
    randomize_start_pos: bool = True
    randomize_start_yaw: bool = False
    rand_yaw_range: float = 1.2

    # History / contact
    history_encoding: bool = True
    contact_buf_len: int = 10  # G1 override (was 100 in LeggedRobot)

    # Misc env flags
    normalize_obs: bool = True
    record_video: bool = False
    teleop_mode: bool = False
    num_privileged_obs: int | None = None

    # ── Sub-configs ─────────────────────────────────────────────────────
    init_state: G1InitStateCfg = G1InitStateCfg()
    control: G1ControlCfg = G1ControlCfg()
    asset: G1AssetCfg = G1AssetCfg()
    terrain: G1TerrainCfg = G1TerrainCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    domain_rand: G1DomainRandCfg = G1DomainRandCfg()
    normalization: G1NormalizationCfg = G1NormalizationCfg()
    noise: G1NoiseCfg = G1NoiseCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    motion: G1MotionCfg = G1MotionCfg()
    evaluations: G1EvaluationsCfg = G1EvaluationsCfg()

    # ── Terrain Importer (Isaac Lab terrain system) ──────────────────────
    # Replaces the custom Terrain class from IsaacGym.
    # Uses TerrainImporter + TerrainGeneratorCfg for trimesh terrain generation.
    # The terrain_generator sub_terrains can be customized to match the original
    # terrain proportions. Default: flat terrain with roughness.
    terrain_importer: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        visual_material=None,  # Disable visual material to avoid RTX crash in headless mode
        terrain_generator=TerrainGeneratorCfg(
            size=(18.0, 4.0),  # terrain_length x terrain_width from G1TerrainCfg
            border_width=5.0,  # border_size from G1TerrainCfg
            num_rows=10,  # from G1TerrainCfg
            num_cols=40,  # from G1TerrainCfg
            horizontal_scale=0.1,  # from G1TerrainCfg
            vertical_scale=0.005,  # from G1TerrainCfg
            slope_threshold=1.5,  # slope_treshold from G1TerrainCfg
            curriculum=False,  # from G1TerrainCfg
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=1.0),
            },
        ),
        max_init_terrain_level=5,  # from G1TerrainCfg
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,  # from G1TerrainCfg
            dynamic_friction=1.0,  # from G1TerrainCfg
            restitution=0.0,  # from G1TerrainCfg
        ),
        debug_vis=False,
    )

    # USD asset path (from T3 conversion)
    usd_asset_path: str = _USD_ASSET_PATH

    def __post_init__(self):
        """Set derived values and asset path."""
        # Set the URDF asset path on the asset sub-config
        self.asset.file = self.usd_asset_path

        # Ensure observation/action spaces match
        self.observation_space = self.num_observations
        self.action_space = self.num_actions


@configclass
class G1MimicActorCriticCfg(RslRlPpoActorCriticCfg):
    """Extended actor-critic config with motion encoder fields for ActorCriticMimic."""
    motion_latent_dim: int = 64
    num_motion_steps: int = 10


@configclass
class G1MimicAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Extended algorithm config with custom PPO fields for TWIST2 runner."""
    dagger_update_freq: int = 20
    normalizer_update_iterations: int = 3000


@configclass
class G1MimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 30000
    save_interval: int = 500
    experiment_name: str = "g1_mimic"

    policy_class_name: str = "ActorCriticMimic"
    algorithm_class_name: str = "PPO"
    runner_class_name: str = "OnPolicyRunnerMimic"

    logger: str = "wandb"
    wandb_project: str = "teleopit_isaaclab"

    policy: G1MimicActorCriticCfg = G1MimicActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        motion_latent_dim=64,
        num_motion_steps=10,
    )

    algorithm: G1MimicAlgorithmCfg = G1MimicAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        dagger_update_freq=20,
        normalizer_update_iterations=3000,
    )
