from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUntypedBaseClass=false, reportUnknownArgumentType=false

import os
import re
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporter

from teleopit_train.pose.utils.motion_lib_pkl import MotionLib
from teleopit_train.pose.utils.torch_utils import euler_from_quaternion
from teleopit_train.pose.utils import torch_utils as pose_torch_utils
from teleopit_train.utils.math_utils import quat_rotate_inverse, quat_from_euler_xyz, torch_rand_float

from .g1_mimic_cfg import G1MimicEnvCfg


class G1MimicEnv(DirectRLEnv):
    cfg: G1MimicEnvCfg

    def __init__(self, cfg: G1MimicEnvCfg, render_mode: str | None = None, **kwargs: Any):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._init_buffers()
        self._prepare_evaluation_functions()
        self._contact_indices_ready = False
        if self.robot.num_joints != 29:
            raise ValueError(f"Expected 29 joints, got {self.robot.num_joints}")

    def _setup_scene(self):
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cfg.usd_asset_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=self.cfg.asset.disable_gravity,
                    linear_damping=self.cfg.asset.linear_damping,
                    angular_damping=self.cfg.asset.angular_damping,
                    max_linear_velocity=self.cfg.asset.max_linear_velocity,
                    max_angular_velocity=self.cfg.asset.max_angular_velocity,
                    max_depenetration_velocity=10.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(self.cfg.init_state.pos),
                joint_pos=dict(self.cfg.init_state.default_joint_angles),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
            },
        )
        self.robot = Articulation(robot_cfg)

        # ── Terrain: use TerrainImporter instead of ground plane ──────────
        self.terrain_importer = TerrainImporter(self.cfg.terrain_importer)
        # Expose terrain_levels / terrain_types from the importer for curriculum
        self.terrain_levels = getattr(self.terrain_importer, "terrain_levels", None)
        self.terrain_types = getattr(self.terrain_importer, "terrain_types", None)
        self.max_terrain_level = getattr(self.terrain_importer, "max_terrain_level", 0)

        self.contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                history_length=1,
                update_period=0.0,
            )
        )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def __post_init__(self):
        sensor_body_names = self.contact_sensor.body_names
        fallback_idx = 0 if len(sensor_body_names) > 0 else -1

        feet_indices: list[int] = []
        for name in self.cfg.asset.feet_bodies:
            try:
                indices, _ = self.contact_sensor.find_bodies([f".*{name}.*"])
                feet_indices.extend(int(i) for i in indices)
            except ValueError:
                continue
        if len(feet_indices) == 0 and fallback_idx >= 0:
            feet_indices = [fallback_idx]
        while len(feet_indices) < len(self.cfg.asset.feet_bodies):
            feet_indices.append(feet_indices[-1])
        self.feet_indices = feet_indices

        penalised_contact_indices: list[int] = []
        for name in self.cfg.asset.penalize_contacts_on:
            try:
                indices, _ = self.contact_sensor.find_bodies([f".*{name}.*"])
            except ValueError:
                continue
            penalised_contact_indices.extend(int(i) for i in indices if int(i) not in penalised_contact_indices)
        if len(penalised_contact_indices) == 0 and fallback_idx >= 0:
            penalised_contact_indices = [fallback_idx]
        self.penalised_contact_indices = penalised_contact_indices

        termination_contact_indices: list[int] = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            try:
                indices, _ = self.contact_sensor.find_bodies([f".*{name}.*"])
            except ValueError:
                continue
            termination_contact_indices.extend(int(i) for i in indices if int(i) not in termination_contact_indices)
        if len(termination_contact_indices) == 0 and fallback_idx >= 0:
            termination_contact_indices = [fallback_idx]
        self.termination_contact_indices = termination_contact_indices

    def step(self, actions: torch.Tensor):
        return super().step(actions)

    def _init_buffers(self):
        self.root_states = self.robot.data.root_state_w
        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel
        self.rigid_body_states = self.robot.data.body_state_w
        self.contact_forces = self.contact_sensor.data.net_forces_w
        self.default_dof_pos = self.robot.data.default_joint_pos.clone()
        self.default_root_state = self.robot.data.default_root_state.clone()
        self.joint_effort_limits = self.robot.data.joint_effort_limits.clone()

        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.delayed_actions = torch.zeros_like(self.actions)
        self.action_history_buf = torch.zeros(
            (self.num_envs, self.cfg.domain_rand.action_buf_len, self.cfg.num_actions),
            device=self.device,
        )
        self.action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.contact_buf = torch.zeros(
            (self.num_envs, self.cfg.contact_buf_len, len(self.cfg.asset.feet_bodies)), device=self.device
        )
        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.cfg.asset.feet_bodies)), dtype=torch.bool, device=self.device
        )

        self.friction_coeffs_tensor = torch.ones((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.mass_params_tensor = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.motor_strength = torch.ones((2, self.num_envs, self.cfg.num_actions), dtype=torch.float, device=self.device)
        self.p_gains, self.d_gains = self._build_pd_gains()

        self._gravity_rand_interval_steps = max(1, int(self.cfg.domain_rand.gravity_rand_interval_s / self.step_dt))
        self._push_interval_steps = max(1, int(self.cfg.domain_rand.push_interval_s / self.step_dt))
        self._gravity_steps_since_last = 0
        self._torso_body_idx = self._resolve_torso_body_idx()

        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.roll = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.pitch = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.yaw = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.cfg.history_len, self.cfg.n_proprio), dtype=torch.float, device=self.device
        )
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.total_env_steps_counter = 0
        self._last_post_step_counter = -1

        self._init_motion_lib()
        self._init_motion_buffers()

        # ── Reward-related buffers ──────────────────────────────────────────
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.torques = torch.zeros((self.num_envs, self.cfg.num_actions), dtype=torch.float, device=self.device)
        self.torque_limits = self.joint_effort_limits.clone()
        self.projected_gravity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.gravity_vec = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.gravity_vec[:, 2] = -1.0  # normalized gravity direction
        self.feet_air_time = torch.zeros((self.num_envs, len(self.cfg.asset.feet_bodies)), dtype=torch.float, device=self.device)
        self.contact_filt = torch.zeros((self.num_envs, len(self.cfg.asset.feet_bodies)), dtype=torch.bool, device=self.device)
        self.episode_length = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        # Key body IDs use robot body indices (for rigid_body_states)
        self._key_body_ids = self._resolve_body_indices(self.cfg.motion.key_bodies)

        # DOF error weights for tracking rewards
        self._dof_err_w = torch.tensor(self.cfg.dof_err_w, device=self.device, dtype=torch.float)

        # Reward scales dict (from config dataclass -> plain dict)
        self.reward_scales = {}
        scales_obj = self.cfg.rewards.scales
        for attr_name in dir(scales_obj):
            if not attr_name.startswith("_"):
                val = getattr(scales_obj, attr_name)
                if isinstance(val, (int, float)):
                    self.reward_scales[attr_name] = float(val)

        self._prepare_reward_function()

    def _init_motion_lib(self):
        motion_file = self._resolve_motion_file(self.cfg.motion.motion_file)
        self._motion_file = motion_file
        self._motion_lib = None
        self.motion_names = []
        if motion_file is not None:
            self._motion_lib = MotionLib(
                motion_file=motion_file,
                device=self.device,
                sample_ratio=self.cfg.motion.sample_ratio,
                motion_decompose=self.cfg.motion.motion_decompose,
                motion_smooth=self.cfg.motion.motion_smooth,
            )
            self.motion_names = self._motion_lib.get_motion_names()

    def _resolve_motion_file(self, motion_file: str) -> str | None:
        if os.path.isabs(motion_file) and os.path.exists(motion_file):
            return motion_file

        env_dir = os.path.dirname(os.path.abspath(__file__))
        teleopit_root = os.path.normpath(os.path.join(env_dir, "..", ".."))
        candidates = [
            motion_file,
            os.path.normpath(os.path.join(env_dir, motion_file)),
            os.path.normpath(os.path.join(teleopit_root, motion_file)),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _pre_physics_step(self, actions: torch.Tensor):
        clipped_actions = torch.clamp(actions, -10.0, 10.0)
        self.last_actions[:] = self.actions
        self.actions[:] = clipped_actions

        self.action_history_buf = torch.roll(self.action_history_buf, shifts=-1, dims=1)
        self.action_history_buf[:, -1, :] = self.actions

        if self.cfg.domain_rand.action_delay:
            delayed_indices = torch.clamp(self.cfg.domain_rand.action_buf_len - 1 - self.action_delay_steps, min=0)
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.delayed_actions[:] = self.action_history_buf[env_ids, delayed_indices]
        else:
            self.delayed_actions[:] = self.actions

        if self.cfg.domain_rand.randomize_gravity:
            self._gravity_steps_since_last += 1
            if self._gravity_steps_since_last >= self._gravity_rand_interval_steps:
                self._randomize_gravity()
                self._gravity_steps_since_last = 0

        if self.cfg.domain_rand.push_robots and self.common_step_counter % self._push_interval_steps == 0:
            self._push_robots()

    def _apply_action(self):
        act = self.delayed_actions if self.cfg.domain_rand.action_delay else self.actions
        target_dof_pos = self.default_dof_pos + self.cfg.control.action_scale * act
        pos_err = target_dof_pos - self.dof_pos

        if self.cfg.domain_rand.randomize_motor:
            torques = (
                self.motor_strength[0] * self.p_gains * pos_err
                - self.motor_strength[1] * self.d_gains * self.dof_vel
            )
        else:
            torques = self.p_gains * pos_err - self.d_gains * self.dof_vel

        torques = torch.clip(torques, -self.joint_effort_limits, self.joint_effort_limits)
        self.torques[:] = torques
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._post_physics_step()

        mimic_obs = self._get_mimic_obs()
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)

        obs_buf = torch.cat(
            (
                mimic_obs,
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
                imu_obs,
                (self.dof_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos,
                self.dof_vel * self.cfg.normalization.obs_scales.dof_vel,
                self.action_history_buf[:, -1],
            ),
            dim=-1,
        )

        if self.cfg.noise.add_noise:
            if self.sim.has_gui() or self.sim.has_rtx_sensors():
                noise_factor = 1.0
            else:
                noise_factor = min(
                    self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),
                    1.0,
                )
            obs_buf = obs_buf + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * noise_factor

        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat(
                (
                    self.mass_params_tensor,
                    self.friction_coeffs_tensor,
                    self.motor_strength[0] - 1.0,
                    self.motor_strength[1] - 1.0,
                    self.base_lin_vel,
                ),
                dim=-1,
            )
        else:
            priv_latent = torch.cat(
                (
                    torch.zeros((self.num_envs, self.cfg.n_priv_latent), dtype=torch.float, device=self.device),
                    self.base_lin_vel,
                ),
                dim=-1,
            )

        policy_obs = torch.cat((obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)), dim=-1)
        if policy_obs.shape[-1] != self.cfg.num_observations:
            raise RuntimeError(
                f"Observation dim mismatch: got {policy_obs.shape[-1]}, expected {self.cfg.num_observations}"
            )

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            obs_buf.unsqueeze(1).expand(-1, self.cfg.history_len, -1),
            torch.cat((self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)), dim=1),
        )
        # Sanitize NaN values that can arise from invalid physics states
        policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return {"policy": policy_obs}

    def _post_physics_step(self):
        if self.common_step_counter == self._last_post_step_counter:
            return

        if self.common_step_counter > self._last_post_step_counter:
            self.total_env_steps_counter += self.common_step_counter - max(self._last_post_step_counter, 0)
        self._last_post_step_counter = int(self.common_step_counter)

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        euler = euler_from_quaternion(self.base_quat)
        self.roll[:] = euler[:, 0]
        self.pitch[:] = euler[:, 1]
        self.yaw[:] = euler[:, 2]
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def _get_mimic_obs(self) -> torch.Tensor:
        num_steps = self._tar_motion_steps_priv.shape[0]
        if num_steps <= 0:
            raise RuntimeError("Invalid number of target motion steps for mimic observations")

        motion_lib = self._motion_lib
        if motion_lib is None:
            return torch.zeros(
                (self.num_envs, self.cfg.n_mimic_obs * num_steps),
                dtype=torch.float,
                device=self.device,
            )

        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_motion_steps_priv * self.step_dt + motion_times
        tiled_motion_ids = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, *_ = motion_lib.calc_motion_frame(
            tiled_motion_ids.reshape(-1), obs_motion_times.reshape(-1)
        )

        euler = euler_from_quaternion(root_rot)
        roll = euler[:, 0].reshape(self.num_envs, num_steps, 1)
        pitch = euler[:, 1].reshape(self.num_envs, num_steps, 1)
        yaw = euler[:, 2].reshape(self.num_envs, num_steps, 1)

        if not self.cfg.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)

        mimic_obs_buf = torch.cat(
            (
                root_pos.reshape(self.num_envs, num_steps, 3),
                roll,
                pitch,
                yaw,
                root_vel.reshape(self.num_envs, num_steps, 3),
                root_ang_vel.reshape(self.num_envs, num_steps, 3),
                dof_pos.reshape(self.num_envs, num_steps, self.cfg.num_actions),
            ),
            dim=-1,
        )
        return mimic_obs_buf.reshape(self.num_envs, -1)

    def _get_noise_scale_vec(self) -> torch.Tensor:
        noise_scale_vec = torch.zeros((1, self.cfg.n_proprio), dtype=torch.float, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec

        noise_start_dim = self.cfg.n_mimic_obs * len(self.cfg.tar_motion_steps_priv)
        ang_vel_dim = 3
        imu_dim = 2
        dof_dim = self.cfg.num_actions

        noise_scale_vec[:, noise_start_dim : noise_start_dim + ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, noise_start_dim + ang_vel_dim : noise_start_dim + ang_vel_dim + imu_dim] = (
            self.cfg.noise.noise_scales.imu
        )
        dof_pos_start = noise_start_dim + ang_vel_dim + imu_dim
        dof_vel_start = dof_pos_start + dof_dim
        action_start = dof_vel_start + dof_dim
        noise_scale_vec[:, dof_pos_start : dof_pos_start + dof_dim] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, dof_vel_start : dof_vel_start + dof_dim] = self.cfg.noise.noise_scales.dof_vel
        noise_scale_vec[:, action_start : action_start + dof_dim] = self.cfg.noise.noise_scales.dof_pos
        return noise_scale_vec

    def _get_rewards(self) -> torch.Tensor:
        if not self._contact_indices_ready:
            self.__post_init__()
            self._contact_indices_ready = True

        self.rew_buf = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]()
            if isinstance(rew, (int, float)):
                rew = torch.full((self.num_envs,), rew, dtype=torch.float, device=self.device)
            if name in self.cfg.rewards.regularization_names:
                self.rew_buf += rew * self.reward_scales[name] * self.cfg.rewards.regularization_scale
            else:
                self.rew_buf += rew * self.reward_scales[name]
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)

        # Update last-step buffers for next reward computation
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # Regularization scale curriculum
        if self.cfg.rewards.regularization_scale_curriculum:
            mean_ep_len = torch.mean(self.episode_length.float()).item()
            if mean_ep_len > 420.0:
                self.cfg.rewards.regularization_scale *= (1.0 + self.cfg.rewards.regularization_scale_gamma)
            elif mean_ep_len < 50.0:
                self.cfg.rewards.regularization_scale *= (1.0 - self.cfg.rewards.regularization_scale_gamma)
            lo, hi = self.cfg.rewards.regularization_scale_range
            self.cfg.rewards.regularization_scale = max(min(self.cfg.rewards.regularization_scale, hi), lo)

        for i in range(len(self.eval_functions)):
            name = self.eval_names[i]
            error = self.eval_functions[i]()
            if isinstance(error, tuple):
                error = error[0]
            self.episode_means[name] += (-self.episode_means[name] + error) / (self.episode_length_buf + 1.0)

        return self.rew_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if self._motion_lib is not None:
            self._update_ref_motion()

        if self.cfg.enable_early_termination:
            root_height_cutoff = self.root_states[:, 2] < self.cfg.rewards.termination_height
            died |= root_height_cutoff

            if self.cfg.pose_termination and self._motion_lib is not None:
                pose_error = torch.norm(self.dof_pos - self._ref_dof_pos, dim=-1)
                died |= pose_error > self.cfg.pose_termination_dist

            if self.cfg.track_root and self._motion_lib is not None:
                root_track_error = torch.norm(self.root_states[:, :2] - self._ref_root_pos[:, :2], dim=-1)
                died |= root_track_error > self.cfg.root_tracking_termination_dist

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.numel() == self.num_envs:
            resolved_env_ids = self.robot._ALL_INDICES
        else:
            resolved_env_ids = env_ids
        env_ids = torch.as_tensor(resolved_env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        super()._reset_idx(env_ids)

        dof_pos = self.default_dof_pos[env_ids].clone()
        dof_vel = torch.zeros_like(dof_pos)
        root_state = self.default_root_state[env_ids].clone()

        motion_lib = self._motion_lib
        if motion_lib is not None:
            motion_ids = motion_lib.sample_motions(env_ids.numel())
            if self.cfg.rand_reset:
                motion_times = motion_lib.sample_time(motion_ids)
            else:
                motion_times = torch.zeros_like(motion_ids, dtype=torch.float)

            self._motion_ids[env_ids] = motion_ids
            self._motion_time_offsets[env_ids] = motion_times

            root_pos, root_rot, root_vel, root_ang_vel, ref_dof_pos, ref_dof_vel, *_ = motion_lib.calc_motion_frame(
                motion_ids, motion_times
            )
            root_pos[:, 2] += self.cfg.motion.height_offset
            root_pos[:, :2] += self._env_origins()[env_ids, :2]

            self._ref_root_pos[env_ids] = root_pos
            self._ref_root_rot[env_ids] = root_rot
            self._ref_dof_pos[env_ids] = ref_dof_pos
            self._ref_root_vel[env_ids] = root_vel
            self._ref_root_ang_vel[env_ids] = root_ang_vel

            dof_pos = ref_dof_pos
            dof_vel = ref_dof_vel * 0.8
            root_state[:, :3] = root_pos
            root_state[:, 3:7] = root_rot
            root_state[:, 7:10] = root_vel * 0.8
            root_state[:, 10:13] = root_ang_vel * 0.8

        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        self._process_rigid_shape_props(env_ids)
        self._process_rigid_body_props(env_ids)
        self._randomize_motor_strength(env_ids)
        self._randomize_action_delay(env_ids)
        if self.cfg.domain_rand.randomize_gravity:
            self._randomize_gravity()
            self._gravity_steps_since_last = 0

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.delayed_actions[env_ids] = 0.0
        self.action_history_buf[env_ids] = 0.0
        self.obs_history_buf[env_ids] = 0.0
        self.contact_buf[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.last_dof_vel[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0
        # Populate self.extras["episode"] for wandb logging
        if len(env_ids) > 0:
            self.extras["episode"] = {}
            
            # Collect reward episode sums (averaged by episode length)
            for key in self.episode_sums.keys():
                # Compute mean reward per step for each env being reset
                episode_lengths = torch.clamp(self.episode_length_buf[env_ids], min=1.0)
                mean_values = self.episode_sums[key][env_ids] / episode_lengths
                
                # Store in extras (runner expects dict of tensors)
                self.extras["episode"][key] = mean_values
            
            # Collect error episode means (prefix with "error_")
            for name in self.episode_means.keys():
                self.extras["episode"][f"error_{name}"] = self.episode_means[name][env_ids].clone()
            
            # Reset episode_means for the reset envs
            for name in self.episode_means.keys():
                self.episode_means[name][env_ids] = 0.0
        self.episode_length[env_ids] = self.episode_length_buf[env_ids].float()

        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

    def _init_motion_buffers(self):
        self._motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._ref_root_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._ref_root_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self._ref_root_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._ref_root_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._ref_dof_pos = torch.zeros_like(self.dof_pos)
        self._ref_dof_vel = torch.zeros_like(self.dof_vel)
        num_bodies = self.rigid_body_states.shape[1] if len(self.rigid_body_states.shape) > 2 else 1
        self._ref_body_pos = torch.zeros((self.num_envs, num_bodies, 3), dtype=torch.float, device=self.device)
        self._tar_motion_steps_priv = torch.tensor(
            self.cfg.tar_motion_steps_priv, dtype=torch.float, device=self.device
        )
        self._tar_motion_steps = torch.tensor(self.cfg.tar_motion_steps, dtype=torch.float, device=self.device)

    def _get_motion_times(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.episode_length_buf * self.step_dt + self._motion_time_offsets
        return self.episode_length_buf[env_ids] * self.step_dt + self._motion_time_offsets[env_ids]

    def _update_ref_motion(self):
        motion_lib = self._motion_lib
        if motion_lib is None:
            return
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, *extra = motion_lib.calc_motion_frame(
            self._motion_ids, motion_times
        )
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self._env_origins()[:, :2]

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        if len(extra) > 0 and extra[0] is not None:
            body_pos = extra[0]
            self._ref_body_pos[:] = _convert_to_global_root_body_pos(root_pos, root_rot, body_pos)

    def _build_pd_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        p_gains = torch.zeros((self.num_envs, self.cfg.num_actions), dtype=torch.float, device=self.device)
        d_gains = torch.zeros_like(p_gains)
        for joint_id, joint_name in enumerate(self.robot.joint_names):
            kp = self._lookup_gain(joint_name, self.cfg.control.stiffness)
            kd = self._lookup_gain(joint_name, self.cfg.control.damping)
            p_gains[:, joint_id] = kp
            d_gains[:, joint_id] = kd
        return p_gains, d_gains

    @staticmethod
    def _lookup_gain(joint_name: str, gain_cfg: dict[str, float]) -> float:
        for key, value in gain_cfg.items():
            if key in joint_name:
                return float(value)
        return 0.0

    def _resolve_torso_body_idx(self) -> int:
        for idx, body_name in enumerate(self.robot.body_names):
            if self.cfg.asset.torso_name in body_name:
                return idx
        return 0

    def _process_rigid_shape_props(self, env_ids: torch.Tensor):
        if not self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor[env_ids] = 1.0
            return

        fric_min, fric_max = self.cfg.domain_rand.friction_range
        friction_samples = torch_rand_float(fric_min, fric_max, (len(env_ids), 1), device=self.device)
        self.friction_coeffs_tensor[env_ids] = friction_samples

        env_ids_cpu = env_ids.to(dtype=torch.int64, device="cpu")
        materials = self.robot.root_physx_view.get_material_properties()
        friction_cpu = friction_samples.detach().cpu().repeat(1, materials.shape[1])
        materials[env_ids_cpu, :, 0] = friction_cpu
        materials[env_ids_cpu, :, 1] = friction_cpu
        self.robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _process_rigid_body_props(self, env_ids: torch.Tensor):
        self.mass_params_tensor[env_ids] = 0.0
        if not self.cfg.domain_rand.randomize_base_mass:
            return

        mass_min, mass_max = self.cfg.domain_rand.added_mass_range
        added_mass = torch_rand_float(mass_min, mass_max, (len(env_ids), 1), device=self.device)
        self.mass_params_tensor[env_ids, 0:1] = added_mass

        env_ids_cpu = env_ids.to(dtype=torch.int64, device="cpu")
        masses = self.robot.root_physx_view.get_masses()
        masses[env_ids_cpu, self._torso_body_idx] = (
            self.robot.data.default_mass[env_ids_cpu, self._torso_body_idx].detach().cpu()
            + added_mass.squeeze(-1).detach().cpu()
        )
        self.robot.root_physx_view.set_masses(masses, env_ids_cpu)

    def _randomize_motor_strength(self, env_ids: torch.Tensor):
        self.motor_strength[:, env_ids, :] = 1.0
        if not self.cfg.domain_rand.randomize_motor:
            return

        motor_min, motor_max = self.cfg.domain_rand.motor_strength_range
        kp_scale = torch_rand_float(motor_min, motor_max, (len(env_ids), 1), device=self.device)
        kd_scale = torch_rand_float(motor_min, motor_max, (len(env_ids), 1), device=self.device)
        self.motor_strength[0, env_ids, :] = kp_scale
        self.motor_strength[1, env_ids, :] = kd_scale

    def _randomize_action_delay(self, env_ids: torch.Tensor):
        self.action_delay_steps[env_ids] = 0
        if not self.cfg.domain_rand.action_delay:
            return

        delay_max = max(1, self.cfg.domain_rand.action_buf_len)
        delay = torch_rand_float(0.0, float(delay_max), (len(env_ids), 1), device=self.device)
        self.action_delay_steps[env_ids] = torch.floor(delay).long().squeeze(-1).clamp(max=delay_max - 1)

    def _randomize_gravity(self):
        if not self.cfg.domain_rand.randomize_gravity:
            return

        gravity_delta = torch_rand_float(
            self.cfg.domain_rand.gravity_range[0],
            self.cfg.domain_rand.gravity_range[1],
            (1, 3),
            device=self.device,
        ).squeeze(0)
        gravity = torch.tensor(self.cfg.sim.gravity, dtype=torch.float, device=self.device) + gravity_delta

        try:
            import carb

            physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
            physics_sim_view.set_gravity(carb.Float3(*gravity.detach().cpu().tolist()))
        except Exception:
            pass

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        root_velocity = self.root_states[:, 7:13].clone()
        root_velocity[:, :2] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.robot.write_root_velocity_to_sim(root_velocity)

    def _env_origins(self) -> torch.Tensor:
        if hasattr(self.scene, "env_origins"):
            return self.scene.env_origins
        return torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

    # ── Body index resolution helpers ───────────────────────────────────

    def _resolve_body_indices(self, body_names: list[str]) -> list[int]:
        """Resolve exact body names to indices."""
        indices = []
        for name in body_names:
            for idx, bname in enumerate(self.robot.body_names):
                if bname == name:
                    indices.append(idx)
                    break
        return indices

    def _resolve_body_indices_partial(self, name_patterns: list[str]) -> list[int]:
        """Resolve partial body name patterns to indices."""
        indices = []
        for pattern in name_patterns:
            for idx, bname in enumerate(self.robot.body_names):
                if pattern in bname and idx not in indices:
                    indices.append(idx)
        return indices

    # ── Reward infrastructure ───────────────────────────────────────────

    def _prepare_reward_function(self):
        """Prepare list of reward functions from cfg.rewards.scales.

        Mirrors the original IsaacGym _prepare_reward_function:
        - Remove zero-scale entries
        - Multiply non-zero scales by dt
        - Build (function, name) lists
        - Create episode_sums dict
        """
        # dt for this env = step_dt (physics_dt * decimation)
        dt = self.step_dt

        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= dt

        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            fn_name = "_reward_" + name
            if not hasattr(self, fn_name):
                raise AttributeError(f"Reward function {fn_name} not found on {type(self).__name__}")
            self.reward_functions.append(getattr(self, fn_name))

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _prepare_evaluation_functions(self):
        eval_cfg = self.cfg.evaluations
        self.evaluations = {}
        for attr_name in vars(eval_cfg):
            if attr_name.startswith("_"):
                continue
            value = getattr(eval_cfg, attr_name)
            if isinstance(value, bool):
                self.evaluations[attr_name] = value

        self.eval_functions = []
        self.eval_names = []
        for name, enabled in self.evaluations.items():
            if not enabled:
                continue
            fn_name = "_error_" + name
            if not hasattr(self, fn_name):
                continue
            self.eval_names.append(name)
            self.eval_functions.append(getattr(self, fn_name))

        self.episode_means = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.eval_names
        }

    def _error_tracking_joint_dof(self):
        dof_diff = self._ref_dof_pos - self.dof_pos
        return torch.mean(torch.abs(dof_diff), dim=-1)

    def _error_tracking_joint_vel(self):
        vel_diff = self._ref_dof_vel - self.dof_vel
        return torch.mean(torch.abs(vel_diff), dim=-1)

    def _error_tracking_root_translation(self):
        root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        return torch.mean(torch.abs(root_pos_diff), dim=-1)

    def _error_tracking_root_rotation(self):
        root_rot_err = pose_torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
        if root_rot_err.ndim == 1:
            return torch.abs(root_rot_err)
        return torch.mean(torch.abs(root_rot_err), dim=-1)

    def _error_tracking_root_vel(self):
        local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
        root_vel_diff = local_ref_root_vel - self.base_lin_vel
        return torch.mean(torch.abs(root_vel_diff), dim=-1)

    def _error_tracking_root_ang_vel(self):
        local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
        root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
        return torch.mean(torch.abs(root_ang_vel_diff), dim=-1)

    def _error_tracking_keybody_pos(self):
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3]
        key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        if not self.cfg.global_obs:
            base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)
            key_body_pos = _convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)

        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        if not self.cfg.global_obs:
            _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot).unbind(dim=-1)
            ref_yaw_quat = quat_from_euler_xyz(0 * ref_yaw, 0 * ref_yaw, ref_yaw)
            tar_key_body_pos = _convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)

        per_body_tensor = torch.mean(torch.abs(key_body_pos - tar_key_body_pos), dim=-1)
        scalar = torch.mean(per_body_tensor, dim=-1)
        return scalar, per_body_tensor

    def _error_tracking_feet_slip(self):
        return self._error_feet_slip()

    def _error_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        err = torch.sqrt(foot_speed_norm)
        err *= contact
        return torch.sum(err, dim=1)

    # ── Tracking reward functions (from HumanoidMimic) ──────────────────

    def _reward_tracking_joint_dof(self):
        dof_diff = self._ref_dof_pos - self.dof_pos
        dof_err = torch.sum(self._dof_err_w * dof_diff * dof_diff, dim=-1)
        return torch.exp(-0.15 * dof_err)

    def _reward_tracking_joint_vel(self):
        vel_diff = self._ref_dof_vel - self.dof_vel
        vel_err = torch.sum(self._dof_err_w * vel_diff * vel_diff, dim=-1)
        return torch.exp(-0.01 * vel_err)

    def _reward_tracking_root_translation(self):
        root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
        return torch.exp(-5.0 * root_pos_err)

    def _reward_tracking_root_rotation(self):
        root_rot_err = pose_torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
        root_rot_err = root_rot_err * root_rot_err
        return torch.exp(-5.0 * root_rot_err)

    def _reward_tracking_root_vel(self):
        if self.cfg.global_obs:
            root_vel_diff = self._ref_root_vel - self.root_states[:, 7:10]
            root_ang_vel_diff = self._ref_root_ang_vel - self.root_states[:, 10:13]
        else:
            local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
            root_vel_diff = local_ref_root_vel - self.base_lin_vel
            local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
            root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
        root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
        root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
        return torch.exp(-1.0 * (root_vel_err + 0.5 * root_ang_vel_err))

    def _reward_tracking_keybody_pos(self):
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3]
        key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)
        key_body_pos = _convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)

        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot).unbind(dim=-1)
        ref_yaw_quat = quat_from_euler_xyz(0 * ref_yaw, 0 * ref_yaw, ref_yaw)
        tar_key_body_pos = _convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)

        diff = key_body_pos - tar_key_body_pos
        err = torch.sum(diff * diff, dim=-1)
        err = torch.sum(err, dim=-1)
        return torch.exp(-10.0 * err)

    # ── Base reward functions (from Humanoid) ───────────────────────────

    def _reward_alive(self):
        return 1.0

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_dof_pos_limits(self):
        lower = self.robot.data.soft_joint_pos_limits[..., 0]
        upper = self.robot.data.soft_joint_pos_limits[..., 1]
        out_of_limits = -(self.dof_pos - lower).clip(max=0.0)
        out_of_limits += (self.dof_pos - upper).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_torque_limits(self):
        out_of_limits = torch.sum(
            (torch.abs(self.torques) / self.torque_limits - self.cfg.rewards.soft_torque_limit).clip(min=0),
            dim=1,
        )
        return out_of_limits

    def _reward_feet_stumble(self):
        rew = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )
        return rew.float()

    def _reward_feet_contact_forces(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < self.cfg.rewards.max_contact_force] = 0
        rew[rew > self.cfg.rewards.max_contact_force] -= self.cfg.rewards.max_contact_force
        return rew

    def _reward_torque_penalty(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.step_dt), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_base_acc(self):
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.step_dt), dim=1)

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.step_dt
        tgt_air_time = self.cfg.rewards.feet_air_time_target
        air_time = (self.feet_air_time - tgt_air_time) * first_contact
        air_time = air_time.clamp(max=0.0)
        self.feet_air_time *= ~self.contact_filt
        rew_airtime = air_time.sum(dim=1)
        rew_airtime *= torch.norm(self._ref_root_vel[:, :2], dim=1) > 0.05
        return rew_airtime

    # ── G1-specific reward functions (from g1_mimic.py) ─────────────────

    def _reward_ankle_dof_acc(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, ankle_dof_idx] / self.step_dt), dim=1)

    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)

    # ── Regularization-only rewards (in regularization_names, may have zero scale) ──

    def _reward_thigh_torque_roll_yaw(self):
        # Thigh roll/yaw joint indices: hip_roll(1,7), hip_yaw(0,6)
        thigh_idx = [0, 1, 6, 7]
        return torch.sum(torch.square(self.torques[:, thigh_idx]), dim=1)

    def _reward_thigh_roll_yaw_acc(self):
        thigh_idx = [0, 1, 6, 7]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, thigh_idx] / self.step_dt), dim=1)


# ── Module-level helpers (for JIT compatibility) ────────────────────────

def _convert_to_global_root_body_pos(root_pos, root_rot, body_pos):
    """Convert local body positions to global frame."""
    root_rot_expand = root_rot.unsqueeze(-2).repeat(1, body_pos.shape[1], 1)
    flat_rot = root_rot_expand.reshape(-1, 4)
    flat_pos = body_pos.reshape(-1, 3)
    flat_global = pose_torch_utils.quat_rotate(flat_rot, flat_pos)
    global_pos = flat_global.reshape(body_pos.shape)
    global_pos = global_pos + root_pos.unsqueeze(1)
    return global_pos


def _convert_to_local_root_body_pos(root_rot, body_pos):
    """Convert body positions to local root frame."""
    root_inv_rot = pose_torch_utils.quat_conjugate(root_rot)
    root_rot_expand = root_inv_rot.unsqueeze(-2).repeat(1, body_pos.shape[1], 1)
    flat_rot = root_rot_expand.reshape(-1, 4)
    flat_pos = body_pos.reshape(-1, 3)
    flat_local = pose_torch_utils.quat_rotate(flat_rot, flat_pos)
    return flat_local.reshape(body_pos.shape)
