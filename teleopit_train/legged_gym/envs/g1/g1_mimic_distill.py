from isaacgym.torch_utils import *

import torch

from teleopit_train.legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .g1_mimic_distill_config import G1MimicPrivCfg, G1MimicStuCfg
from teleopit_train.legged_gym.gym_utils.math import *
from teleopit_train.pose.utils import torch_utils
from teleopit_train.legged_gym.envs.base.legged_robot import euler_from_quaternion
from teleopit_train.legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos




class G1MimicDistill(HumanoidMimic):
    def __init__(self, cfg: G1MimicPrivCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        if self.obs_type == 'student':
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)
        
        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        self._ref_root_pos_delta_local[env_ids] = root_pos_delta_local
        self._ref_root_rot_delta_local[env_ids] = root_rot_delta_local
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
    
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        self._ref_root_pos_delta_local[:] = root_pos_delta_local
        self._ref_root_rot_delta_local[:] = root_rot_delta_local
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
    def _update_motion_difficulty(self, env_ids):
        if self.obs_type == 'priv':
            super()._update_motion_difficulty(env_ids)
        elif self.obs_type == 'student':
            super()._update_motion_difficulty(env_ids) # currently we use the same strategy for student
        else:
            super()._update_motion_difficulty(env_ids)

    def _get_body_indices(self):
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                  torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                                requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
    
    def _init_buffers(self):
        super()._init_buffers()
        self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_obs_single), device=self.device)
        self.privileged_obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_priv_obs_single), device=self.device)
    
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2
        
        noise_scale_vec[:, 0:ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, ang_vel_dim:ang_vel_dim+imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, ang_vel_dim+imu_dim:ang_vel_dim+imu_dim+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, ang_vel_dim+imu_dim+self.num_dof:ang_vel_dim+imu_dim+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        
        return noise_scale_vec
            
    def _get_mimic_obs(self):
        num_steps = self._tar_motion_steps_priv.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_motion_steps_priv * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)

        root_vel_local = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)
        

        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :] # local body pos
        whole_key_body_pos_global = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        whole_key_body_pos_global = whole_key_body_pos_global.reshape(self.num_envs, num_steps, -1)
        

        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
        dof_vel = dof_vel.reshape(self.num_envs, num_steps, dof_vel.shape[-1])
        root_vel_local = root_vel_local.reshape(self.num_envs, num_steps, root_vel_local.shape[-1])
        root_ang_vel_local = root_ang_vel_local.reshape(self.num_envs, num_steps, root_ang_vel_local.shape[-1])
        root_pos_delta_local = root_pos_delta_local.reshape(self.num_envs, num_steps, root_pos_delta_local.shape[-1])
        root_rot_delta_local = root_rot_delta_local.reshape(self.num_envs, num_steps, root_rot_delta_local.shape[-1])
        root_pos_distance_to_target = root_pos - self.root_states[:, 0:3].reshape(self.num_envs, 1, -1)
        
        # teacher
        priv_mimic_obs_buf = torch.cat((
            root_pos, # 3 dims
            root_pos_distance_to_target, # 3 dims
            roll, pitch, yaw, # 3 dims
            root_vel_local, # 3 dims
            root_ang_vel_local, # 3 dims
            root_pos_delta_local, # 3 dims
            root_rot_delta_local, # 3 dims
            dof_pos, # num_dof dims
            whole_key_body_pos if not self.global_obs else whole_key_body_pos_global,
        ), dim=-1) # shape: (num_envs, num_steps, 21 + num_dof + num_key_bodies * 3)
        
        
        # v0 - Modified for better observability
        mimic_obs_buf = torch.cat((
            # root position: xy velocity + z position
            root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
            root_pos[..., 2:3], # 1 dim (z position)
            # root rotation: roll/pitch + yaw angular velocity
            roll, pitch, # 2 dims (roll/pitch orientation)
            root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
            dof_pos, # num_dof dims
        ), dim=-1)[:, self._tar_motion_steps_idx_in_teacher, :] # shape: (num_envs, 1, 6 + num_dof)
            
        return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)

    def compute_observations(self):
        # imu_obs = torch.stack((self.roll, self.pitch, self.yaw), dim=1) # @YanjieZe: adding yaw would lead to global rotation tracking.
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        priv_mimic_obs, mimic_obs = self._get_mimic_obs()
        
        proprio_obs_buf = torch.cat((
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        
        if self.cfg.noise.add_noise and self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec
        else:
            proprio_obs_buf += 0.
        dof_vel_start_dim = 3 + 2 + self.dof_pos.shape[1]

        # disable ankle dof velocity
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.
        
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1) # shape: (num_envs, num_key_bodies * 3)
        
        priv_info = torch.cat((
            self.base_lin_vel, # 3 dims
            self.root_states[:, 0:3], # 3 dims
            self.root_states[:, 3:7], # 4 dims
            key_body_pos, # num_bodies * 3 dims
            self.contact_forces[:, self.feet_indices, 2] > 5., # 2 dims, foot contact
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1,
        ), dim=-1)
        
        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)
        
        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)
        
        self.privileged_obs_buf = priv_obs_buf
        # self.privileged_obs_buf = torch.cat([priv_obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
            # self.obs_buf = torch.cat([priv_obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        elif self.obs_type == 'student':
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        

        if self.cfg.env.history_len > 0:
            # Use in-place operations to avoid memory leaks
            reset_mask = (self.episode_length_buf <= 1)
            
            # For reset episodes, fill entire history with current observation
            if reset_mask.any():
                reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[reset_indices] = priv_obs_buf[reset_indices].unsqueeze(1).expand(
                    -1, self.cfg.env.history_len, -1
                )
            
            # For continuing episodes, shift history and add new observation
            continue_mask = ~reset_mask
            if continue_mask.any():
                continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                # Shift history left (remove oldest, move others)
                self.privileged_obs_history_buf[continue_indices, :-1] = self.privileged_obs_history_buf[continue_indices, 1:]
                # Add new observation at the end
                self.privileged_obs_history_buf[continue_indices, -1] = priv_obs_buf[continue_indices]
            
            if self.obs_type == 'priv':
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            elif self.obs_type == 'student':
                # Use the same in-place update pattern for regular observations
                # For reset episodes, fill entire history with current observation
                if reset_mask.any():
                    reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                    self.obs_history_buf[reset_indices] = obs_buf[reset_indices].unsqueeze(1).expand(
                        -1, self.cfg.env.history_len, -1
                    )
                
                # For continuing episodes, shift history and add new observation
                if continue_mask.any():
                    continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                    # Shift history left (remove oldest, move others)
                    self.obs_history_buf[continue_indices, :-1] = self.obs_history_buf[continue_indices, 1:]
                    # Add new observation at the end
                    self.obs_history_buf[continue_indices, -1] = obs_buf[continue_indices]


############################################################################################################
##################################### Extra Reward Functions################################################
############################################################################################################

    def _reward_waist_dof_acc(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, waist_dof_idx], dim=1)
    
    def _reward_waist_dof_vel(self):
        waist_dof_idx = [13, 14]
        return torch.sum(torch.square(self.dof_vel[:, waist_dof_idx]), dim=1)
    
    def _reward_ankle_dof_acc(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
    
    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = [4, 5, 10, 11]
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)
    
    def _reward_ankle_action(self):
        return torch.norm(self.action_history_buf[:, -1, [4, 5, 10, 11]], dim=1)
