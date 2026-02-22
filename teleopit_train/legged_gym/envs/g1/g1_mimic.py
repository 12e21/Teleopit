from isaacgym.torch_utils import *

import torch

from teleopit_train.legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .g1_mimic_config import G1MimicCfg
from teleopit_train.legged_gym.gym_utils.math import *
from teleopit_train.pose.utils import torch_utils
from teleopit_train.legged_gym.envs.base.legged_robot import euler_from_quaternion
from teleopit_train.legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos


class G1Mimic(HumanoidMimic):
    def __init__(self, cfg: G1MimicCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        

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
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
        
        # align mocap
        mimic_obs_buf = torch.cat((
            root_pos[..., 0:3], # 3 dims
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 0:3], # 3 dims
            dof_pos, # num_dof dims
        ), dim=-1)[:, :] # shape: (num_envs, 1, 3*4 + num_dof)
        
        return mimic_obs_buf.reshape(self.num_envs, -1)

    def _get_current_mimic_obs(self):
        # for Joao evaluation
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
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
        

        cur_dof_pos = dof_pos[:, 0, :]
        cur_root_pos = root_pos[:, 0, :]
        cur_root_rot = root_rot[:, 0, :]
        return cur_dof_pos, cur_root_pos, cur_root_rot
    
    def _get_current_dof(self):
        # for Joao evaluation
        current_dof = self.dof_pos
        current_root_pos = self.root_states[:, 0:3]
        current_root_rot = self.root_states[:, 3:7]
        return current_dof, current_root_pos, current_root_rot
    
    
    
    def compute_observations(self):
        # imu_obs = torch.stack((self.roll, self.pitch, self.yaw - self.init_yaw), dim=1)
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        mimic_obs = self._get_mimic_obs()
        obs_buf = torch.cat((
                            mimic_obs, # (9 + num_dof) * num_steps
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        if self.cfg.noise.add_noise and self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_buf += 0.
        
        dof_vel_start_dim = mimic_obs.shape[1] + 5 + self.dof_pos.shape[1]
        obs_buf[:, [dof_vel_start_dim + 4, dof_vel_start_dim + 5, dof_vel_start_dim + 10, dof_vel_start_dim + 11]] = 0.
        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
                self.base_lin_vel,
            ), dim=-1)
        else:
            priv_latent = torch.zeros((self.num_envs, self.cfg.env.n_priv_latent), device=self.device)
            priv_latent = torch.cat((priv_latent, self.base_lin_vel), dim=-1)

       
        self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)
            )
        


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
