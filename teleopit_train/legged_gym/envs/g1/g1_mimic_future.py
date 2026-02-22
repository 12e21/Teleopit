from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from teleopit_train.legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill
from .g1_mimic_future_config import G1MimicStuFutureCfg
from teleopit_train.legged_gym.gym_utils.math import *
from teleopit_train.pose.utils import torch_utils
from teleopit_train.legged_gym.envs.base.legged_robot import euler_from_quaternion
from teleopit_train.legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos


class G1MimicFuture(G1MimicDistill):
    """Student policy environment with future motion support and curriculum masking.
    Extends G1MimicDistill to add future motion capabilities while maintaining 
    all original RL+BC functionality.
    
    Curriculum Masked Privilege Information (CMP)
    """
    
    def __init__(self, cfg: G1MimicStuFutureCfg, sim_params, physics_engine, sim_device, headless):
        # Store future motion configuration
        self.future_cfg = cfg.env


        # Evaluation mode parameters
        self.evaluation_mode = getattr(cfg.env, 'evaluation_mode', False)
        self.force_full_masking = getattr(cfg.env, 'force_full_masking', False)
        
        
        # Initialize FALCON-style curriculum force application flag BEFORE super().__init__
        # This is needed because reset_idx is called during parent initialization
        self.enable_force_curriculum = getattr(cfg.env, 'enable_force_curriculum', False)
        
        # Initialize force curriculum attributes with default values before calling super().__init__
        # This prevents AttributeError during reset_idx call in parent initialization
        if self.enable_force_curriculum:
            # Essential attributes that are needed before full initialization
            self.force_scale_curriculum = True  # Will be properly set in _init_force_curriculum_components
            # Initialize with empty values - will be properly set after super().__init__()
            self.episode_length_counter = None
            self.force_scale = None
        
        # Call parent constructor
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Fix motion difficulty initialization - should start at 10, not 100
        num_motions = self._motion_lib.num_motions()
        self.motion_difficulty = 10 * torch.ones((num_motions), device=self.device, dtype=torch.float)
        self.mean_motion_difficulty = 10.
        
        # Only initialize future motion components if obs_type is 'student_future'
        if self.obs_type == 'student_future':
            # Initialize future motion target steps
            self._tar_motion_steps_future = torch.tensor(
                getattr(cfg.env, 'tar_motion_steps_future', [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), 
                device=self.device, dtype=torch.long
            )
            
            # Find indices of future steps in the privileged teacher steps
            self._tar_motion_steps_future_idx = torch.searchsorted(
                self._tar_motion_steps_priv, self._tar_motion_steps_future
            )
            
            # Initialize masking buffer
            self.future_mask = torch.ones(
                (self.num_envs, len(self._tar_motion_steps_future)), 
                device=self.device, dtype=torch.bool
            )
            
            print(f"Future motion enabled with {len(self._tar_motion_steps_future)} future frames")
            print(f"Future steps: {self._tar_motion_steps_future.tolist()}")
        
        # Initialize error aware sampling logging
        if hasattr(cfg.motion, 'use_error_aware_sampling') and cfg.motion.use_error_aware_sampling:
            self._error_log_counter = 0
            # Track body errors for statistics
            self.body_error_history = []
            print("Error aware sampling logging initialized")
        
        # Initialize FALCON-style curriculum force application if enabled
        if self.enable_force_curriculum:
            self._init_force_curriculum_components(cfg)
            force_links = getattr(cfg.env.force_curriculum, 'force_apply_links', ['left_rubber_hand', 'right_rubber_hand'])
            print(f"Force curriculum enabled with force application to {len(force_links)} links: {force_links}")
    
    def _get_unified_motion_data(self):
        """Get unified motion data for both privileged and future frames in a single sampling call.
        Returns processed motion data that can be used by both _get_mimic_obs and _get_future_motion_obs.
        """
        # Determine which time steps to sample
        if self.obs_type == 'student_future' and hasattr(self, '_tar_motion_steps_future'):
            all_steps = torch.cat([self._tar_motion_steps_priv, self._tar_motion_steps_future])
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = self._tar_motion_steps_future.shape[0]
        else:
            all_steps = self._tar_motion_steps_priv
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = 0
        
        total_steps = all_steps.shape[0]
        assert total_steps > 0, "Invalid number of target observation steps"
        
        # Single motion sampling call for all time steps
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = all_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = \
            self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        # Apply motion domain randomization noise (unified for all frames)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self._apply_motion_domain_randomization(
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)
        
        # Unified data processing for all frames
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, total_steps, 1)
        pitch = pitch.reshape(self.num_envs, total_steps, 1)
        yaw = yaw.reshape(self.num_envs, total_steps, 1)

        root_vel_local = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)
        
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :] # local body pos
        whole_key_body_pos_global = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        
        # Reshape all observations (unified)
        root_pos = root_pos.reshape(self.num_envs, total_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, total_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, total_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, total_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, total_steps, dof_pos.shape[-1])
        dof_vel = dof_vel.reshape(self.num_envs, total_steps, dof_vel.shape[-1])
        root_vel_local = root_vel_local.reshape(self.num_envs, total_steps, root_vel_local.shape[-1])
        root_ang_vel_local = root_ang_vel_local.reshape(self.num_envs, total_steps, root_ang_vel_local.shape[-1])
        root_pos_delta_local = root_pos_delta_local.reshape(self.num_envs, total_steps, root_pos_delta_local.shape[-1])
        root_rot_delta_local = root_rot_delta_local.reshape(self.num_envs, total_steps, root_rot_delta_local.shape[-1])
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, total_steps, -1)
        whole_key_body_pos_global = whole_key_body_pos_global.reshape(self.num_envs, total_steps, -1)
        
        root_pos_distance_to_target = root_pos - self.root_states[:, 0:3].reshape(self.num_envs, 1, -1)
        
        return {
            'root_pos': root_pos,
            'root_vel': root_vel,
            'root_rot': root_rot,
            'root_ang_vel': root_ang_vel,
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'body_pos': body_pos,
            'root_pos_delta_local': root_pos_delta_local,
            'root_rot_delta_local': root_rot_delta_local,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'root_vel_local': root_vel_local,
            'root_ang_vel_local': root_ang_vel_local,
            'whole_key_body_pos': whole_key_body_pos,
            'whole_key_body_pos_global': whole_key_body_pos_global,
            'root_pos_distance_to_target': root_pos_distance_to_target,
            'num_priv_steps': num_priv_steps,
            'num_future_steps': num_future_steps,
            'total_steps': total_steps
        }

    def _build_future_obs_from_data(self, motion_data):
        """Build future motion observations from unified motion data."""
        if self.obs_type != 'student_future':
            return torch.zeros(self.num_envs, 0, device=self.device)
            
        if motion_data['num_future_steps'] == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        
        # Extract future motion data (from privileged steps onwards)
        num_priv_steps = motion_data['num_priv_steps']
        
        # Get future frame data by slicing unified data
        root_pos = motion_data['root_pos'][:, num_priv_steps:]
        root_vel_local = motion_data['root_vel_local'][:, num_priv_steps:]
        root_ang_vel_local = motion_data['root_ang_vel_local'][:, num_priv_steps:]
        roll = motion_data['roll'][:, num_priv_steps:]
        pitch = motion_data['pitch'][:, num_priv_steps:]
        dof_pos = motion_data['dof_pos'][:, num_priv_steps:]
        
        # Future motion observations (same rich structure as teacher priv_mimic_obs)
        future_obs = torch.cat((
            # root position: xy velocity + z position
            root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
            root_pos[..., 2:3], # 1 dim (z position)
            # root rotation: roll/pitch + yaw angular velocity
            roll, pitch, # 2 dims (roll/pitch orientation)
            root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
            dof_pos, # num_dof dims
        ), dim=-1) # shape: (num_envs, num_future_steps, 6 + num_dof)
        
        return future_obs



    def _get_mimic_obs(self):
        """Override to use unified motion sampling for both privileged and future observations."""
        # Get unified motion data (SINGLE sampling call for all frames)
        motion_data = self._get_unified_motion_data()
        
        # Extract privileged motion data (first num_priv_steps)
        num_steps = motion_data['num_priv_steps']
        
        root_pos = motion_data['root_pos'][:, :num_steps]
        root_vel = motion_data['root_vel'][:, :num_steps]
        root_rot = motion_data['root_rot'][:, :num_steps]
        root_ang_vel = motion_data['root_ang_vel'][:, :num_steps]
        dof_pos = motion_data['dof_pos'][:, :num_steps]
        dof_vel = motion_data['dof_vel'][:, :num_steps]
        root_pos_delta_local = motion_data['root_pos_delta_local'][:, :num_steps]
        root_rot_delta_local = motion_data['root_rot_delta_local'][:, :num_steps]
        roll = motion_data['roll'][:, :num_steps]
        pitch = motion_data['pitch'][:, :num_steps]
        yaw = motion_data['yaw'][:, :num_steps]
        root_vel_local = motion_data['root_vel_local'][:, :num_steps]
        root_ang_vel_local = motion_data['root_ang_vel_local'][:, :num_steps]
        whole_key_body_pos = motion_data['whole_key_body_pos'][:, :num_steps]
        whole_key_body_pos_global = motion_data['whole_key_body_pos_global'][:, :num_steps]
        root_pos_distance_to_target = motion_data['root_pos_distance_to_target'][:, :num_steps]
        
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
        
        # student
        mimic_obs_buf = torch.cat((
            # root position: xy velocity + z position
            root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
            root_pos[..., 2:3], # 1 dim (z position)
            # root rotation: roll/pitch + yaw angular velocity
            roll, pitch, # 2 dims (roll/pitch orientation)
            root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
            dof_pos, # num_dof dims
        ), dim=-1)[:, self._tar_motion_steps_idx_in_teacher, :] # shape: (num_envs, 1, 6 + 2*num_dof)
            
        priv_mimic_obs = priv_mimic_obs_buf.reshape(self.num_envs, -1)
        mimic_obs = mimic_obs_buf.reshape(self.num_envs, -1)

        
        # Add future motion observations only if using student_future obs_type
        if self.obs_type == 'student_future':
            # Build future observations from the SAME motion_data (no additional sampling!)
            future_obs = self._build_future_obs_from_data(motion_data)
            
            return priv_mimic_obs, mimic_obs, future_obs
        else:
            # Return original format for compatibility
            return priv_mimic_obs, mimic_obs

    def compute_observations(self):
        """Override to include future motion observations while maintaining compatibility."""
        # Get IMU observations (same as parent)
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        # Get motion observations
        if self.obs_type == 'student_future':
            priv_mimic_obs, mimic_obs, future_obs = self._get_mimic_obs()
        else:
            priv_mimic_obs, mimic_obs = self._get_mimic_obs()
            future_obs = None
        
        # Proprioceptive observations (same as parent)
        proprio_obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,   # 3 dims
            imu_obs,    # 2 dims
            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
            self.reindex(self.action_history_buf[:, -1]),
        ), dim=-1)
        
        # Add noise if enabled (same as parent)
        if self.cfg.noise.add_noise and self.headless:
            noise_scale = min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.)
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * noise_scale
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec
        
        # Disable ankle dof velocity (same as parent)
        dof_vel_start_dim = 3 + 2 + self.dof_pos.shape[1]
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.
        
        # Private information for critic (same as parent)
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1)
        
        priv_info = torch.cat((
            self.base_lin_vel, # 3 dims
            self.root_states[:, 0:3], # 3 dims
            self.root_states[:, 3:7], # 4 dims
            key_body_pos, # num_bodies * 3 dims
            self.contact_forces[:, self.feet_indices, 2] > 5., # 2 dims
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1,
        ), dim=-1)
        
        # Current observation (for history) - same as parent
        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)
        
        # Privileged observation - same as parent
        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)
        
        self.privileged_obs_buf = priv_obs_buf
        
        # Build final observation based on obs_type
        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
        elif self.obs_type == 'student_future':
            # Include current observation, history, and optional future observations
            obs_components = [
                obs_buf,
                self.obs_history_buf.view(self.num_envs, -1)
            ]

            if future_obs is not None:
                future_obs_flat = future_obs.view(self.num_envs, -1)
                obs_components.append(future_obs_flat)

            self.obs_buf = torch.cat(obs_components, dim=-1)
        else:
            # Default student behavior (maintains full compatibility)
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        # Update history buffers (same as parent) - using in-place operations to avoid memory leak
        if self.cfg.env.history_len > 0:
            # Find episodes that need to be reset
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
            else:
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



    def _get_force_curriculum_info(self):
        """Get force curriculum information for logging."""
        if not self.enable_force_curriculum:
            return {}
            
        info = {
            # Force curriculum parameters
            'force_curriculum_enabled': self.enable_force_curriculum,
            'force_scale_curriculum_enabled': self.force_scale_curriculum,
            
            # Current force scale statistics
            'force_scale_mean': self.apply_force_scale.mean().item(),
            'force_scale_std': self.apply_force_scale.std().item(),
            'force_scale_min_val': self.apply_force_scale.min().item(),
            'force_scale_max_val': self.apply_force_scale.max().item(),
            
            # Applied force magnitudes
            'left_force_magnitude_mean': torch.norm(self.left_ee_apply_force, dim=1).mean().item(),
            'right_force_magnitude_mean': torch.norm(self.right_ee_apply_force, dim=1).mean().item(),
            'total_force_magnitude_mean': (torch.norm(self.left_ee_apply_force, dim=1) + torch.norm(self.right_ee_apply_force, dim=1)).mean().item(),
            
            # Force curriculum thresholds
            'force_scale_up_threshold': self.force_scale_up_threshold,
            'force_scale_down_threshold': self.force_scale_down_threshold,
            
            # Force ranges
            'force_x_range_low': self.apply_force_x_range[0].item(),
            'force_x_range_high': self.apply_force_x_range[1].item(),
            'force_y_range_low': self.apply_force_y_range[0].item(),
            'force_y_range_high': self.apply_force_y_range[1].item(),
            'force_z_range_low': self.apply_force_z_range[0].item(),
            'force_z_range_high': self.apply_force_z_range[1].item(),
        }
        
        return info
    
    def reset_idx(self, env_ids):
        """Override reset to include force curriculum updates."""
        # Call parent reset
        super().reset_idx(env_ids)
        
        # Update force curriculum for reset environments
        if self.enable_force_curriculum:
            self._update_force_curriculum(env_ids)
    
    def pre_physics_step(self, actions):
        """Override pre_physics_step to include force updates."""
        # Call parent pre_physics_step
        super().pre_physics_step(actions)
        
        # Update force curriculum components
        if self.enable_force_curriculum:
            self._calculate_ee_forces()
    
    def post_physics_step(self):
        """Override post_physics_step to include force application."""
        # Apply forces here since pre_physics_step seems to not be called
        if self.enable_force_curriculum:
            self._calculate_ee_forces()
        
        # Call parent post_physics_step
        super().post_physics_step()


    def _log_error_aware_sampling_progress(self):
        """Log error aware sampling progress including body error statistics and max key body errors per motion."""
        import os
        
        if not hasattr(self, '_error_log_counter'):
            self._error_log_counter = 0

        self._error_log_counter += 1
        
        
        # Log every 2000*24 steps (similar to motion difficulty logging)
        if self._error_log_counter % (500*24) == 0:
            # Calculate current body errors for all environments
            body_errors = self._error_tracking_keybody_pos()  # (num_envs,)
            
            # Store body errors for statistics
            self.body_error_history.append(body_errors.clone())
            
            # Calculate statistics across all current environments
            body_error_sum = torch.sum(body_errors).item()
            body_error_max = torch.max(body_errors).item()  
            body_error_min = torch.min(body_errors).item()
            body_error_mean = torch.mean(body_errors).item()
            
            # Create log directories
            base_log_dir = "../../logs"
            if not os.path.exists(base_log_dir):
                os.makedirs(base_log_dir, exist_ok=True)
            
            # Log body error statistics
            error_stats_log_dir = os.path.join(base_log_dir, "logs_body_error_stats")
            if not os.path.exists(error_stats_log_dir):
                os.makedirs(error_stats_log_dir, exist_ok=True)
            
            log_file = os.path.join(error_stats_log_dir, f"{str(self._error_log_counter//24)}.txt")
            with open(log_file, 'w') as f:
                f.write("Body Error Statistics Across Motions:\n")
                f.write(f"Sum: {body_error_sum:.4f}\n")
                f.write(f"Max: {body_error_max:.4f}\n")
                f.write(f"Min: {body_error_min:.4f}\n") 
                f.write(f"Mean: {body_error_mean:.4f}\n")
                f.write(f"Total Environments: {self.num_envs}\n")
                f.write(f"Log Counter: {self._error_log_counter}\n")
            
            # Log to wandb if available
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "error_aware_sampling/body_error_sum": body_error_sum,
                        "error_aware_sampling/body_error_max": body_error_max,
                        "error_aware_sampling/body_error_min": body_error_min,
                        "error_aware_sampling/body_error_mean": body_error_mean,
                        "iteration": self._error_log_counter
                    })
            except ImportError:
                pass  # wandb not available
            except Exception as e:
                print(f"Error logging body error stats to wandb: {e}")


    def _log_max_key_body_error_per_motion(self):
        """Log max key body error for each motion (similar to logs_motion_difficulty)."""
        import os
        
        if not hasattr(self, '_error_log_counter'):
            self._error_log_counter = 0
        self._error_log_counter += 1
        
        # Log every 2000*24 steps (similar to motion difficulty logging)  
        if self._error_log_counter % (500*24) == 0:
            # Get motion names and max key body errors
            motion_names = self._motion_lib.get_motion_names()
            
            # Filter for motions with significant errors (threshold can be adjusted)
            ERROR_THRESHOLD = 0.05  # Log motions with max key body error > 0.05
            high_error_motions = [
                (name, error.item()) 
                for name, error in zip(motion_names, self.max_key_body_error) 
                if error > ERROR_THRESHOLD
            ]
            
            if high_error_motions:
                # Create log directory
                base_log_dir = "../../logs"
                if not os.path.exists(base_log_dir):
                    os.makedirs(base_log_dir, exist_ok=True)
                log_dir = os.path.join(base_log_dir, "logs_max_key_body_error")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # Write to file
                log_file = os.path.join(log_dir, f"{str(self._error_log_counter//24)}.txt")
                with open(log_file, 'w') as f:
                    f.write(f"Motions with max key body error > {ERROR_THRESHOLD}:\n")
                    for name, error in high_error_motions:
                        f.write(f"Motion: {name}, Max Key Body Error: {error:.4f}\n")
                    f.write(f"\nTotal motions with high error: {len(high_error_motions)}\n")
                    f.write(f"Total motions: {len(motion_names)}\n")
                    f.write(f"Log Counter: {self._error_log_counter}\n")
                
                # Log to wandb if available
                try:
                    import wandb
                    if wandb.run is not None:
                        # Log aggregate statistics
                        all_errors = [error.item() for error in self.max_key_body_error]
                        wandb.log({
                            "error_aware_sampling/num_high_error_motions": len(high_error_motions),
                            "error_aware_sampling/total_motions": len(motion_names),
                            "error_aware_sampling/high_error_motion_ratio": len(high_error_motions) / len(motion_names),
                            "error_aware_sampling/max_key_body_error_overall_max": max(all_errors),
                            "error_aware_sampling/max_key_body_error_overall_mean": sum(all_errors) / len(all_errors),
                            "error_aware_sampling/max_key_body_error_threshold": ERROR_THRESHOLD,
                            "iteration": self._error_log_counter
                        })
                except ImportError:
                    pass  # wandb not available
                except Exception as e:
                    print(f"Error logging max key body error stats to wandb: {e}")
    
    def _post_physics_step_callback(self):
        """Override to add error aware sampling logging."""
        # Call parent callback first
        super()._post_physics_step_callback()
        
        # Add error aware sampling logging if enabled
        if hasattr(self.cfg.motion, 'use_error_aware_sampling') and self.cfg.motion.use_error_aware_sampling:
            self._log_error_aware_sampling_progress()
            self._log_max_key_body_error_per_motion()

    # ============================= FALCON Force Curriculum Methods =============================
    
    def _init_force_curriculum_components(self, cfg):
        """Initialize FALCON-style curriculum force application components."""
        # Force curriculum parameters from config
        force_cfg = cfg.env.force_curriculum
        
        # Force application settings
        self.force_apply_links = getattr(force_cfg, 'force_apply_links', ['left_rubber_hand', 'right_rubber_hand'])
        self.force_apply_body_indices = []
        
        # Get body indices for force application links using gym API
        for link_name in self.force_apply_links:
            # Find body index using gym API
            body_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], link_name)
            if body_idx != -1:
                self.force_apply_body_indices.append(body_idx)
            else:
                print(f"Warning: Force application link '{link_name}' not found in robot model")
        
        self.force_apply_body_indices = torch.tensor(self.force_apply_body_indices, device=self.device, dtype=torch.long)
        
        # Curriculum learning parameters
        self.force_scale_curriculum = getattr(force_cfg, 'force_scale_curriculum', True)
        self.force_scale_initial_scale = getattr(force_cfg, 'force_scale_initial_scale', 0.1)
        self.force_scale_up_threshold = getattr(force_cfg, 'force_scale_up_threshold', 210)
        self.force_scale_down_threshold = getattr(force_cfg, 'force_scale_down_threshold', 200)
        self.force_scale_up = getattr(force_cfg, 'force_scale_up', 0.02)
        self.force_scale_down = getattr(force_cfg, 'force_scale_down', 0.02)
        self.force_scale_max = getattr(force_cfg, 'force_scale_max', 1.0)
        self.force_scale_min = getattr(force_cfg, 'force_scale_min', 0.0)
        
        # Force application ranges
        self.apply_force_x_range = torch.tensor(getattr(force_cfg, 'apply_force_x_range', [-40.0, 40.0]), device=self.device)
        self.apply_force_y_range = torch.tensor(getattr(force_cfg, 'apply_force_y_range', [-40.0, 40.0]), device=self.device)
        self.apply_force_z_range = torch.tensor(getattr(force_cfg, 'apply_force_z_range', [-50.0, 5.0]), device=self.device)
        
        # Force randomization
        self.zero_force_prob = getattr(force_cfg, 'zero_force_prob', [0.25, 0.25, 0.25])
        self.randomize_force_duration = getattr(force_cfg, 'randomize_force_duration', [150, 250])
        
        # Advanced force settings
        self.max_force_estimation = getattr(force_cfg, 'max_force_estimation', True)
        self.use_lpf = getattr(force_cfg, 'use_lpf', False)
        self.force_filter_alpha = getattr(force_cfg, 'force_filter_alpha', 0.05)
        
        # Task-specific force behavior
        self.only_apply_z_force_when_walking = getattr(force_cfg, 'only_apply_z_force_when_walking', False)
        self.only_apply_resistance_when_walking = getattr(force_cfg, 'only_apply_resistance_when_walking', True)
        
        # Initialize force curriculum state
        self.force_scale = torch.full((self.num_envs,), self.force_scale_initial_scale, device=self.device)
        self.episode_length_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Force application state
        self.applied_forces = torch.zeros((self.num_envs, len(self.force_apply_body_indices), 3), device=self.device)
        self.force_duration_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.force_duration_target = torch.randint(
            self.randomize_force_duration[0], self.randomize_force_duration[1] + 1,
            (self.num_envs,), device=self.device
        )
        
        # Phase-based force modulation (triangular wave)
        self.force_phase = torch.zeros(self.num_envs, device=self.device)
        
        # Filtered forces for low-pass filtering
        if self.use_lpf:
            self.filtered_forces = torch.zeros_like(self.applied_forces)
        
        print(f"Force curriculum initialized with {len(self.force_apply_body_indices)} force application points")
        print(f"Force ranges: X={self.apply_force_x_range}, Y={self.apply_force_y_range}, Z={self.apply_force_z_range}")
        print(f"Initial force scale: {self.force_scale_initial_scale}")
    
    def _update_force_curriculum(self, env_ids):
        """Update force scale based on episode performance (curriculum learning)."""
        if not self.force_scale_curriculum:
            return
        
        # Skip if force curriculum components not yet fully initialized
        if self.episode_length_counter is None or self.force_scale is None:
            return
            
        # Get episode lengths for the specified environments
        episode_lengths = self.episode_length_counter[env_ids]
        
        # Update force scale based on performance
        # Good performance (long episodes) -> increase force
        good_performance_mask = episode_lengths > self.force_scale_up_threshold
        self.force_scale[env_ids[good_performance_mask]] = torch.clamp(
            self.force_scale[env_ids[good_performance_mask]] + self.force_scale_up,
            self.force_scale_min, self.force_scale_max
        )
        
        # Poor performance (short episodes) -> decrease force
        poor_performance_mask = episode_lengths < self.force_scale_down_threshold
        self.force_scale[env_ids[poor_performance_mask]] = torch.clamp(
            self.force_scale[env_ids[poor_performance_mask]] - self.force_scale_down,
            self.force_scale_min, self.force_scale_max
        )
        
        # Reset episode length counter
        self.episode_length_counter[env_ids] = 0
        
        # Reset force duration counters and targets
        self.force_duration_counter[env_ids] = 0
        self.force_duration_target[env_ids] = torch.randint(
            self.randomize_force_duration[0], self.randomize_force_duration[1] + 1,
            (len(env_ids),), device=self.device
        )
    
    def _calculate_ee_forces(self):
        """Calculate end-effector forces based on FALCON's curriculum force approach."""
        # Increment episode length counter
        self.episode_length_counter += 1
        
        # Increment force duration counter
        self.force_duration_counter += 1
        
        # Check which environments need new force application
        need_new_forces = self.force_duration_counter >= self.force_duration_target
        
        if need_new_forces.any():
            # Reset force duration counter and set new targets
            self.force_duration_counter[need_new_forces] = 0
            self.force_duration_target[need_new_forces] = torch.randint(
                self.randomize_force_duration[0], self.randomize_force_duration[1] + 1,
                (need_new_forces.sum(),), device=self.device
            )
            
            # Generate new forces for environments that need them
            num_new_forces = need_new_forces.sum().item()
            num_links = len(self.force_apply_body_indices)
            
            # Generate random forces for each axis
            force_x = torch.rand(num_new_forces, num_links, device=self.device) * \
                     (self.apply_force_x_range[1] - self.apply_force_x_range[0]) + self.apply_force_x_range[0]
            force_y = torch.rand(num_new_forces, num_links, device=self.device) * \
                     (self.apply_force_y_range[1] - self.apply_force_y_range[0]) + self.apply_force_y_range[0]
            force_z = torch.rand(num_new_forces, num_links, device=self.device) * \
                     (self.apply_force_z_range[1] - self.apply_force_z_range[0]) + self.apply_force_z_range[0]
            
            # Apply zero force probability
            zero_x_mask = torch.rand(num_new_forces, num_links, device=self.device) < self.zero_force_prob[0]
            zero_y_mask = torch.rand(num_new_forces, num_links, device=self.device) < self.zero_force_prob[1]
            zero_z_mask = torch.rand(num_new_forces, num_links, device=self.device) < self.zero_force_prob[2]
            
            force_x[zero_x_mask] = 0.0
            force_y[zero_y_mask] = 0.0
            force_z[zero_z_mask] = 0.0
            
            # Stack forces
            new_forces = torch.stack([force_x, force_y, force_z], dim=-1)  # [num_new_forces, num_links, 3]
            
            # Update forces for environments that need new forces
            self.applied_forces[need_new_forces] = new_forces
        
        # Apply phase-based modulation (triangular wave)
        self.force_phase += 0.02  # Increment phase
        self.force_phase = torch.fmod(self.force_phase, 2.0)  # Keep phase in [0, 2)
        
        # Triangular wave: 0->1->0 over phase [0, 2)
        phase_modulation = torch.where(
            self.force_phase < 1.0,
            self.force_phase,  # Rising edge [0, 1)
            2.0 - self.force_phase  # Falling edge [1, 2)
        )
        
        # Apply curriculum scaling and phase modulation
        final_forces = self.applied_forces.clone()
        for i in range(len(self.force_apply_body_indices)):
            final_forces[:, i] *= self.force_scale.unsqueeze(-1) * phase_modulation.unsqueeze(-1)
        
        # Apply low-pass filtering if enabled
        if self.use_lpf:
            self.filtered_forces = (1.0 - self.force_filter_alpha) * self.filtered_forces + \
                                  self.force_filter_alpha * final_forces
            final_forces = self.filtered_forces
        
        # Apply forces to simulation using the correct tensor API
        if len(self.force_apply_body_indices) > 0:
            # Create forces tensor for all rigid bodies (initialize with zeros)
            all_forces = torch.zeros((self.num_envs * self.num_bodies, 3), device=self.device, dtype=torch.float)
            
            for i, body_idx in enumerate(self.force_apply_body_indices):
                # Calculate global body indices for all environments
                for env_id in range(self.num_envs):
                    global_body_idx = env_id * self.num_bodies + body_idx
                    all_forces[global_body_idx] = final_forces[env_id, i]
     
            # Apply forces using the tensor API
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(all_forces),
                None,  # No torques
                gymapi.ENV_SPACE
            )
    
