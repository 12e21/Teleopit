import glob, os, pickle, yaml
import torch
from train_mimic.pose.utils.torch_utils import quat_diff, quat_to_exp_map, slerp, euler_from_quaternion
from tqdm import tqdm
from rich import print
from train_mimic.pose.utils.isaacgym_torch_utils import quat_rotate_inverse, quat_mul, quat_conjugate
import sys
from types import ModuleType
import numpy as np

# Patch sys.modules to fake missing modules from numpy 2.x
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

# Patch potentially missing modules
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))


def smooth(x, box_pts, device):
    box = torch.ones(box_pts, device=device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding='same'
    )
    return smoothed.squeeze(0).T


class MotionLib:
    def __init__(self, motion_file, device, 
                 motion_decompose=False, 
                 motion_smooth=True, 
                 motion_height_adjust=False,
                 sample_ratio=1.0 # only sample a portion of the motion
                 ):
        self._device = device

        # motion augmentation by decomposing long motion into short motions
        self._motion_decompose = motion_decompose
        # motion smoothing
        self._motion_smooth = motion_smooth
        # motion height adjustment
        self._motion_height_adjust = motion_height_adjust
        # sample a portion of the motion
        self._sample_ratio = sample_ratio
        
        # load motions
        self._load_motions(motion_file)
        
        
    def _load_motions(self, motion_file):
        self._motion_names = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_files = []
        
        self._motion_root_pos_delta = []
        self._motion_root_pos = []
        self._motion_root_rot = []
        self._motion_root_vel = []
        self._motion_root_ang_vel = []
        self._motion_dof_pos = []
        self._motion_root_pos_delta_local = []
        self._motion_root_rot_delta_local = []
        self._motion_dof_vel = []
        self._motion_local_body_pos = []
        self._body_link_list = []
        
        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        
        num_sub_motions_total = 0
            
        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            if torch.rand(1) > self._sample_ratio and num_motion_files > 1:
                continue
            
            curr_file = motion_files[i]
            if not os.path.exists(curr_file):
                print(f"Motion file {curr_file} does not exist")
                continue

            try:
                with open(curr_file, "rb") as f:
                    motion_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading motion file {curr_file}: {e}")
                continue
            fps = motion_data["fps"]
            curr_weight = motion_weights[i]
            root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float, device=self._device)
            root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float, device=self._device)
            dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float, device=self._device)
            local_body_pos = torch.tensor(motion_data["local_body_pos"], dtype=torch.float, device=self._device)
            if self._body_link_list is None or len(self._body_link_list) == 0:
                self._body_link_list = motion_data["link_body_list"]
            num_frames = root_pos.shape[0]
            motion_len_s = 1.0 / fps * (num_frames - 1)
            
            if self._motion_height_adjust:
                # compute the lowest body part in reference motion
                body_pos = local_body_pos + root_pos.unsqueeze(1)
                lowest_body_part = torch.min(body_pos[..., 2])
                # adjust the height of the root position
                root_pos[..., 2] -= lowest_body_part
                
            try:
                self._add_motions(root_pos, root_rot, dof_pos, local_body_pos, fps, curr_weight, curr_file)
            except Exception as e:
                print(f"Error adding motion {curr_file}: {e}")
                continue
            
            
            if self._motion_decompose:
                # Decompose long motion into short motions
                base_motion_len_s = 10.0 # 10 seconds for each sub-motion
                # base_motion_len_s = 20.0 # 20 seconds for each sub-motion
                # base_motion_len_s = 30.0 # 30 seconds for each sub-motion
                if motion_len_s < base_motion_len_s:
                    continue
                # divide motion into sub-motions of base_motion_len
                num_sub_motions = int(motion_len_s / base_motion_len_s)
                # if the motion is longer than the base_motion_len, add one more sub-motion
                if motion_len_s > base_motion_len_s * num_sub_motions:
                    num_sub_motions += 1
                
                num_sub_motions_total += num_sub_motions
                for i in range(num_sub_motions):
                    start_idx = int(i * base_motion_len_s * fps)
                    end_idx = int(start_idx + base_motion_len_s * fps)
                    
                    # get the sub-motion
                    sub_root_pos = root_pos[start_idx:end_idx]
                    sub_root_rot = root_rot[start_idx:end_idx]
                    sub_dof_pos = dof_pos[start_idx:end_idx]
                    sub_local_body_pos = local_body_pos[start_idx:end_idx]
                    # sub_weight = curr_weight + i # we increase the weight of the sub-motion by i
                    sub_weight = curr_weight
                    self._add_motions(sub_root_pos, sub_root_rot, sub_dof_pos, sub_local_body_pos, fps, sub_weight, curr_file)
                # print(f"Decomposed {curr_file} into {num_sub_motions} sub-motions")
        
        print(f"Total number of sub-motions: {num_sub_motions_total}")
                        
        assert len(self._motion_weights) == len(self._motion_names), f"len(self._motion_weights) = {len(self._motion_weights)}, len(self._motion_names) = {len(self._motion_names)}"
        assert len(self._motion_weights) == len(self._motion_files), f"len(self._motion_weights) = {len(self._motion_weights)}, len(self._motion_files) = {len(self._motion_files)}"
        assert len(self._motion_weights) == len(self._motion_fps), f"len(self._motion_weights) = {len(self._motion_weights)}, len(self._motion_fps) = {len(self._motion_fps)}"
        
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights /= torch.sum(self._motion_weights)
        
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float, device=self._device)

        self._motion_root_pos_delta = torch.stack(self._motion_root_pos_delta, dim=0)
        
        self._motion_root_pos = torch.cat(self._motion_root_pos, dim=0)
        self._motion_root_rot = torch.cat(self._motion_root_rot, dim=0)
        self._motion_root_vel = torch.cat(self._motion_root_vel, dim=0)
        self._motion_root_ang_vel = torch.cat(self._motion_root_ang_vel, dim=0)
        self._motion_dof_pos = torch.cat(self._motion_dof_pos, dim=0)
        self._motion_dof_vel = torch.cat(self._motion_dof_vel, dim=0)
        self._motion_local_body_pos = torch.cat(self._motion_local_body_pos, dim=0)
        self._motion_root_pos_delta_local = torch.cat(self._motion_root_pos_delta_local, dim=0)
        self._motion_root_rot_delta_local = torch.cat(self._motion_root_rot_delta_local, dim=0)
        
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        
        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

    def _add_motions(self, root_pos, root_rot, dof_pos, local_body_pos, fps, curr_weight, curr_file):
        dt = 1.0 / fps
        num_frames = root_pos.shape[0]
        curr_len = dt * (num_frames - 1)
        
        root_pos_delta = root_pos[-1] - root_pos[0]
        root_pos_delta[..., -1] = 0.0
        
        root_vel = torch.gradient(root_pos, spacing=dt, dim=0)[0]
        
        # compute the delta pos per frame
        root_pos_delta_local = torch.zeros_like(root_pos)
        root_pos_delta_local[1:, :] = root_pos[1:, :] - root_pos[:-1, :] # cur frame delta pos = cur frame pos - last frame pos
        root_pos_delta_local[0, :] = 0.0 # first frame delta pos = 0
        root_pos_delta_local[1:, :] = quat_rotate_inverse(root_rot[:-1, :], root_pos_delta_local[1:, :]) # rotate the delta pos to local frame via last frame rot
        
        # compute the delta rot per frame
        root_rot_delta_local = torch.zeros_like(root_pos)
        root_rot_delta_local[1:, :] = euler_from_quaternion(quat_diff(root_rot[1:, :], root_rot[:-1, :])) # cur frame delta rot = cur frame rot - last frame rot
        root_rot_delta_local[0, :] = 0.0
        root_rot_delta_local[1:, :] = quat_rotate_inverse(root_rot[:-1, :], root_rot_delta_local[1:, :]) # rotate the delta rot to local frame via last frame rot
        
        root_ang_vel = self._compute_so3_derivative(root_rot, dt)
        
        dof_vel = torch.gradient(dof_pos, spacing=dt, dim=0)[0]
        
        self._motion_weights.append(curr_weight)
        self._motion_fps.append(fps)
        self._motion_dt.append(dt)
        self._motion_num_frames.append(num_frames)
        self._motion_lengths.append(curr_len)
        self._motion_files.append(curr_file)
        
        self._motion_root_pos_delta.append(root_pos_delta)
        self._motion_root_pos.append(root_pos)
        self._motion_root_rot.append(root_rot)
        self._motion_root_vel.append(root_vel)
        self._motion_root_ang_vel.append(root_ang_vel)
        self._motion_dof_pos.append(dof_pos)
        self._motion_root_pos_delta_local.append(root_pos_delta_local)
        self._motion_root_rot_delta_local.append(root_rot_delta_local)
        self._motion_dof_vel.append(dof_vel)
        self._motion_local_body_pos.append(local_body_pos)
        self._motion_names.append(os.path.basename(curr_file))
    
    def _compute_so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations using central differences.
        
        Args:
            rotations: Quaternion rotations with shape (T, 4).
            dt: Time step.
        Returns:
            Angular velocities with shape (T, 3).
        """
        if rotations.shape[0] < 3:
            # For very short sequences, fall back to forward differences
            root_drot = quat_diff(rotations[:-1], rotations[1:])
            omega = quat_to_exp_map(root_drot) / dt
            omega = torch.cat([omega, omega[-1:]], dim=0)  # Repeat last
            return omega
        
        # Use central differences for interior points
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega_interior = quat_to_exp_map(q_rel) / (2.0 * dt)
        
        # Handle boundaries with forward/backward differences
        q_start_rel = quat_mul(rotations[1], quat_conjugate(rotations[0]))
        omega_start = quat_to_exp_map(q_start_rel) / dt
        
        q_end_rel = quat_mul(rotations[-1], quat_conjugate(rotations[-2]))
        omega_end = quat_to_exp_map(q_end_rel) / dt
        
        # Combine all parts
        omega = torch.cat([omega_start.unsqueeze(0), omega_interior, omega_end.unsqueeze(0)], dim=0)
        return omega
    
    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
        
    def num_motions(self):
        return self._motion_weights.shape[0]
    
    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()
    
    def sample_motions(self, n, motion_difficulty=None, max_key_body_error=None, 
                      use_error_aware_sampling=False, error_sampling_power=5.0, 
                      error_sampling_threshold=0.15):
        if motion_difficulty is not None:
            if use_error_aware_sampling and max_key_body_error is not None:
                # Apply error aware sampling formula
                error_aware_prob = torch.ones_like(motion_difficulty)
                
                # Apply error aware probability only when motion_difficulty == 1
                difficulty_one_mask = (motion_difficulty == 1.0)
                if difficulty_one_mask.any():
                    normalized_error = torch.clamp(max_key_body_error / error_sampling_threshold, max=1.0)
                    error_prob = normalized_error ** error_sampling_power
                    error_aware_prob[difficulty_one_mask] = error_prob[difficulty_one_mask]
                
                # For motion_difficulty > 1, use original difficulty
                difficulty_gt_one_mask = (motion_difficulty > 1.0)
                error_aware_prob[difficulty_gt_one_mask] = motion_difficulty[difficulty_gt_one_mask]
                
                motion_prob = self._motion_weights * error_aware_prob
            else:
                motion_prob = self._motion_weights * motion_difficulty
        else:
            motion_prob = self._motion_weights
        
        motion_ids = torch.multinomial(motion_prob, num_samples=n, replacement=True)
        return motion_ids
    
    def sample_time(self, motion_ids):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        
        motion_time = motion_len * phase
        return motion_time
                
    def _fetch_motion_files(self, motion_file: str):
        if os.path.isdir(motion_file):
            motion_files = sorted(glob.glob(os.path.join(motion_file, "*.pkl")))
            if not motion_files:
                raise FileNotFoundError(
                    f"No .pkl files found in directory: {motion_file}"
                )
            motion_weights = [1.0] * len(motion_files)
        elif motion_file.endswith(".yaml"):
            motion_files = []
            motion_weights = []
            with open(motion_file, "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            
            motion_root_path = motion_config["root_path"]
            motion_list = motion_config["motions"]
            for motion_entry in motion_list:
                curr_file = os.path.join(motion_root_path, motion_entry['file'])
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
        
        return motion_files, motion_weights
    
    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]
        
        phase = times / self._motion_lengths[motion_ids]
        phase = torch.clip(phase, 0.0, 1.0)
        
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0.float()
        
        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx0 += frame_start_idx
        frame_idx1 += frame_start_idx
        
        return frame_idx0, frame_idx1, blend
        
    def calc_motion_frame(self, motion_ids, motion_times):
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times -= motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        
        root_pos0 = self._motion_root_pos[frame_idx0]
        root_pos1 = self._motion_root_pos[frame_idx1]
        
        root_rot0 = self._motion_root_rot[frame_idx0]
        root_rot1 = self._motion_root_rot[frame_idx1]
        
        root_vel = self._motion_root_vel[frame_idx0]
        root_ang_vel = self._motion_root_ang_vel[frame_idx0]
        
        dof_pos0 = self._motion_dof_pos[frame_idx0]
        dof_pos1 = self._motion_dof_pos[frame_idx1]
        
        local_key_body_pos0 = self._motion_local_body_pos[frame_idx0]
        local_key_body_pos1 = self._motion_local_body_pos[frame_idx1]
        
        dof_vel = self._motion_dof_vel[frame_idx0]
        
        blend_unsqueeze = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
        root_pos += motion_loop_num.unsqueeze(-1) * self._motion_root_pos_delta[motion_ids]
        root_rot = slerp(root_rot0, root_rot1, blend)
        
        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1
        
        local_key_body_pos = (1.0 - blend_unsqueeze.unsqueeze(1)) * local_key_body_pos0 + blend_unsqueeze.unsqueeze(1) * local_key_body_pos1
        
        # compute the root pos delta compared to last frame
        root_pos_delta_local0 = self._motion_root_pos_delta_local[frame_idx0]
        root_pos_delta_local1 = self._motion_root_pos_delta_local[frame_idx1]
        root_pos_delta_local = (1.0 - blend_unsqueeze) * root_pos_delta_local0 + blend_unsqueeze * root_pos_delta_local1

        # compute the root rot delta compared to last frame 
        root_rot_delta_local0 = self._motion_root_rot_delta_local[frame_idx0]
        root_rot_delta_local1 = self._motion_root_rot_delta_local[frame_idx1]
        # we use linear interpolation for root rot delta, as it is euler angle
        root_rot_delta_local = (1.0 - blend_unsqueeze) * root_rot_delta_local0 + blend_unsqueeze * root_rot_delta_local1

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local
    
    def get_key_body_idx(self, key_body_names):
        key_body_idx = []
        for key_body_name in key_body_names:
            key_body_idx.append(self._body_link_list.index(key_body_name))
        return key_body_idx # list
    
    def get_motion_names(self):
        return self._motion_names
        