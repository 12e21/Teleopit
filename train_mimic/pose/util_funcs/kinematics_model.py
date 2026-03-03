import torch

from isaacgym.torch_utils import quat_rotate, quat_apply
import pytorch_kinematics as pk


class KinematicsModel:
    def __init__(self, file_path: str, device):
        self.device = device        
        if file_path.endswith(".urdf"):
            self.chain = pk.build_chain_from_urdf(open(file_path, mode="rb").read())
        elif file_path.endswith(".xml") or file_path.endswith(".mjcf"):
            self.chain = pk.build_chain_from_mjcf(open(file_path, mode="rb").read())
            
        self.chain = self.chain.to(device=device)
        self.num_joints = len(self.chain.get_joint_parameter_names())
        self.reindex = [12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    def forward_kinematics(self, joint_angles: torch.Tensor, base_pos, base_rot, key_bodies: list):
        assert joint_angles.shape[1] == self.num_joints, f"number of joints mismatch: {joint_angles.shape[1]} != {self.num_joints}"
        joint_angles = joint_angles[:, self.reindex]
        
        global_key_body_pos = torch.zeros((joint_angles.shape[0], len(key_bodies), 3), device=self.device)
        
        ret = self.chain.forward_kinematics(joint_angles)
        
        for i, key_body in enumerate(key_bodies):
            tg = ret[key_body]
            m = tg.get_matrix()
            pos = m[:, :3, 3]
            global_key_body_pos[:, i, :] = pos
            
        flat_global_key_body_pos = quat_apply(base_rot, global_key_body_pos.view(-1, 3)).view(global_key_body_pos.shape)
        exit()
        # global_key_body_pos = base_pos + quat_apply(base_rot, global_key_body_pos)
        return global_key_body_pos
        