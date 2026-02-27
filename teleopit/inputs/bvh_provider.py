from typing import Dict, Any, Tuple
import numpy as np
from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
from teleopit.retargeting.gmr.utils.lafan_vendor import utils
from scipy.spatial.transform import Rotation as R


def _load_bvh_file(bvh_file: str, format: str = "lafan1"):
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    # hc_mocap uses meters, lafan1/nokov use centimeters
    scale_divisor = 1.0 if format == "hc_mocap" else 100.0

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / scale_divisor
            result[bone] = [position, orientation]

        if format == "lafan1":
            left_foot_key = "LeftFoot" if "LeftFoot" in result else "LeftAnkle"
            right_foot_key = "RightFoot" if "RightFoot" in result else "RightAnkle"
            left_toe_key = "LeftToe" if "LeftToe" in result else "LeftToeBase"
            right_toe_key = "RightToe" if "RightToe" in result else "RightToeBase"
            result["LeftFootMod"] = [result[left_foot_key][0], result[left_toe_key][1]]
            result["RightFootMod"] = [result[right_foot_key][0], result[right_toe_key][1]]
        elif format == "nokov":
            left_foot_key = "LeftFoot" if "LeftFoot" in result else "LeftAnkle"
            right_foot_key = "RightFoot" if "RightFoot" in result else "RightAnkle"
            left_toe_key = "LeftToe" if "LeftToe" in result else "LeftToeBase"
            right_toe_key = "RightToe" if "RightToe" in result else "RightToeBase"
            result["LeftFootMod"] = [result[left_foot_key][0], result[left_toe_key][1]]
            result["RightFootMod"] = [result[right_foot_key][0], result[right_toe_key][1]]
        elif format == "hc_mocap":
            # hc_mocap has no Toe joints — use hc_Foot_L/R position + orientation
            result["LeftFootMod"] = [result["hc_Foot_L"][0], result["hc_Foot_L"][1]]
            result["RightFootMod"] = [result["hc_Foot_R"][0], result["hc_Foot_R"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")

        frames.append(result)

    # Read FPS from BVH Frame Time field
    if data.frametime is not None and data.frametime > 0:
        fps = int(round(1.0 / data.frametime))
    else:
        fps = 30

    # Estimate human height from skeleton offsets (sum Y offsets from root to head)
    bone_names = list(data.bones)
    offsets = data.offsets  # shape: (num_joints, 3)
    parents = data.parents
    # Build path: root → Spine → hc_Chest → hc_Chest1 → neck → hc_Head (or similar)
    # Sum absolute Y offsets along the longest vertical chain
    if format == "hc_mocap":
        # Upward chain: root → Spine → hc_Chest → hc_Chest1 → neck → hc_Head → end_site
        up_joints = ["Spine", "hc_Chest", "hc_Chest1", "neck", "hc_Head"]
        up_height = 0.0
        for jname in up_joints:
            if jname in bone_names:
                idx = bone_names.index(jname)
                up_height += abs(offsets[idx][1])  # Y offset in BVH (Y-up)
        # Downward chain: root → hc_Hip_L → hc_Knee_L → hc_Foot_L
        down_joints = ["hc_Hip_L", "hc_Knee_L", "hc_Foot_L"]
        down_height = 0.0
        for jname in down_joints:
            if jname in bone_names:
                idx = bone_names.index(jname)
                down_height += abs(offsets[idx][1])  # Y offset in BVH (Y-up)
        human_height = up_height + down_height
        if human_height < 0.5:  # fallback
            human_height = 1.75
    else:
        human_height = 1.75

    # Downsample hc_mocap from 60fps to 30fps
    if format == "hc_mocap" and fps == 60:
        frames = frames[::2]

    return frames, human_height, fps


class BVHInputProvider:
    
    def __init__(self, bvh_path: str, human_format: str = "lafan1"):
        self.bvh_path = bvh_path
        self.human_format = human_format
        self._frames, self._human_height, self._fps = _load_bvh_file(bvh_path, format=human_format)
        self._current_frame = 0
        
    def get_frame(self) -> Dict[str, Tuple[Any, Any]]:
        if self._current_frame >= len(self._frames):
            raise StopIteration("No more frames available")
        
        frame_data = self._frames[self._current_frame]
        self._current_frame += 1
        
        return {
            body_name: (np.array(data[0]), np.array(data[1]))
            for body_name, data in frame_data.items()
        }
    
    def reset(self) -> None:
        self._current_frame = 0
    
    def is_available(self) -> bool:
        return self._current_frame < len(self._frames)
    
    def __len__(self) -> int:
        return len(self._frames)
    
    @property
    def fps(self) -> int:
        return self._fps
    
    @property
    def human_height(self) -> float:
        return self._human_height
