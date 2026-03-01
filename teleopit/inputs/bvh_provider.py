from typing import Dict, Any, Tuple
import numpy as np
from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
from teleopit.retargeting.gmr.utils.lafan_vendor import utils
from scipy.spatial.transform import Rotation as R


def _load_bvh_file(bvh_file: str, format: str = "lafan1"):
    data = read_bvh(bvh_file)
    bone_names = list(data.bones)
    bone_parents = np.array(data.parents, dtype=np.int32)
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
            if format == "hc_mocap":
                position = position + np.array([0.0, 0.0, 0.9526])
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
            # hc_Foot_L/R position + LeftToeBase/RightToeBase orientation
            result["LeftFootMod"] = [result["hc_Foot_L"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["hc_Foot_R"][0], result["RightToeBase"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")

        frames.append(result)

    # Read FPS from BVH Frame Time field
    if data.frametime is not None and data.frametime > 0:
        fps = int(round(1.0 / data.frametime))
    else:
        fps = 30

    if format == "hc_mocap":
        # Use frame 0 global FK positions for accurate height (head_Y - lowest_foot_Y)
        head_idx = list(data.bones).index("hc_Head") if "hc_Head" in data.bones else None
        toe_l_idx = list(data.bones).index("LeftToeBase") if "LeftToeBase" in data.bones else None
        toe_r_idx = list(data.bones).index("RightToeBase") if "RightToeBase" in data.bones else None
        if head_idx is not None and toe_l_idx is not None:
            head_y = global_data[1][0, head_idx, 1]  # Y in BVH Y-up space
            toe_y = min(global_data[1][0, toe_l_idx, 1], global_data[1][0, toe_r_idx, 1])
            human_height = head_y - toe_y
        else:
            human_height = 1.75
        if human_height < 0.5:
            human_height = 1.75
    else:
        human_height = 1.75

    # Downsample hc_mocap from 60fps to 30fps
    if format == "hc_mocap" and fps == 60:
        frames = frames[::2]
        fps = 30

    return frames, human_height, fps, bone_names, bone_parents


class BVHInputProvider:
    
    def __init__(self, bvh_path: str, human_format: str = "lafan1"):
        self.bvh_path = bvh_path
        self.human_format = human_format
        self._frames, self._human_height, self._fps, self._bone_names, self._bone_parents = _load_bvh_file(bvh_path, format=human_format)
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

    @property
    def bone_names(self) -> list[str]:
        return self._bone_names

    @property
    def bone_parents(self) -> np.ndarray:
        return self._bone_parents
