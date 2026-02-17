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

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100
            result[bone] = [position, orientation]

        left_foot_key = "LeftFoot" if "LeftFoot" in result else "LeftAnkle"
        right_foot_key = "RightFoot" if "RightFoot" in result else "RightAnkle"
        left_toe_key = "LeftToe" if "LeftToe" in result else "LeftToeBase"
        right_toe_key = "RightToe" if "RightToe" in result else "RightToeBase"
            
        if format == "lafan1":
            result["LeftFootMod"] = [result[left_foot_key][0], result[left_toe_key][1]]
            result["RightFootMod"] = [result[right_foot_key][0], result[right_toe_key][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result[left_foot_key][0], result[left_toe_key][1]]
            result["RightFootMod"] = [result[right_foot_key][0], result[right_toe_key][1]]
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
    
    human_height = 1.75
    return frames, human_height, 30


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
