from __future__ import annotations

import re
from typing import Dict, Any, Tuple

import numpy as np
from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
from teleopit.retargeting.gmr.utils.lafan_vendor import utils
from scipy.spatial.transform import Rotation as R


# Channel name → single-letter axis (same mapping as extract.py)
_CHANNELMAP = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z',
}


def _parse_bvh_header(bvh_path: str) -> tuple[str, int]:
    """Parse the BVH header to extract Euler rotation order and channel count.

    Mimics the parsing behaviour of ``extract.read_bvh``: ``channels`` is
    updated by every ``CHANNELS`` line (so the *last* one wins), while
    ``euler_order`` is captured only once from the first ``CHANNELS`` line.

    Returns (euler_order, channels) e.g. ("zxy", 3) for hc_mocap.
    """
    euler_order: str | None = None
    channels: int = 3

    with open(bvh_path, "r") as f:
        for line in f:
            if "MOTION" in line:
                break
            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if euler_order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis:2 + channelie]
                    if all(p in _CHANNELMAP for p in parts):
                        euler_order = "".join(_CHANNELMAP[p] for p in parts)

    return euler_order or "zyx", channels


def process_single_bvh_frame(
    data_floats: np.ndarray,
    offsets: np.ndarray,
    bone_names: list[str],
    bone_parents: np.ndarray,
    euler_order: str,
    rotation_quat: np.ndarray,
    rotation_matrix: np.ndarray,
    format: str,
    scale_divisor: float,
    channels: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Process a single BVH frame (one line of motion data) into a pose dict.

    The output format matches ``BVHInputProvider.get_frame()``.
    """
    N = len(bone_names)

    # Parse positions and euler angles from the flat float array
    positions = offsets.copy()  # (N, 3)
    if channels == 3:
        positions[0] = data_floats[0:3]
        euler_deg = data_floats[3:].reshape(N, 3)
    elif channels == 6:
        data_block = data_floats.reshape(N, 6)
        positions[:] = data_block[:, 0:3]
        euler_deg = data_block[:, 3:6]
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    # Euler → local quaternions → FK
    local_quats = utils.euler_to_quat(np.radians(euler_deg), order=euler_order)
    global_quats, global_pos = utils.quat_fk(
        local_quats[np.newaxis], positions[np.newaxis], bone_parents
    )
    global_quats = global_quats[0]  # (N, 4)
    global_pos = global_pos[0]      # (N, 3)

    # Coordinate transform (Y-up → Z-up) + scale + height offset
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, bone in enumerate(bone_names):
        orientation = utils.quat_mul(rotation_quat, global_quats[i])
        position = global_pos[i] @ rotation_matrix.T / scale_divisor
        if format == "hc_mocap":
            position = position + np.array([0.0, 0.0, 0.9526])
        result[bone] = (position, orientation)

    # Foot synthesis
    if format == "lafan1" or format == "nokov":
        left_foot_key = "LeftFoot" if "LeftFoot" in result else "LeftAnkle"
        right_foot_key = "RightFoot" if "RightFoot" in result else "RightAnkle"
        left_toe_key = "LeftToe" if "LeftToe" in result else "LeftToeBase"
        right_toe_key = "RightToe" if "RightToe" in result else "RightToeBase"
        result["LeftFootMod"] = (result[left_foot_key][0], result[left_toe_key][1])
        result["RightFootMod"] = (result[right_foot_key][0], result[right_toe_key][1])
    elif format == "hc_mocap":
        result["LeftFootMod"] = (result["hc_Foot_L"][0], result["LeftToeBase"][1])
        result["RightFootMod"] = (result["hc_Foot_R"][0], result["RightToeBase"][1])
    else:
        raise ValueError(f"Invalid format: {format}")

    return result


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
            # hc_Foot_L/R position + LeftToeBase/RightToeBase orientation (if available)
            left_toe_key = "LeftToeBase" if "LeftToeBase" in result else "hc_Foot_L"
            right_toe_key = "RightToeBase" if "RightToeBase" in result else "hc_Foot_R"
            result["LeftFootMod"] = [result["hc_Foot_L"][0], result[left_toe_key][1]]
            result["RightFootMod"] = [result["hc_Foot_R"][0], result[right_toe_key][1]]
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
        # Fallback to foot joints if toe joints not present
        if toe_l_idx is None:
            toe_l_idx = list(data.bones).index("hc_Foot_L") if "hc_Foot_L" in data.bones else None
        if toe_r_idx is None:
            toe_r_idx = list(data.bones).index("hc_Foot_R") if "hc_Foot_R" in data.bones else None
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

    # Detect whether skeleton has toe joints (affects IK config selection)
    has_toe = "LeftToeBase" in set(data.bones)

    # Downsample hc_mocap from 60fps to 30fps
    if format == "hc_mocap" and fps == 60:
        frames = frames[::2]
        fps = 30

    return frames, human_height, fps, bone_names, bone_parents, has_toe


class BVHInputProvider:
    
    def __init__(self, bvh_path: str, human_format: str = "lafan1"):
        self.bvh_path = bvh_path
        self._frames, self._human_height, self._fps, self._bone_names, self._bone_parents, has_toe = _load_bvh_file(bvh_path, format=human_format)
        # Auto-select IK config: hc_mocap without toe joints needs different ankle offsets
        if human_format == "hc_mocap" and not has_toe:
            self.human_format = "hc_mocap_no_toe"
        else:
            self.human_format = human_format
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
