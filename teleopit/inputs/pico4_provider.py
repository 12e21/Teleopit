"""Pico4 VR full-body motion capture input provider.

Uses xrobotoolkit_sdk to receive real-time body tracking data from a Pico4
headset. Outputs the same ``{joint_name: (position, quat)}`` dict format as
the BVH providers, so downstream retargeting is unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.rot_utils import quat_mul_np

logger = logging.getLogger(__name__)

# Pre-compute the provider-space to Teleopit-space transform.
_INPUT_TO_TELEOPIT_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_INPUT_TO_TELEOPIT_QUAT = R.from_matrix(_INPUT_TO_TELEOPIT_MATRIX).as_quat(scalar_first=True)

# SMPL-X 24 body joint names from xrobotoolkit_sdk
BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

HumanFrame = Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]


def _coordinate_transform_input(body_pose_dict: dict) -> dict:
    """Transform provider-space poses into Teleopit's expected coordinates."""
    for body_name, value in body_pose_dict.items():
        x, y, z = value[0]
        qw, qx, qy, qz = value[1]

        orientation = quat_mul_np(
            _INPUT_TO_TELEOPIT_QUAT, np.array([qw, qx, qy, qz]), scalar_first=True
        )
        position = np.array([x, y, z]) @ _INPUT_TO_TELEOPIT_MATRIX.T

        body_pose_dict[body_name] = [position.tolist(), orientation.tolist()]

    return body_pose_dict


class Pico4InputProvider:
    """Real-time input provider using Pico4 full-body tracking via xrobotoolkit_sdk.

    Implements the same interface as ``BVHInputProvider`` / ``UDPBVHInputProvider``
    so it plugs directly into the Teleopit pipeline.
    """

    def __init__(self, human_format: str = "xrobot", timeout: float = 60.0) -> None:
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise ImportError(
                "xrobotoolkit_sdk is required for Pico4 input. "
                "Install the Pico4 SDK or use a different input provider."
            ) from exc

        self._xrt = xrt
        self._xrt.init()
        self._human_format = human_format
        self._timeout = timeout
        self._closed = False
        self._has_received_data = False
        logger.info("Pico4InputProvider initialized (xrobotoolkit_sdk)")

    @property
    def fps(self) -> int:
        """Pico4 body tracking runs at ~30 fps."""
        return 30

    @property
    def human_format(self) -> str:
        return self._human_format

    def is_available(self) -> bool:
        """Whether the provider is healthy and can be polled.

        Returns ``True`` as long as the provider has not been closed.
        This matches ``UDPBVHInputProvider`` semantics where
        ``is_available()`` means "provider is alive" (not "data ready
        right now").  The actual blocking-until-data logic lives in
        ``get_frame()``.
        """
        return not self._closed

    def get_frame(self) -> HumanFrame:
        """Get the current body tracking frame.

        On the first call, blocks up to ``timeout`` seconds waiting for
        body data so the headset has time to connect.  After data has
        been received at least once, uses a short timeout (100 ms) so
        that a mid-session tracking drop is detected within a few
        control cycles instead of stalling the loop for the full
        ``timeout`` duration.

        Returns:
            Dict mapping joint name to (position_3d, quaternion_wxyz) tuples
            after Teleopit's input-space transform.

        Raises:
            TimeoutError: If no body data arrives within the timeout.
        """
        # Short timeout after first frame to avoid stalling the control loop
        wait = self._timeout if not self._has_received_data else 0.1
        deadline = time.monotonic() + wait
        while not self._xrt.is_body_data_available():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No Pico4 body data received within {wait:.1f}s timeout"
                )
            time.sleep(0.05)

        self._has_received_data = True
        body_poses = self._xrt.get_body_joints_pose()  # list of [x,y,z,qx,qy,qz,qw]

        body_pose_dict: dict[str, list] = {}
        for i, joint_name in enumerate(BODY_JOINT_NAMES):
            pos = [body_poses[i][0], body_poses[i][1], body_poses[i][2]]
            # SDK returns [x,y,z,qx,qy,qz,qw] — convert to scalar-first [w,x,y,z]
            rot = [body_poses[i][6], body_poses[i][3], body_poses[i][4], body_poses[i][5]]
            body_pose_dict[joint_name] = [pos, rot]

        body_pose_dict = _coordinate_transform_input(body_pose_dict)

        # Convert lists to numpy arrays for downstream compatibility
        result: HumanFrame = {}
        for name, (pos, quat) in body_pose_dict.items():
            result[name] = (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))

        return result

    def close(self) -> None:
        """Mark provider as closed."""
        self._closed = True
        logger.info("Pico4InputProvider closed")
