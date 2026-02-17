from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .gmr import GeneralMotionRetargeting

Float64Array = NDArray[np.float64]
HumanFrame = dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]


def _quat_conjugate(quat: Float64Array) -> Float64Array:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(a: Float64Array, b: Float64Array) -> Float64Array:
    aw = float(a[0])
    ax = float(a[1])
    ay = float(a[2])
    az = float(a[3])
    bw = float(b[0])
    bx = float(b[1])
    by = float(b[2])
    bz = float(b[3])
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_rotate_inverse(quat: Float64Array, vec: Float64Array) -> Float64Array:
    q_inv = _quat_conjugate(quat)
    pure = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float64)
    rotated = _quat_multiply(_quat_multiply(q_inv, pure), quat)
    return rotated[1:4]


def _quat_to_euler(quat: Float64Array) -> tuple[float, float, float]:
    qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = float(np.sign(sinp) * (np.pi / 2.0))
    else:
        pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def extract_mimic_obs(
    qpos: Float64Array,
    last_qpos: Float64Array | None = None,
    dt: float = 1.0 / 50.0,
) -> NDArray[np.float32]:
    if qpos.ndim != 1:
        raise ValueError("qpos must be a 1D array")
    if qpos.shape[0] < 36:
        raise ValueError("qpos must contain 7D root + 29D joints")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    current: Float64Array = np.asarray(qpos, dtype=np.float64)
    previous: Float64Array = current if last_qpos is None else np.asarray(last_qpos, dtype=np.float64)
    if previous.shape != current.shape:
        raise ValueError("last_qpos shape must match qpos shape")

    root_pos = current[0:3]
    last_root_pos = previous[0:3]
    root_quat = current[3:7]
    last_root_quat = previous[3:7]
    joints = current[7:36]

    base_vel_world = (root_pos - last_root_pos) / dt
    quat_delta: Float64Array = _quat_multiply(root_quat, _quat_conjugate(last_root_quat))
    quat_delta /= max(np.linalg.norm(quat_delta), 1e-8)
    yaw_delta = 2.0 * np.arctan2(np.linalg.norm(quat_delta[1:4]), max(quat_delta[0], 1e-8))
    base_ang_vel_world = np.array([0.0, 0.0, yaw_delta / dt], dtype=np.float64)

    base_vel_local = _quat_rotate_inverse(root_quat, base_vel_world)
    base_ang_vel_local = _quat_rotate_inverse(root_quat, base_ang_vel_world)
    roll, pitch, _ = _quat_to_euler(root_quat)

    mimic_obs = np.concatenate(
        (
            base_vel_local[:2],
            root_pos[2:3],
            np.array([roll, pitch], dtype=np.float64),
            base_ang_vel_local[2:3],
            joints,
        )
    )
    if mimic_obs.shape[0] != 35:
        raise ValueError(f"Expected 35D mimic obs, got {mimic_obs.shape[0]}")
    return mimic_obs.astype(np.float32, copy=False)


class RetargetingModule:
    def __init__(
        self,
        robot_name: str,
        human_format: str,
        actual_human_height: float | None = None,
        solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = False,
        use_velocity_limit: bool = False,
    ) -> None:
        if actual_human_height is None:
            self._gmr = GeneralMotionRetargeting(
                src_human=human_format,
                tgt_robot=robot_name,
                solver=solver,
                damping=damping,
                verbose=verbose,
                use_velocity_limit=use_velocity_limit,
            )
        else:
            self._gmr = GeneralMotionRetargeting(
                src_human=human_format,
                tgt_robot=robot_name,
                actual_human_height=actual_human_height,
                solver=solver,
                damping=damping,
                verbose=verbose,
                use_velocity_limit=use_velocity_limit,
            )

    def retarget(self, human_data: HumanFrame) -> Float64Array:
        qpos = cast(Float64Array, self._gmr.retarget(human_data))
        return np.asarray(qpos, dtype=np.float64)
