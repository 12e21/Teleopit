"""MuJoCo FK helpers for motion dataset generation and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


DEFAULT_G1_XML_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "g1" / "g1_sim2sim_29dof.xml"
)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from xyzw to wxyz convention."""
    return np.asarray(q)[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from wxyz to xyzw convention."""
    return np.asarray(q)[..., [1, 2, 3, 0]]


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion array in wxyz convention."""
    arr = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return arr / norms


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions: q1 * q2."""
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a wxyz quaternion."""
    q = np.asarray(q, dtype=np.float32)
    conj = q.copy()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by quaternion(s) ``q`` in wxyz convention."""
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=np.float32)
    v_quat[..., 1:4] = v
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[..., 1:4]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by the inverse of quaternion(s) ``q``."""
    return quat_rotate(quat_conjugate(q), v)


def quat_to_angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from quaternion sequence in wxyz convention."""
    q = normalize_quaternion(q)
    q_dot = np.gradient(q, dt, axis=0)
    product = quat_multiply(q_dot, quat_conjugate(q))
    return (2.0 * product[..., 1:4]).astype(np.float32)


def quaternion_angle_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return angle error in radians between wxyz quaternion arrays."""
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


@dataclass(frozen=True)
class MotionConsistencyStats:
    frames_checked: int
    bodies_checked: int
    pos_mean: float
    pos_p95: float
    pos_max: float
    quat_mean: float
    quat_p95: float
    quat_max: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class MotionFkExtractor:
    """Extract body world poses from root pose + joint positions via MuJoCo FK."""

    def __init__(self, xml_path: str | Path | None = None) -> None:
        resolved_xml = Path(xml_path) if xml_path is not None else DEFAULT_G1_XML_PATH
        resolved_xml = resolved_xml.expanduser().resolve()
        if not resolved_xml.is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {resolved_xml}")

        self.xml_path = resolved_xml
        self.model = mujoco.MjModel.from_xml_path(str(resolved_xml))
        self.data = mujoco.MjData(self.model)
        self.num_actions = int(self.model.nq - 7)

    def _body_ids(self, body_names: Iterable[str]) -> list[int]:
        ids: list[int] = []
        missing: list[str] = []
        for name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                missing.append(name)
            else:
                ids.append(body_id)
        if missing:
            raise ValueError(
                f"Body names not found in MuJoCo model {self.xml_path}: {sorted(missing)}"
            )
        return ids

    def extract(
        self,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        joint_pos: np.ndarray,
        body_names: Iterable[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        root_pos = np.asarray(root_pos, dtype=np.float32)
        root_quat_wxyz = normalize_quaternion(root_quat_wxyz)
        joint_pos = np.asarray(joint_pos, dtype=np.float32)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must be (T,3), got {root_pos.shape}")
        if root_quat_wxyz.ndim != 2 or root_quat_wxyz.shape[1] != 4:
            raise ValueError(f"root_quat_wxyz must be (T,4), got {root_quat_wxyz.shape}")
        if joint_pos.ndim != 2 or joint_pos.shape[1] != self.num_actions:
            raise ValueError(
                f"joint_pos must be (T,{self.num_actions}), got {joint_pos.shape}"
            )
        if not (root_pos.shape[0] == root_quat_wxyz.shape[0] == joint_pos.shape[0]):
            raise ValueError(
                "root_pos, root_quat_wxyz, and joint_pos must have identical time dimensions"
            )

        body_name_list = [str(name) for name in body_names]
        body_ids = self._body_ids(body_name_list)
        num_frames = int(root_pos.shape[0])
        num_bodies = len(body_ids)

        body_pos_w = np.empty((num_frames, num_bodies, 3), dtype=np.float32)
        body_quat_w = np.empty((num_frames, num_bodies, 4), dtype=np.float32)

        for frame_idx in range(num_frames):
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.qpos[:3] = root_pos[frame_idx]
            self.data.qpos[3:7] = root_quat_wxyz[frame_idx]
            self.data.qpos[7 : 7 + self.num_actions] = joint_pos[frame_idx]
            mujoco.mj_forward(self.model, self.data)
            body_pos_w[frame_idx] = self.data.xpos[body_ids]
            body_quat_w[frame_idx] = self.data.xquat[body_ids]

        return body_pos_w, normalize_quaternion(body_quat_w)


def compute_body_velocities(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute linear and angular velocity sequences from FK outputs."""
    body_pos_w = np.asarray(body_pos_w, dtype=np.float32)
    body_quat_w = normalize_quaternion(body_quat_w)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    body_ang_vel_w = quat_to_angular_velocity(body_quat_w, dt).astype(np.float32)
    return body_lin_vel_w, body_ang_vel_w


def _sample_frame_indices(total_frames: int, sample_count: int, seed: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be > 0, got {total_frames}")
    if sample_count <= 0 or sample_count >= total_frames:
        return np.arange(total_frames, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total_frames, size=sample_count, replace=False))
    return idx.astype(np.int64)


def compute_npz_fk_consistency(
    npz_path: str | Path,
    *,
    model_path: str | Path | None = None,
    sample_count: int = 32,
    seed: int = 0,
) -> MotionConsistencyStats:
    """Compare NPZ body poses against MuJoCo FK derived from the same qpos."""
    from train_mimic.data.dataset_lib import inspect_npz

    path = Path(npz_path).expanduser().resolve()
    inspect_npz(path)
    data = np.load(path, allow_pickle=True)

    body_names = [str(name) for name in np.asarray(data["body_names"])]
    if "pelvis" not in body_names:
        raise ValueError(f"NPZ body_names missing required root body 'pelvis': {path}")

    pelvis_idx = body_names.index("pelvis")
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w = normalize_quaternion(np.asarray(data["body_quat_w"], dtype=np.float32))

    frame_indices = _sample_frame_indices(int(joint_pos.shape[0]), sample_count, seed)
    root_pos = body_pos_w[frame_indices, pelvis_idx, :]
    root_quat_wxyz = body_quat_w[frame_indices, pelvis_idx, :]

    extractor = MotionFkExtractor(model_path)
    fk_body_pos_w, fk_body_quat_w = extractor.extract(
        root_pos,
        root_quat_wxyz,
        joint_pos[frame_indices],
        body_names,
    )

    pos_err = np.linalg.norm(fk_body_pos_w - body_pos_w[frame_indices], axis=-1)
    quat_err = quaternion_angle_error(fk_body_quat_w, body_quat_w[frame_indices])

    return MotionConsistencyStats(
        frames_checked=int(frame_indices.shape[0]),
        bodies_checked=int(len(body_names)),
        pos_mean=float(pos_err.mean()),
        pos_p95=float(np.percentile(pos_err, 95)),
        pos_max=float(pos_err.max()),
        quat_mean=float(quat_err.mean()),
        quat_p95=float(np.percentile(quat_err, 95)),
        quat_max=float(quat_err.max()),
    )

