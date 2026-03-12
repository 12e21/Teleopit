"""Smooth transition interpolator for retargeted 36D qpos.

Prevents violent robot motion when switching from default/idle pose to a new
motion command by gradually blending between the start pose and the live
retargeted target over a configurable duration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _slerp(q0: NDArray, q1: NDArray, t: float) -> NDArray:
    """Spherical linear interpolation between two wxyz quaternions."""
    q0 = q0 / max(np.linalg.norm(q0), 1e-8)
    q1 = q1 / max(np.linalg.norm(q1), 1e-8)
    dot = float(np.dot(q0, q1))
    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    # Fall back to lerp for nearly identical quaternions
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / max(np.linalg.norm(result), 1e-8)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


class QposInterpolator:
    """Smoothly interpolates retargeted qpos from a start pose to the live target.

    Operates on 36D qpos: pos(3) + quat_wxyz(4) + joints(29).
    Position and joints use linear interpolation; quaternion uses SLERP.

    Parameters
    ----------
    duration : float
        Transition duration in seconds. 0.0 disables interpolation.
    policy_hz : float
        Policy frequency (steps per second) for step-based progress.
    """

    def __init__(self, duration: float, policy_hz: float) -> None:
        self._duration = max(duration, 0.0)
        self._policy_hz = policy_hz
        self._total_steps = int(self._duration * policy_hz)
        self._step = 0
        self._start_qpos: NDArray | None = None
        self._active = False

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, start_qpos: NDArray) -> None:
        """Begin interpolation from *start_qpos* toward future targets."""
        if self._total_steps <= 0:
            return
        self._start_qpos = np.array(start_qpos, dtype=np.float64).ravel()
        self._step = 0
        self._active = True

    def apply(self, target_qpos: NDArray) -> NDArray:
        """Return interpolated qpos.  Passthrough when inactive or finished."""
        if not self._active or self._start_qpos is None:
            return target_qpos

        if self._step >= self._total_steps:
            self._active = False
            return target_qpos

        alpha = self._step / self._total_steps
        self._step += 1

        result = np.empty_like(target_qpos)
        # Position: lerp
        result[0:3] = (1.0 - alpha) * self._start_qpos[0:3] + alpha * target_qpos[0:3]
        # Quaternion: SLERP
        result[3:7] = _slerp(self._start_qpos[3:7], target_qpos[3:7], alpha)
        # Joints: lerp
        result[7:] = (1.0 - alpha) * self._start_qpos[7:] + alpha * target_qpos[7:]
        return result
