"""Quaternion utilities for coordinate transforms."""

import numpy as np


def quat_mul_np(x, y, scalar_first=True):
    """Quaternion multiplication on arrays of quaternions.

    Args:
        x: quaternion(s) of shape (..., 4)
        y: quaternion(s) of shape (..., 4)
        scalar_first: True if quaternions are in [w, x, y, z] format

    Returns:
        Quaternion product in the same format.
    """
    if not scalar_first:
        x = x[..., [3, 0, 1, 2]]
        y = y[..., [3, 0, 1, 2]]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
        x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
        x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
        x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0,
    ], axis=-1)

    if not scalar_first:
        res = res[..., [1, 2, 3, 0]]

    return res
