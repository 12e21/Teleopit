from __future__ import annotations

from collections.abc import Sequence

import torch


def to_torch(
    x: object,
    dtype: torch.dtype = torch.float,
    device: torch.device | str = "cuda:0",
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([x, y, z, w], dim=-1).view(shape)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    xyz = q[:, :3]
    t = xyz.cross(v, dim=-1) * 2
    return (v + q[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to quaternion in IsaacGym/Isaac Lab `(x, y, z, w)` order."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


def torch_rand_float(
    lower: float,
    upper: float,
    shape: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def get_axis_params(
    value: float,
    axis: int | None = None,
    x_value: float = 0.0,
    dtype: type = float,
    n_dims: int = 3,
    axis_idx: int | None = None,
) -> list[float]:
    if axis is None:
        axis = axis_idx
    if axis is None:
        raise ValueError("either `axis` or `axis_idx` must be provided")
    params = [0.0] * n_dims
    assert axis < n_dims, "the axis dim should be within the vector dimensions"
    params[axis] = float(value)
    params[0] = float(x_value)
    return [dtype(p) for p in params]
