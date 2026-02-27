# pyright: reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false

import numpy as np
import torch
from typing import cast

from teleopit_train.pose.utils import isaacgym_torch_utils as ref
from teleopit_train.utils import math_utils as uut


def _as_tensor(x: object) -> torch.Tensor:
    return cast(torch.Tensor, x)


def _rand_unit_quat(batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
    q = torch.randn(batch, 4, device=device)
    return cast(torch.Tensor, q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9))


def test_quat_mul_equivalence() -> None:
    q1 = _rand_unit_quat(128)
    q2 = _rand_unit_quat(128)
    out_ref = _as_tensor(ref.quat_mul(q1, q2))
    out_uut = uut.quat_mul(q1, q2)
    assert torch.allclose(out_uut, out_ref, atol=1e-6)


def test_quat_apply_equivalence() -> None:
    q = _rand_unit_quat(128)
    v = torch.randn(128, 3)
    out_ref = _as_tensor(ref.quat_apply(q, v))
    out_uut = uut.quat_apply(q, v)
    assert torch.allclose(out_uut, out_ref, atol=1e-6)


def test_quat_rotate_inverse_equivalence() -> None:
    q = _rand_unit_quat(128)
    v = torch.randn(128, 3)
    out_ref = _as_tensor(ref.quat_rotate_inverse(q, v))
    out_uut = uut.quat_rotate_inverse(q, v)
    assert torch.allclose(out_uut, out_ref, atol=1e-6)


def test_quat_rotate_inverse_known_value() -> None:
    q = torch.tensor([[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=torch.float32)
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    expected = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32)
    out = uut.quat_rotate_inverse(q, v)
    assert torch.allclose(out, expected, atol=1e-6)


def test_quat_from_euler_xyz_equivalence() -> None:
    roll = torch.randn(128)
    pitch = torch.randn(128)
    yaw = torch.randn(128)
    out_ref = _as_tensor(ref.quat_from_euler_xyz(roll, pitch, yaw))
    out_uut = uut.quat_from_euler_xyz(roll, pitch, yaw)
    assert torch.allclose(out_uut, out_ref, atol=1e-6)


def test_torch_rand_float_equivalence_with_seed() -> None:
    _ = torch.manual_seed(1234)
    out_ref = _as_tensor(ref.torch_rand_float(-1.25, 3.75, (64, 3), "cpu"))
    _ = torch.manual_seed(1234)
    out_uut = uut.torch_rand_float(-1.25, 3.75, (64, 3), "cpu")
    assert torch.allclose(out_uut, out_ref, atol=1e-6)


def test_to_torch_equivalence() -> None:
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    out_ref = ref.to_torch(data, dtype=torch.float64, device="cpu", requires_grad=False)
    out_uut = uut.to_torch(data, dtype=torch.float64, device="cpu", requires_grad=False)
    assert torch.equal(out_uut, out_ref)
    assert out_uut.dtype == torch.float64


def test_get_axis_params_equivalence() -> None:
    out_ref = ref.get_axis_params(2.5, axis_idx=2, x_value=-1.0, dtype=np.float64, n_dims=3)
    out_uut = uut.get_axis_params(2.5, axis=2, x_value=-1.0, dtype=np.float64, n_dims=3)
    assert np.allclose(out_uut, out_ref, atol=1e-6)
