from __future__ import annotations

import unittest

import numpy as np

from teleopit.sim.runtime_components import PolicyStepRunner


class _Controller:
    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target)

    def get_target_dof_pos(self, raw_action: np.ndarray) -> np.ndarray:
        del raw_action
        return self.target.copy()


class _Joint:
    def __init__(self, name: str) -> None:
        self.name = name


class _Model:
    def __init__(
        self,
        joint_ids: list[int],
        qpos_addresses: list[int],
        limited: list[bool],
        ranges: list[tuple[float, float]],
    ) -> None:
        self.nu = len(joint_ids)
        self.njnt = len(qpos_addresses)
        self.nq = max(qpos_addresses, default=-1) + 1
        self.actuator_trnid = np.column_stack(
            (np.asarray(joint_ids, dtype=np.int32), -np.ones(len(joint_ids), dtype=np.int32))
        )
        self.jnt_qposadr = np.asarray(qpos_addresses, dtype=np.int32)
        self.jnt_limited = np.asarray(limited, dtype=np.uint8)
        self.jnt_range = np.asarray(ranges, dtype=np.float64)

    def joint(self, joint_id: int) -> _Joint:
        return _Joint(f"joint_{joint_id}")


class _Robot:
    def __init__(self, model: _Model | None = None) -> None:
        if model is not None:
            self.model = model


def _make_runner(robot: object, target: np.ndarray, num_actions: int) -> PolicyStepRunner:
    zeros = np.zeros(num_actions, dtype=np.float32)
    return PolicyStepRunner(
        robot=robot,
        controller=_Controller(target),
        obs_builder=object(),
        policy_hz=50.0,
        decimation=4,
        num_actions=num_actions,
        kps=zeros.copy(),
        kds=zeros.copy(),
        torque_limits=np.ones(num_actions, dtype=np.float32),
        default_dof_pos=zeros.copy(),
    )


def _valid_model() -> _Model:
    return _Model(
        joint_ids=[0, 1, 2],
        qpos_addresses=[0, 1, 2],
        limited=[True, False, True],
        ranges=[(-1.0, 1.0), (-0.25, 0.25), (-2.0, 2.0)],
    )


class JointLimitGuardTests(unittest.TestCase):
    def test_robot_without_model_is_pass_through(self) -> None:
        target = np.asarray([-3.0, 4.0], dtype=np.float32)
        runner = _make_runner(_Robot(), target, 2)
        actual = runner.compute_target_dof_pos(np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(actual, target)
        np.testing.assert_array_equal(runner.last_joint_limit_clip_mask, [False, False])

    def test_limited_joints_clip_both_directions_and_unlimited_does_not(self) -> None:
        runner = _make_runner(_Robot(_valid_model()), np.asarray([-3.0, 9.0, 4.0]), 3)
        actual = runner.compute_target_dof_pos(np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(actual, [-1.0, 9.0, 2.0])
        np.testing.assert_array_equal(runner.last_joint_limit_clip_mask, [True, False, True])
        np.testing.assert_array_equal(runner.last_joint_limit_correction, [2.0, 0.0, -2.0])
        self.assertEqual(runner.last_joint_limit_max_correction, 2.0)

    def test_diagnostics_and_output_are_independent_copies(self) -> None:
        runner = _make_runner(_Robot(_valid_model()), np.asarray([-3.0, 0.0, 4.0]), 3)
        output = runner.compute_target_dof_pos(np.zeros(3, dtype=np.float32))
        output[:] = 123.0
        original = runner.last_original_target
        guarded = runner.last_guarded_target
        mask = runner.last_joint_limit_clip_mask
        correction = runner.last_joint_limit_correction
        assert original is not None and guarded is not None and mask is not None and correction is not None
        original[:] = 77.0
        guarded[:] = 77.0
        mask[:] = False
        correction[:] = 77.0
        np.testing.assert_array_equal(runner.last_original_target, [-3.0, 0.0, 4.0])
        np.testing.assert_array_equal(runner.last_guarded_target, [-1.0, 0.0, 2.0])
        np.testing.assert_array_equal(runner.last_joint_limit_clip_mask, [True, False, True])
        np.testing.assert_array_equal(runner.last_joint_limit_correction, [2.0, 0.0, -2.0])

    def test_dtype_shape_finiteness_and_valid_mapping(self) -> None:
        runner = _make_runner(_Robot(_valid_model()), np.asarray([0.5, 0.0, -0.5]), 3)
        actual = runner.compute_target_dof_pos(np.zeros(3, dtype=np.float32))
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (3,))
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_array_equal(runner.last_joint_limit_clip_mask, [False, False, False])
        self.assertEqual(runner.joint_limit_joint_names, ("joint_0", "joint_1", "joint_2"))

    def test_reset_clears_diagnostics(self) -> None:
        runner = _make_runner(_Robot(_valid_model()), np.asarray([-3.0, 0.0, 4.0]), 3)
        runner.compute_target_dof_pos(np.zeros(3, dtype=np.float32))
        runner.reset()
        self.assertIsNone(runner.last_original_target)
        self.assertIsNone(runner.last_guarded_target)
        self.assertIsNone(runner.last_joint_limit_clip_mask)
        self.assertIsNone(runner.last_joint_limit_correction)
        self.assertEqual(runner.last_joint_limit_max_correction, 0.0)

    def test_finish_step_preserves_raw_action_semantics(self) -> None:
        runner = _make_runner(_Robot(_valid_model()), np.asarray([-3.0, 0.0, 4.0]), 3)
        raw_action = np.asarray([0.3, -0.4, 0.5], dtype=np.float32)
        runner.compute_target_dof_pos(raw_action)
        runner.finish_step(raw_action, np.zeros(10, dtype=np.float64))
        np.testing.assert_array_equal(runner.last_action, raw_action)
        np.testing.assert_array_equal(runner.last_guarded_target, [-1.0, 0.0, 2.0])

    def test_duplicate_joint_mapping_raises(self) -> None:
        model = _Model([0, 0], [0, 1], [True, True], [(-1.0, 1.0), (-1.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "repeats joint id"):
            _make_runner(_Robot(model), np.zeros(2), 2)

    def test_invalid_joint_id_raises(self) -> None:
        model = _Model([0, 2], [0, 1], [True, True], [(-1.0, 1.0), (-1.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "invalid joint id"):
            _make_runner(_Robot(model), np.zeros(2), 2)

    def test_invalid_qpos_address_raises(self) -> None:
        model = _valid_model()
        model.jnt_qposadr[1] = model.nq
        with self.assertRaisesRegex(ValueError, "invalid qpos address"):
            _make_runner(_Robot(model), np.zeros(3), 3)

    def test_too_few_actuators_raises(self) -> None:
        model = _Model([0], [0], [True], [(-1.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "expected at least 2"):
            _make_runner(_Robot(model), np.zeros(2), 2)


if __name__ == "__main__":
    unittest.main()
