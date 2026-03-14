from __future__ import annotations

import numpy as np

from conftest import requires_mujoco
from teleopit.bus.in_process import InProcessBus
from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE
from teleopit.interfaces import RobotState


class _DummyRobot:
    def __init__(self) -> None:
        self.num_actions = 2
        self.kps = np.array([1.0, 1.0], dtype=np.float32)
        self.kds = np.array([0.1, 0.1], dtype=np.float32)
        self.torque_limits = np.array([10.0, 10.0], dtype=np.float32)
        self.default_dof_pos = np.array([0.5, -0.5], dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)
        self._qpos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._qvel = np.zeros(3, dtype=np.float32)
        self._timestamp = 0.0

    def get_state(self) -> RobotState:
        return RobotState(
            qpos=self._qpos.copy(),
            qvel=self._qvel.copy(),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.zeros(3, dtype=np.float32),
            timestamp=self._timestamp,
            base_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            base_lin_vel=np.zeros(3, dtype=np.float32),
        )

    def set_action(self, action: np.ndarray) -> None:
        self._last_action = np.asarray(action, dtype=np.float32)[:2]

    def step(self) -> None:
        self._qvel[:2] = self._last_action
        self._qpos[:2] += self._last_action * 0.01
        self._timestamp += 0.02


class _DummyController:
    _expected_obs_dim = 4

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.1, -0.1], dtype=np.float32)

    def reset(self) -> None:
        pass


class _DummyObsBuilder:
    def build(self, state: object, mimic_obs: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


class _DummyInputProvider:
    fps = 1

    def __init__(self) -> None:
        self._frames = 0

    def get_frame(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self._frames += 1
        return {"Pelvis": (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))}

    def is_available(self) -> bool:
        return self._frames < 3


class _DummyRetargeter:
    def retarget(self, human_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.zeros(29, dtype=np.float64),
        )


class _DummyRecorder:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def record_step(self, data: dict[str, object]) -> None:
        self.frames.append(data)


@requires_mujoco
def test_simulation_loop_runs_and_records_without_viewers() -> None:
    from teleopit.sim.loop import SimulationLoop

    bus = InProcessBus()
    robot = _DummyRobot()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=_DummyObsBuilder(),
        bus=bus,
        cfg={"policy_hz": 50.0, "pd_hz": 50.0, "realtime": False, "transition_duration": 0.0},
        viewers=set(),
    )

    recorder = _DummyRecorder()
    result = loop.run(
        input_provider=_DummyInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=2,
        recorder=recorder,
    )

    assert result["steps"] == 2
    assert len(recorder.frames) == 2

    latest_action = bus.get_latest(TOPIC_ACTION)
    latest_mimic = bus.get_latest(TOPIC_MIMIC_OBS)
    latest_state = bus.get_latest(TOPIC_ROBOT_STATE)

    assert isinstance(latest_action, np.ndarray)
    np.testing.assert_allclose(latest_action, np.array([0.1, -0.1], dtype=np.float32))
    assert isinstance(latest_mimic, np.ndarray)
    assert latest_mimic.shape == (35,)
    assert latest_state is not None

    target = np.asarray(recorder.frames[-1]["target_dof_pos"], dtype=np.float32)
    np.testing.assert_allclose(target, np.array([0.6, -0.6], dtype=np.float32))
