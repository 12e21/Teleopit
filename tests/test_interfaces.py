"""Tests for teleopit.interfaces — Protocol imports and structural subtyping."""
import numpy as np

from teleopit.interfaces import (
    Controller,
    InputProvider,
    MessageBus,
    ObservationBuilder,
    Recorder,
    Retargeter,
    Robot,
    RobotState,
)


# ── RobotState dataclass ───────────────────────────────────────

class TestRobotState:
    def test_creation(self):
        state = RobotState(
            qpos=np.zeros(5),
            qvel=np.ones(5),
            quat=np.array([1, 0, 0, 0], dtype=np.float64),
            ang_vel=np.zeros(3),
            timestamp=1.23,
        )
        assert state.timestamp == 1.23
        assert state.qpos.shape == (5,)
        assert state.quat.shape == (4,)

    def test_fields_are_mutable(self):
        state = RobotState(
            qpos=np.zeros(3),
            qvel=np.zeros(3),
            quat=np.array([1, 0, 0, 0], dtype=np.float64),
            ang_vel=np.zeros(3),
            timestamp=0.0,
        )
        state.timestamp = 9.9
        assert state.timestamp == 9.9


# ── Structural subtyping checks ────────────────────────────────

class _FakeInputProvider:
    def get_frame(self):
        return {"body": [0, 0, 0]}

    def is_available(self):
        return True


class _FakeRetargeter:
    def retarget(self, human_data):
        n = 5
        return np.zeros(3), np.zeros(4), np.zeros(n)


class _FakeController:
    def compute_action(self, obs):
        return obs * 0.1

    def reset(self):
        pass


class _FakeRobot:
    def get_state(self):
        return RobotState(
            qpos=np.zeros(3), qvel=np.zeros(3),
            quat=np.array([1, 0, 0, 0]), ang_vel=np.zeros(3), timestamp=0.0,
        )

    def set_action(self, action):
        pass

    def step(self):
        pass


class _FakeMessageBus:
    def publish(self, topic, data):
        pass

    def subscribe(self, topic):
        return None


class _FakeRecorder:
    def record_step(self, data):
        pass

    def save(self, path):
        pass


class _FakeObservationBuilder:
    def build_observation(self, state, history, action_mimic):
        return np.zeros(10)


class TestProtocolSubtyping:
    """Verify runtime_checkable protocols accept structural subtypes."""

    def test_input_provider_isinstance(self):
        assert isinstance(_FakeInputProvider(), InputProvider)

    def test_retargeter_isinstance(self):
        assert isinstance(_FakeRetargeter(), Retargeter)

    def test_controller_isinstance(self):
        assert isinstance(_FakeController(), Controller)

    def test_robot_isinstance(self):
        assert isinstance(_FakeRobot(), Robot)

    def test_message_bus_isinstance(self):
        assert isinstance(_FakeMessageBus(), MessageBus)

    def test_recorder_isinstance(self):
        assert isinstance(_FakeRecorder(), Recorder)

    def test_observation_builder_isinstance(self):
        assert isinstance(_FakeObservationBuilder(), ObservationBuilder)

    def test_plain_object_not_protocol(self):
        """A plain object should NOT satisfy any protocol."""
        obj = object()
        assert not isinstance(obj, InputProvider)
        assert not isinstance(obj, Robot)
        assert not isinstance(obj, Controller)
