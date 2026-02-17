"""Tests for teleopit.bus — InProcessBus pub/sub, zero-copy, get_latest."""
import numpy as np

from teleopit.bus.in_process import InProcessBus
from teleopit.bus.topics import TOPIC_ACTION, TOPIC_ROBOT_STATE


class TestInProcessBusPublishSubscribe:
    """Pub/sub basics."""

    def test_publish_and_get_latest(self):
        bus = InProcessBus()
        bus.publish("test_topic", 42)
        assert bus.get_latest("test_topic") == 42

    def test_get_latest_returns_none_for_unknown_topic(self):
        bus = InProcessBus()
        assert bus.get_latest("nonexistent") is None

    def test_subscribe_callback_fires(self):
        bus = InProcessBus()
        received = []
        bus.subscribe("evt", lambda d: received.append(d))
        bus.publish("evt", "hello")
        assert received == ["hello"]

    def test_multiple_subscribers(self):
        bus = InProcessBus()
        a, b = [], []
        bus.subscribe("t", lambda d: a.append(d))
        bus.subscribe("t", lambda d: b.append(d))
        bus.publish("t", 99)
        assert a == [99] and b == [99]


class TestInProcessBusZeroCopy:
    """Verify zero-copy semantics (object identity preserved)."""

    def test_numpy_array_identity(self):
        bus = InProcessBus()
        arr = np.array([1.0, 2.0, 3.0])
        bus.publish("arr", arr)
        retrieved = bus.get_latest("arr")
        assert retrieved is arr  # same object, zero-copy

    def test_dict_identity(self):
        bus = InProcessBus()
        data = {"key": "value"}
        bus.publish("d", data)
        assert bus.get_latest("d") is data


class TestInProcessBusUnsubscribe:
    """Unsubscribe behavior."""

    def test_unsubscribe_stops_callback(self):
        bus = InProcessBus()
        received = []
        cb = lambda d: received.append(d)
        bus.subscribe("t", cb)
        bus.publish("t", 1)
        bus.unsubscribe("t", cb)
        bus.publish("t", 2)
        assert received == [1]

    def test_unsubscribe_nonexistent_is_noop(self):
        bus = InProcessBus()
        bus.unsubscribe("no_topic", lambda d: None)  # should not raise


class TestTopicConstants:
    """Topic string constants exist and are non-empty."""

    def test_topic_constants_are_strings(self):
        assert isinstance(TOPIC_ACTION, str) and len(TOPIC_ACTION) > 0
        assert isinstance(TOPIC_ROBOT_STATE, str) and len(TOPIC_ROBOT_STATE) > 0
