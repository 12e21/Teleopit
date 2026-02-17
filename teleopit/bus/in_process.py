"""In-process message bus with zero-copy pub/sub."""

from typing import Any, Callable


class InProcessBus:
    """Zero-copy message bus using Python object references."""

    def __init__(self) -> None:
        """Initialize the message bus."""
        self._subscribers: dict[str, list[Callable[[Any], None]]] = {}
        self._latest: dict[str, Any] = {}

    def publish(self, topic: str, data: Any) -> None:
        """Publish data to a topic (zero-copy).

        Args:
            topic: Topic name
            data: Data to publish (stored by reference)
        """
        self._latest[topic] = data
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                callback(data)

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to a topic.

        Args:
            topic: Topic name
            callback: Function to call when data is published
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Unsubscribe from a topic.

        Args:
            topic: Topic name
            callback: Function to remove from subscribers
        """
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(callback)
            except ValueError:
                pass

    def get_latest(self, topic: str) -> Any | None:
        """Get the latest message for a topic.

        Args:
            topic: Topic name

        Returns:
            Latest data or None if no data published yet
        """
        return self._latest.get(topic)
