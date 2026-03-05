"""Core abstract interfaces for Teleopit using typing.Protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any, Dict
import numpy as np


@dataclass
class RobotState:
    """Robot state snapshot."""
    qpos: np.ndarray  # Joint positions
    qvel: np.ndarray  # Joint velocities
    quat: np.ndarray  # Base orientation quaternion
    ang_vel: np.ndarray  # Base angular velocity
    timestamp: float  # Timestamp in seconds
    base_pos: np.ndarray | None = None  # Base position in world frame
    base_lin_vel: np.ndarray | None = None  # Base linear velocity in body frame


@runtime_checkable
class InputProvider(Protocol):
    """Provides human motion input from various sources."""
    
    def get_frame(self) -> Dict[str, Any]:
        """Get current frame of human motion data."""
        ...
    
    def is_available(self) -> bool:
        """Check if input source is available."""
        ...


@runtime_checkable
class Retargeter(Protocol):
    """Retargets human motion to robot motion."""
    
    def retarget(self, human_data: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retarget human motion to robot. Returns (base_pos, base_rot, joint_pos)."""
        ...


@runtime_checkable
class Controller(Protocol):
    """Generates robot actions from observations."""
    
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute action from observation."""
        ...
    
    def reset(self) -> None:
        """Reset controller state."""
        ...


@runtime_checkable
class Robot(Protocol):
    """Robot interface for state and control."""
    
    def get_state(self) -> RobotState:
        """Get current robot state."""
        ...
    
    def set_action(self, action: np.ndarray) -> None:
        """Set robot action."""
        ...
    
    def step(self) -> None:
        """Step robot simulation/hardware."""
        ...


@runtime_checkable
class MessageBus(Protocol):
    """Message bus for inter-component communication."""
    
    def publish(self, topic: str, data: Any) -> None:
        """Publish data to topic."""
        ...
    
    def subscribe(self, topic: str) -> Any:
        """Subscribe to topic and get latest data."""
        ...


@runtime_checkable
class Recorder(Protocol):
    """Records teleoperation data."""
    
    def record_step(self, data: Dict[str, Any]) -> None:
        """Record single step of data."""
        ...
    
    def save(self, path: str) -> None:
        """Save recorded data to file."""
        ...


@runtime_checkable
class ObservationBuilder(Protocol):
    """Builds observations for controller from robot state."""
    
    def build_observation(self, state: RobotState, history: list[np.ndarray], action_mimic: np.ndarray) -> np.ndarray:
        """Build observation from state, history, and mimic action."""
        ...
