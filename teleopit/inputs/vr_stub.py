"""VR input stub for testing without real VR hardware."""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class VRInputStub:
    """Stub VR input provider that returns static T-pose data."""
    
    def __init__(self):
        """Initialize with static T-pose data."""
        # Static T-pose: standing human with arms extended
        self._frame_data = {
            'pelvis': (np.array([0.0, 0.0, 1.0], dtype=np.float32), 
                      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            'torso': (np.array([0.0, 0.0, 1.2], dtype=np.float32),
                     np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            'head': (np.array([0.0, 0.0, 1.6], dtype=np.float32),
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            'left_shoulder': (np.array([-0.2, 0.0, 1.4], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            'right_shoulder': (np.array([0.2, 0.0, 1.4], dtype=np.float32),
                             np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            'left_hand': (np.array([-0.6, 0.0, 1.4], dtype=np.float32),
                         np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            'right_hand': (np.array([0.6, 0.0, 1.4], dtype=np.float32),
                          np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        }
    
    def get_frame(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return static T-pose frame data.
        
        Returns:
            Dict mapping body part names to (position, quaternion) tuples.
        """
        return self._frame_data
    
    def is_available(self) -> bool:
        """Check if input is available (always True for stub)."""
        return True
