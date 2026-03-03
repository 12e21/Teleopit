"""Unitree wireless remote controller parser.

Parses the 40-byte ``LowState_.wireless_remote`` payload from the Unitree
SDK2 into button states (with edge detection) and joystick axes.

Protocol reference: unitree_sdk2py/utils/joystick.py bit-mask definitions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class Button:
    """Button state with edge detection."""

    pressed: bool = False
    on_pressed: bool = False   # rising edge (just pressed this cycle)
    on_released: bool = False  # falling edge (just released this cycle)


# Bit masks for the 16-bit key field (bytes 2-3, little-endian uint16).
_KEY_R1 = 0x0001
_KEY_L1 = 0x0002
_KEY_START = 0x0004
_KEY_SELECT = 0x0008  # back
_KEY_R2 = 0x0010
_KEY_L2 = 0x0020
_KEY_F1 = 0x0040
_KEY_F2 = 0x0080
_KEY_A = 0x0100
_KEY_B = 0x0200
_KEY_X = 0x0400
_KEY_Y = 0x0800
_KEY_UP = 0x1000
_KEY_RIGHT = 0x2000
_KEY_DOWN = 0x4000
_KEY_LEFT = 0x8000

_BUTTON_MAP: dict[str, int] = {
    "R1": _KEY_R1,
    "L1": _KEY_L1,
    "start": _KEY_START,
    "select": _KEY_SELECT,
    "R2": _KEY_R2,
    "L2": _KEY_L2,
    "F1": _KEY_F1,
    "F2": _KEY_F2,
    "A": _KEY_A,
    "B": _KEY_B,
    "X": _KEY_X,
    "Y": _KEY_Y,
    "up": _KEY_UP,
    "right": _KEY_RIGHT,
    "down": _KEY_DOWN,
    "left": _KEY_LEFT,
}


def _apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value


class UnitreeRemote:
    """Parse Unitree wireless remote controller protocol.

    Call :meth:`update` once per control cycle with the raw 40-byte
    ``wireless_remote`` field from ``LowState_``.

    Buttons
    -------
    LB (L1), RB (R1), LT (L2), RT (R2),
    A, B, X, Y, start, back (select), F1, F2

    Joysticks
    ---------
    lx, ly (left stick), rx, ry (right stick) — range [-1, 1].
    """

    def __init__(self, deadzone: float = 0.05) -> None:
        self._deadzone = deadzone

        # Internal button states keyed by name
        self._buttons: dict[str, Button] = {
            name: Button() for name in _BUTTON_MAP
        }
        # Previous key bitmask for edge detection
        self._prev_keys: int = 0

        # Joystick axes (raw float values from protocol)
        self._lx: float = 0.0
        self._ly: float = 0.0
        self._rx: float = 0.0
        self._ry: float = 0.0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, wireless_remote_bytes: bytes) -> None:
        """Parse 40-byte remote payload and update all states.

        Wireless remote protocol layout (little-endian):
        - bytes[0:2]   : head (uint16)
        - bytes[2:4]   : keys (uint16, button bitmask)
        - bytes[4:8]   : lx (float32)
        - bytes[8:12]  : rx (float32)
        - bytes[12:16] : ry (float32)
        - bytes[16:20] : L2 (float32)
        - bytes[20:24] : ly (float32)
        - remaining     : reserved
        """
        if len(wireless_remote_bytes) < 40:
            return

        keys = struct.unpack_from("<H", wireless_remote_bytes, 2)[0]
        raw_lx = struct.unpack_from("<f", wireless_remote_bytes, 4)[0]
        raw_rx = struct.unpack_from("<f", wireless_remote_bytes, 8)[0]
        raw_ry = struct.unpack_from("<f", wireless_remote_bytes, 12)[0]
        raw_ly = struct.unpack_from("<f", wireless_remote_bytes, 20)[0]

        # Update button states with edge detection
        for name, mask in _BUTTON_MAP.items():
            cur = bool(keys & mask)
            prev = bool(self._prev_keys & mask)
            btn = self._buttons[name]
            btn.pressed = cur
            btn.on_pressed = cur and not prev
            btn.on_released = not cur and prev

        self._prev_keys = keys

        # Joystick axes with deadzone
        self._lx = _apply_deadzone(raw_lx, self._deadzone)
        self._ly = _apply_deadzone(raw_ly, self._deadzone)
        self._rx = _apply_deadzone(raw_rx, self._deadzone)
        self._ry = _apply_deadzone(raw_ry, self._deadzone)

    # ------------------------------------------------------------------
    # Button properties (friendly aliases)
    # ------------------------------------------------------------------

    @property
    def start(self) -> Button:
        return self._buttons["start"]

    @property
    def back(self) -> Button:
        return self._buttons["select"]

    @property
    def A(self) -> Button:
        return self._buttons["A"]

    @property
    def B(self) -> Button:
        return self._buttons["B"]

    @property
    def X(self) -> Button:
        return self._buttons["X"]

    @property
    def Y(self) -> Button:
        return self._buttons["Y"]

    @property
    def LB(self) -> Button:
        return self._buttons["L1"]

    @property
    def RB(self) -> Button:
        return self._buttons["R1"]

    @property
    def LT(self) -> Button:
        return self._buttons["L2"]

    @property
    def RT(self) -> Button:
        return self._buttons["R2"]

    @property
    def F1(self) -> Button:
        return self._buttons["F1"]

    @property
    def F2(self) -> Button:
        return self._buttons["F2"]

    # ------------------------------------------------------------------
    # Joystick properties (with deadzone applied)
    # ------------------------------------------------------------------

    @property
    def lx(self) -> float:
        """Left stick X axis (left/right)."""
        return self._lx

    @property
    def ly(self) -> float:
        """Left stick Y axis (forward/backward)."""
        return self._ly

    @property
    def rx(self) -> float:
        """Right stick X axis (left/right)."""
        return self._rx

    @property
    def ry(self) -> float:
        """Right stick Y axis (up/down)."""
        return self._ry
