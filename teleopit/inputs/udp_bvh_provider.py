"""Real-time UDP BVH input provider.

Receives single-frame BVH motion data over UDP (one packet = one line of
BVH floats) and exposes the latest frame via ``get_frame()``.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.bvh_provider import _parse_bvh_header, process_single_bvh_frame
from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh

logger = logging.getLogger(__name__)


class UDPBVHInputProvider:
    """Receives BVH motion data in real-time over UDP.

    Each UDP packet is a whitespace-separated line of floats identical to one
    motion-data line in a BVH file.  A *reference* BVH file is used to parse
    the skeleton hierarchy (bone names, parents, offsets, Euler order).

    Parameters
    ----------
    reference_bvh : str
        Path to a BVH file whose header defines the skeleton.
    host : str
        Bind address for the UDP socket (default ``""`` = all interfaces).
    port : int
        UDP port number.
    human_format : str
        Format tag passed to the single-frame processor (e.g. ``"hc_mocap"``).
    timeout : float
        Seconds to wait for the first UDP packet before raising.
    """

    def __init__(
        self,
        reference_bvh: str,
        host: str = "",
        port: int = 1118,
        human_format: str = "hc_mocap",
        timeout: float = 30.0,
    ) -> None:
        # --- Parse skeleton metadata from reference BVH ---
        data = read_bvh(reference_bvh)
        self._bone_names: list[str] = list(data.bones)
        self._bone_parents: np.ndarray = np.array(data.parents, dtype=np.int32)
        self._offsets: np.ndarray = data.offsets.copy()

        self._euler_order, self._channels = _parse_bvh_header(reference_bvh)
        self._format = human_format
        # Auto-detect: hc_mocap without toe joints needs different IK config
        self._has_toe = "LeftToeBase" in self._bone_names
        if human_format == "hc_mocap" and not self._has_toe:
            self.human_format = "hc_mocap_no_toe"
        else:
            self.human_format = human_format

        # Coordinate transform matrices (Y-up BVH → Z-up sim)
        self._rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self._rotation_quat = R.from_matrix(self._rotation_matrix).as_quat(
            scalar_first=True
        )
        self._scale_divisor = 1.0 if human_format == "hc_mocap" else 100.0

        # Expected number of floats per packet
        n_bones = len(self._bone_names)
        if self._channels == 3:
            self._expected_floats = 3 + n_bones * 3  # root_pos + N*3 euler
        else:
            self._expected_floats = n_bones * self._channels

        # Compute human height from reference BVH frame-0 FK
        from teleopit.retargeting.gmr.utils.lafan_vendor import utils as _utils

        global_data = _utils.quat_fk(data.quats, data.pos, data.parents)
        head_idx = (
            self._bone_names.index("hc_Head")
            if "hc_Head" in self._bone_names
            else None
        )
        toe_l_idx = (
            self._bone_names.index("LeftToeBase")
            if "LeftToeBase" in self._bone_names
            else None
        )
        toe_r_idx = (
            self._bone_names.index("RightToeBase")
            if "RightToeBase" in self._bone_names
            else None
        )
        # Fallback to foot joints if toe joints not present
        if toe_l_idx is None and "hc_Foot_L" in self._bone_names:
            toe_l_idx = self._bone_names.index("hc_Foot_L")
        if toe_r_idx is None and "hc_Foot_R" in self._bone_names:
            toe_r_idx = self._bone_names.index("hc_Foot_R")
        if head_idx is not None and toe_l_idx is not None and toe_r_idx is not None:
            head_y = global_data[1][0, head_idx, 1]
            toe_y = min(
                global_data[1][0, toe_l_idx, 1], global_data[1][0, toe_r_idx, 1]
            )
            self._human_height = float(head_y - toe_y)
        else:
            self._human_height = 1.75
        if self._human_height < 0.5:
            self._human_height = 1.75

        # --- Thread-safe state ---
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._latest_frame: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None
        self._timeout = timeout
        self._running = True

        # --- UDP socket ---
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(2.0)  # allow periodic shutdown checks
        self._sock.bind((host, port))

        # --- Receiver thread ---
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info("UDPBVHInputProvider listening on %s:%d", host or "0.0.0.0", port)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return the latest received frame.

        The first call blocks until the first UDP packet arrives (up to
        ``timeout`` seconds).  Subsequent calls return immediately with the
        most recent frame.
        """
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No UDP BVH data received within {self._timeout}s"
            )
        with self._lock:
            assert self._latest_frame is not None
            return self._latest_frame

    def is_available(self) -> bool:
        """True while the receiver thread is alive."""
        return self._running and self._thread.is_alive()

    def close(self) -> None:
        """Stop the receiver thread and close the socket."""
        self._running = False
        self._thread.join(timeout=3)
        try:
            self._sock.close()
        except OSError:
            pass

    @property
    def fps(self) -> int:
        return 30

    @property
    def bone_names(self) -> list[str]:
        return self._bone_names

    @property
    def bone_parents(self) -> np.ndarray:
        return self._bone_parents

    @property
    def human_height(self) -> float:
        return self._human_height

    # ------------------------------------------------------------------
    # Background receiver
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                text = raw.decode("utf-8").strip()
                if not text:
                    continue
                floats = np.array(text.split(), dtype=np.float64)
            except (ValueError, UnicodeDecodeError) as exc:
                logger.warning("UDP parse error: %s", exc)
                continue

            if floats.shape[0] != self._expected_floats:
                logger.warning(
                    "UDP packet has %d floats, expected %d — skipping",
                    floats.shape[0],
                    self._expected_floats,
                )
                continue

            try:
                frame = process_single_bvh_frame(
                    data_floats=floats,
                    offsets=self._offsets,
                    bone_names=self._bone_names,
                    bone_parents=self._bone_parents,
                    euler_order=self._euler_order,
                    rotation_quat=self._rotation_quat,
                    rotation_matrix=self._rotation_matrix,
                    format=self._format,
                    scale_divisor=self._scale_divisor,
                    channels=self._channels,
                )
            except Exception:
                logger.exception("Failed to process UDP BVH frame")
                continue

            with self._lock:
                self._latest_frame = frame
            if not self._frame_ready.is_set():
                self._frame_ready.set()
