"""Unitree G1 robot interface via SDK2 Python.

Wraps DDS communication (LowCmd publisher / LowState subscriber) and
provides a RobotState-compatible API for the sim2real controller.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from teleopit.interfaces import RobotState

logger = logging.getLogger(__name__)

# Number of actuated joints used by the policy (29-DOF G1)
_NUM_JOINTS = 29
# Total motor slots in the SDK lowcmd/lowstate (G1 has 35 slots)
_NUM_MOTORS = 35
# FOC control mode byte
_MODE_FOC = 0x01


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        return default if value is None else value
    return getattr(cfg, key, default)


class UnitreeG1Robot:
    """Control a physical G1 robot via Unitree SDK2 Python.

    Responsibilities:
    1. DDS initialisation (ChannelFactory) with LowCmd publisher + LowState subscriber
    2. Read motor state / IMU → RobotState
    3. Send 29-DOF position commands (FOC mode with kp/kd)
    4. Damping mode (kp=0, kd=kd_damping, tau=0 for all 35 motors)
    5. CRC32 on every outgoing LowCmd
    6. motion_switcher service calls for high/low level switching
    """

    def __init__(self, cfg: Any) -> None:
        # ---- Configuration ----
        self._network_interface: str = str(_cfg_get(cfg, "network_interface", "eth0"))
        joint_map_raw = _cfg_get(cfg, "joint_map", list(range(_NUM_JOINTS)))
        self._joint_map: list[int] = [int(j) for j in joint_map_raw]
        self._kp = np.asarray(_cfg_get(cfg, "kp_real", [100] * _NUM_JOINTS), dtype=np.float32)
        self._kd = np.asarray(_cfg_get(cfg, "kd_real", [2] * _NUM_JOINTS), dtype=np.float32)
        self._kd_damping: float = float(_cfg_get(cfg, "kd_damping", 8.0))

        if len(self._joint_map) != _NUM_JOINTS:
            raise ValueError(f"joint_map must have {_NUM_JOINTS} entries, got {len(self._joint_map)}")
        if self._kp.shape[0] != _NUM_JOINTS:
            raise ValueError(f"kp_real must have {_NUM_JOINTS} entries")
        if self._kd.shape[0] != _NUM_JOINTS:
            raise ValueError(f"kd_real must have {_NUM_JOINTS} entries")

        # ---- SDK imports (deferred to avoid import errors when SDK not installed) ----
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HG_LowCmd
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HG_LowState
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_ as HG_MotorCmd
        from unitree_sdk2py.utils.crc import CRC

        self._HG_LowCmd = HG_LowCmd
        self._HG_MotorCmd = HG_MotorCmd
        self._crc = CRC()

        # ---- DDS initialisation ----
        ChannelFactoryInitialize(0, self._network_interface)

        # Publisher for low-level commands
        self._cmd_pub = ChannelPublisher("rt/lowcmd", HG_LowCmd)
        self._cmd_pub.Init()

        # Subscriber for low-level state
        self._lowstate: HG_LowState | None = None
        self._state_sub = ChannelSubscriber("rt/lowstate", HG_LowState)
        self._state_sub.Init(self._on_lowstate, 10)

        # Pre-allocate a command message (IDL dataclasses require all fields)
        self._cmd = HG_LowCmd(
            mode_pr=0,
            mode_machine=0,
            motor_cmd=[HG_MotorCmd(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0) for _ in range(_NUM_MOTORS)],
            reserve=[0, 0, 0, 0],
            crc=0,
        )

        # Wait briefly for first state message
        deadline = time.monotonic() + 3.0
        while self._lowstate is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._lowstate is None:
            logger.warning("No LowState received within 3s — robot may not be connected")

        logger.info("UnitreeG1Robot initialised on %s", self._network_interface)

    # ------------------------------------------------------------------
    # DDS callback
    # ------------------------------------------------------------------

    def _on_lowstate(self, msg: Any) -> None:
        self._lowstate = msg

    # ------------------------------------------------------------------
    # Public API: state reading
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        """Read latest LowState → RobotState(qpos[29], qvel[29], quat[4], ang_vel[3])."""
        if self._lowstate is None:
            return RobotState(
                qpos=np.zeros(_NUM_JOINTS, dtype=np.float32),
                qvel=np.zeros(_NUM_JOINTS, dtype=np.float32),
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                ang_vel=np.zeros(3, dtype=np.float32),
                timestamp=time.time(),
            )

        ls = self._lowstate
        qpos = np.zeros(_NUM_JOINTS, dtype=np.float32)
        qvel = np.zeros(_NUM_JOINTS, dtype=np.float32)
        for i in range(_NUM_JOINTS):
            motor_idx = self._joint_map[i]
            qpos[i] = ls.motor_state[motor_idx].q
            qvel[i] = ls.motor_state[motor_idx].dq

        # IMU: quaternion (w, x, y, z) and gyroscope
        quat = np.array(ls.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.array(ls.imu_state.gyroscope, dtype=np.float32)

        return RobotState(
            qpos=qpos,
            qvel=qvel,
            quat=quat,
            ang_vel=ang_vel,
            timestamp=time.time(),
        )

    def get_lowstate(self) -> Any:
        """Return raw LowState (contains wireless_remote etc.)."""
        return self._lowstate

    # ------------------------------------------------------------------
    # Public API: commands
    # ------------------------------------------------------------------

    def send_positions(
        self,
        target_pos: np.ndarray,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        """Send 29-DOF position command via FOC mode.

        Each joint: motor_cmd[joint_map[i]].q = target, kp, kd, dq=0, tau=0, mode=FOC.
        Automatically computes and sets CRC32.
        """
        if target_pos.shape[0] != _NUM_JOINTS:
            raise ValueError(f"target_pos must have {_NUM_JOINTS} entries")

        if kp is None:
            kp = self._kp
        if kd is None:
            kd = self._kd

        cmd = self._cmd
        for i in range(_NUM_JOINTS):
            motor_idx = self._joint_map[i]
            cmd.motor_cmd[motor_idx].mode = _MODE_FOC
            cmd.motor_cmd[motor_idx].q = float(target_pos[i])
            cmd.motor_cmd[motor_idx].kp = float(kp[i])
            cmd.motor_cmd[motor_idx].dq = 0.0
            cmd.motor_cmd[motor_idx].kd = float(kd[i])
            cmd.motor_cmd[motor_idx].tau = 0.0

        cmd.crc = self._crc.Crc(cmd)
        self._cmd_pub.Write(cmd)

    def set_damping(self) -> None:
        """Set all 35 motors to damping mode (kp=0, kd=kd_damping, tau=0)."""
        cmd = self._cmd
        for i in range(_NUM_MOTORS):
            cmd.motor_cmd[i].mode = _MODE_FOC
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = self._kd_damping
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = self._crc.Crc(cmd)
        self._cmd_pub.Write(cmd)

    # ------------------------------------------------------------------
    # motion_switcher: high/low level switching
    # ------------------------------------------------------------------

    def enable_low_level(self) -> None:
        """Stop onboard locomotion service → switch to low-level control.

        Uses MotionSwitcherClient.ReleaseMode() to release the built-in
        locomotion controller, allowing direct LowCmd control.
        """
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )

            client = MotionSwitcherClient()
            client.SetTimeout(5.0)
            client.Init()
            code, _ = client.ReleaseMode()
            logger.info("motion_switcher: ReleaseMode → low-level control (code=%s)", code)
        except Exception as exc:
            logger.error("motion_switcher: enable_low_level failed: %s", exc)

    def enable_high_level(self) -> None:
        """Restore onboard locomotion service → switch to high-level control.

        Uses MotionSwitcherClient.SelectMode("normal") to re-enable the
        built-in locomotion controller for LocoClient usage.
        """
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )

            client = MotionSwitcherClient()
            client.SetTimeout(5.0)
            client.Init()
            code, _ = client.SelectMode("normal")
            logger.info("motion_switcher: SelectMode('normal') → high-level control (code=%s)", code)
        except Exception as exc:
            logger.error("motion_switcher: enable_high_level failed: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Send damping command and clean up."""
        try:
            self.set_damping()
        except Exception:
            pass
        logger.info("UnitreeG1Robot closed")
