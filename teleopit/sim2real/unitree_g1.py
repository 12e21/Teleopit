"""Unitree G1 robot interface via SDK2 Python.

Wraps DDS communication (LowCmd publisher / LowState subscriber) and
provides a RobotState-compatible API for the sim2real controller.

Design based on unitreerobotics/xr_teleoperate patterns:
- LowCmd publisher is created LAZILY (only for low-level/debug mode)
- Uses unitree_hg_msg_dds__LowCmd_() default constructor for messages
- mode_machine is read from LowState and stamped on every LowCmd
- CRC32 computed on every outgoing LowCmd
- Continuous publishing at 250Hz via dedicated thread
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from teleopit.interfaces import RobotState

logger = logging.getLogger(__name__)

# Number of actuated joints used by the policy (29-DOF G1)
_NUM_JOINTS = 29
# Total motor slots in the SDK lowcmd/lowstate (G1 has 35 slots)
_NUM_MOTORS = 35
# Ankle control mode: PR = Pitch/Roll (default for 29-DOF G1)
_MODE_PR = 0
# Motor control rate (Hz) -- matches xr_teleoperate
_PUBLISH_HZ = 250


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        return default if value is None else value
    return getattr(cfg, key, default)


class UnitreeG1Robot:
    """Control a physical G1 robot via Unitree SDK2 Python.

    Two operating paradigms (matching xr_teleoperate):
    - High-level (ai mode): LocoClient handles walking, no LowCmd needed
    - Low-level (debug mode): Direct LowCmd on "rt/lowcmd", 250Hz publish thread

    The LowCmd publisher is created lazily to avoid conflicting with the
    sport service (LocoClient) in high-level mode.
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
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HG_LowState
        from unitree_sdk2py.utils.crc import CRC

        self._LowCmd_Factory = unitree_hg_msg_dds__LowCmd_
        self._ChannelPublisher = ChannelPublisher
        self._crc = CRC()

        # ---- DDS initialisation ----
        ChannelFactoryInitialize(0, self._network_interface)

        # Subscriber for low-level state (always active -- read-only, no conflict)
        self._lowstate: HG_LowState | None = None
        self._state_sub = ChannelSubscriber("rt/lowstate", HG_LowState)
        self._state_sub.Init(self._on_lowstate, 10)

        # LowCmd publisher -- LAZY INIT (created on first low-level command)
        self._cmd_pub: Any = None
        self._cmd: Any = None
        self._low_level_ready: bool = False

        # Lock for thread-safe command updates (main thread writes, publish thread reads)
        self._cmd_lock = threading.Lock()
        self._publish_thread: threading.Thread | None = None
        self._publish_running: bool = False

        # mode_machine tracking (read from LowState, must be set on every LowCmd)
        self._mode_machine: int = 0
        self._mode_machine_updated: bool = False

        # Cached MotionSwitcherClient (lazy init)
        self._motion_switcher: Any = None

        # Wait briefly for first state message
        deadline = time.monotonic() + 3.0
        while self._lowstate is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._lowstate is None:
            logger.warning("No LowState received within 3s -- robot may not be connected")
        else:
            logger.info(
                "UnitreeG1Robot: mode_machine=%d from LowState",
                self._mode_machine,
            )

        logger.info(
            "UnitreeG1Robot initialised on %s (LowCmd publisher: DEFERRED)",
            self._network_interface,
        )

    # ------------------------------------------------------------------
    # DDS callback
    # ------------------------------------------------------------------

    def _on_lowstate(self, msg: Any) -> None:
        self._lowstate = msg
        if not self._mode_machine_updated:
            self._mode_machine = msg.mode_machine
            self._mode_machine_updated = True

    # ------------------------------------------------------------------
    # Lazy LowCmd publisher + 250Hz publish thread
    # ------------------------------------------------------------------

    def _ensure_low_level(self) -> None:
        """Create LowCmd publisher, default message, and start 250Hz publish thread.

        Uses unitree_hg_msg_dds__LowCmd_() default constructor (matching
        xr_teleoperate pattern). Must NOT be called while sport service
        (LocoClient) is active.
        """
        if self._low_level_ready:
            return

        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HG_LowCmd

        self._cmd_pub = self._ChannelPublisher("rt/lowcmd", HG_LowCmd)
        self._cmd_pub.Init()

        # Use default constructor (matching xr_teleoperate)
        self._cmd = self._LowCmd_Factory()
        self._cmd.mode_pr = _MODE_PR
        self._cmd.mode_machine = self._mode_machine

        # Initialize all motors to damping (safe default)
        for i in range(_NUM_MOTORS):
            self._cmd.motor_cmd[i].mode = 1  # Enable motor (G1 hg protocol)
            self._cmd.motor_cmd[i].q = 0.0
            self._cmd.motor_cmd[i].kp = 0.0
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kd = self._kd_damping
            self._cmd.motor_cmd[i].tau = 0.0

        self._low_level_ready = True
        logger.info("LowCmd publisher created (low-level control ready)")

        # Start 250Hz continuous publish thread
        self._start_publish_thread()

    def _start_publish_thread(self) -> None:
        """Start daemon thread that publishes LowCmd at 250Hz."""
        if self._publish_thread is not None:
            return
        self._publish_running = True
        self._publish_thread = threading.Thread(
            target=self._publish_loop, daemon=True, name="lowcmd_publisher",
        )
        self._publish_thread.start()
        logger.info("LowCmd publish thread started (250Hz)")

    def _stop_publish_thread(self) -> None:
        """Stop the 250Hz publish thread."""
        self._publish_running = False
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=1.0)
            self._publish_thread = None
            logger.info("LowCmd publish thread stopped")

    def _publish_loop(self) -> None:
        """Continuously publish LowCmd at 250Hz (matching xr_teleoperate)."""
        dt = 1.0 / _PUBLISH_HZ
        while self._publish_running:
            t0 = time.monotonic()
            with self._cmd_lock:
                self._cmd.mode_machine = self._mode_machine
                self._cmd.crc = self._crc.Crc(self._cmd)
                self._cmd_pub.Write(self._cmd)
            elapsed = time.monotonic() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------
    # Public API: state reading
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        """Read latest LowState -> RobotState(qpos[29], qvel[29], quat[4], ang_vel[3])."""
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
    # Public API: low-level commands (trigger lazy publisher)
    # ------------------------------------------------------------------

    def lock_all_joints(self) -> None:
        """Lock ALL joints to current position with PD gains.

        Called when entering low-level mode to prevent the robot from
        collapsing. Matches xr_teleoperate initialization pattern.
        """
        self._ensure_low_level()

        if self._lowstate is None:
            logger.warning("No LowState -- cannot lock joints")
            return

        ls = self._lowstate
        with self._cmd_lock:
            for i in range(_NUM_JOINTS):
                motor_idx = self._joint_map[i]
                self._cmd.motor_cmd[motor_idx].mode = 1
                self._cmd.motor_cmd[motor_idx].q = ls.motor_state[motor_idx].q
                self._cmd.motor_cmd[motor_idx].kp = float(self._kp[i])
                self._cmd.motor_cmd[motor_idx].kd = float(self._kd[i])
                self._cmd.motor_cmd[motor_idx].dq = 0.0
                self._cmd.motor_cmd[motor_idx].tau = 0.0
        logger.info("All %d joints locked to current position", _NUM_JOINTS)

    def send_positions(
        self,
        target_pos: np.ndarray,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        """Update 29-DOF position targets (published by 250Hz thread)."""
        self._ensure_low_level()

        if target_pos.shape[0] != _NUM_JOINTS:
            raise ValueError(f"target_pos must have {_NUM_JOINTS} entries")

        if kp is None:
            kp = self._kp
        if kd is None:
            kd = self._kd

        with self._cmd_lock:
            for i in range(_NUM_JOINTS):
                motor_idx = self._joint_map[i]
                self._cmd.motor_cmd[motor_idx].mode = 1
                self._cmd.motor_cmd[motor_idx].q = float(target_pos[i])
                self._cmd.motor_cmd[motor_idx].kp = float(kp[i])
                self._cmd.motor_cmd[motor_idx].dq = 0.0
                self._cmd.motor_cmd[motor_idx].kd = float(kd[i])
                self._cmd.motor_cmd[motor_idx].tau = 0.0

    def set_damping(self) -> None:
        """Set all 35 motors to damping mode (kp=0, kd=kd_damping, tau=0)."""
        self._ensure_low_level()

        with self._cmd_lock:
            for i in range(_NUM_MOTORS):
                self._cmd.motor_cmd[i].mode = 1
                self._cmd.motor_cmd[i].q = 0.0
                self._cmd.motor_cmd[i].kp = 0.0
                self._cmd.motor_cmd[i].dq = 0.0
                self._cmd.motor_cmd[i].kd = self._kd_damping
                self._cmd.motor_cmd[i].tau = 0.0

    # ------------------------------------------------------------------
    # motion_switcher: high/low level switching
    # ------------------------------------------------------------------

    def _get_motion_switcher(self) -> Any:
        """Lazily initialise and cache the MotionSwitcherClient."""
        if self._motion_switcher is not None:
            return self._motion_switcher

        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )

        client = MotionSwitcherClient()
        client.SetTimeout(1.0)  # 1s timeout (matching xr_teleoperate)
        client.Init()
        self._motion_switcher = client
        logger.info("MotionSwitcherClient initialised (timeout=1.0s)")
        return client

    def check_mode(self) -> dict | None:
        """Query current motion mode from robot. Returns dict with 'name' key."""
        try:
            client = self._get_motion_switcher()
            code, result = client.CheckMode()
            logger.info("motion_switcher: CheckMode -> code=%s, result=%s", code, result)
            return result
        except Exception as exc:
            logger.error("motion_switcher: CheckMode failed: %s", exc)
            return None

    def enter_debug_mode(self) -> bool:
        """Enter debug mode: release all modes until none active.

        Matches xr_teleoperate MotionSwitcher.Enter_Debug_Mode() pattern.
        After this, "rt/lowcmd" publisher can safely control all joints.
        """
        try:
            client = self._get_motion_switcher()
            for attempt in range(10):
                code, result = client.CheckMode()
                logger.info("enter_debug_mode: CheckMode -> code=%s, result=%s", code, result)
                mode_name = result.get("name", "") if isinstance(result, dict) else ""
                if not mode_name:
                    logger.info("enter_debug_mode: no active mode -- debug mode ready")
                    return True
                client.ReleaseMode()
                logger.info("enter_debug_mode: ReleaseMode called")
                time.sleep(1)
            logger.warning("enter_debug_mode: could not fully release after 10 attempts")
            return False
        except Exception as exc:
            logger.error("enter_debug_mode failed: %s", exc)
            return False

    def exit_debug_mode(self) -> bool:
        """Exit debug mode: SelectMode('ai') to restore onboard locomotion.

        Matches xr_teleoperate MotionSwitcher.Exit_Debug_Mode() pattern.
        """
        # Stop the publish thread first (avoid conflicting with sport service)
        self._stop_publish_thread()

        try:
            client = self._get_motion_switcher()
            code, result = client.SelectMode("ai")
            logger.info("exit_debug_mode: SelectMode('ai') -> code=%s, result=%s", code, result)
            return code == 0
        except Exception as exc:
            logger.error("exit_debug_mode failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up."""
        self._stop_publish_thread()
        logger.info("UnitreeG1Robot closed")
