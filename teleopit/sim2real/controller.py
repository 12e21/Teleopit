"""Sim2Real controller — state machine + dual-mode control loop.

Supports two operating modes for a physical Unitree G1:
- **Gamepad**: joystick → LocoClient.Move() (high-level locomotion)
- **Mocap**: UDP BVH → retarget → RL policy → SDK low-level joint control

The controller reuses the same observation / retargeting / policy pipeline
as sim2sim, but replaces the MuJoCo backend with SDK2 DDS communication.
"""

from __future__ import annotations

import logging
import struct
import time
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from teleopit.controllers.observation import TWIST2ObservationBuilder
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.udp_bvh_provider import UDPBVHInputProvider
from teleopit.interfaces import RobotState
from teleopit.retargeting.core import RetargetingModule, extract_mimic_obs
from teleopit.sim2real.remote import UnitreeRemote
from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

logger = logging.getLogger(__name__)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        return default if value is None else value
    return getattr(cfg, key, default)


class RobotMode(Enum):
    DAMPING = "damping"
    PREPARATION = "preparation"
    STANDING = "standing"
    GAMEPAD = "gamepad"
    MOCAP = "mocap"


class Sim2RealController:
    """G1 real-robot controller — gamepad/mocap dual mode with state machine.

    Parameters (from Hydra config):
        cfg.real_robot  — UnitreeG1Robot configuration
        cfg.input       — UDPBVHInputProvider configuration (mocap mode)
        cfg.controller  — RLPolicyController configuration
        cfg.robot       — robot model params (for obs_builder)
        cfg.gamepad     — velocity limits (max_vx, max_vy, max_vyaw)
        cfg.policy_hz   — control loop frequency (default 50 Hz)
        cfg.mocap_switch — safety check params
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mode = RobotMode.DAMPING

        self.policy_hz: float = float(_cfg_get(cfg, "policy_hz", 50.0))
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # ---- Real robot (SDK) ----
        real_cfg = _cfg_get(cfg, "real_robot")
        self.robot = UnitreeG1Robot(real_cfg)
        self.remote = UnitreeRemote()

        # ---- Gamepad config ----
        gp_cfg = _cfg_get(cfg, "gamepad", {})
        self.max_vx: float = float(_cfg_get(gp_cfg, "max_vx", 0.5))
        self.max_vy: float = float(_cfg_get(gp_cfg, "max_vy", 0.3))
        self.max_vyaw: float = float(_cfg_get(gp_cfg, "max_vyaw", 0.5))

        # ---- LocoClient (high-level locomotion, gamepad mode) ----
        self._loco_client: Any = None  # lazily initialised
        loco_cfg = _cfg_get(cfg, "loco", {})
        self._loco_timeout: float = float(_cfg_get(loco_cfg, "timeout", 10.0))

        # ---- Mocap pipeline (reuse existing components) ----
        input_cfg = _cfg_get(cfg, "input")
        controller_cfg = _cfg_get(cfg, "controller")
        robot_cfg = _cfg_get(cfg, "robot")

        # Resolve reference BVH path
        ref_bvh = str(_cfg_get(input_cfg, "reference_bvh", ""))
        if ref_bvh and not Path(ref_bvh).is_absolute():
            ref_bvh = str((self._project_root / ref_bvh).resolve())

        self.udp_provider = UDPBVHInputProvider(
            reference_bvh=ref_bvh,
            host=str(_cfg_get(input_cfg, "udp_host", "")),
            port=int(_cfg_get(input_cfg, "udp_port", 1118)),
            human_format=str(_cfg_get(input_cfg, "bvh_format", "hc_mocap")),
            timeout=float(_cfg_get(input_cfg, "udp_timeout", 30.0)),
        )

        human_format = _cfg_get(input_cfg, "human_format", None)
        if not human_format or str(human_format) == "null":
            bvh_format = str(_cfg_get(input_cfg, "bvh_format", "hc_mocap"))
            human_format = f"bvh_{bvh_format}"

        self.retargeter = RetargetingModule(
            robot_name=str(_cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(human_format),
            actual_human_height=float(_cfg_get(input_cfg, "human_height", 1.75)),
        )

        # Ensure controller has default_dof_pos
        if _cfg_get(controller_cfg, "default_dof_pos", None) is None:
            default_angles = _cfg_get(robot_cfg, "default_angles", None)
            if default_angles is not None:
                if hasattr(controller_cfg, "__setattr__"):
                    controller_cfg.default_dof_pos = list(default_angles)
                elif isinstance(controller_cfg, dict):
                    controller_cfg["default_dof_pos"] = list(default_angles)

        # Resolve policy path
        policy_path = str(_cfg_get(controller_cfg, "policy_path", ""))
        if policy_path and not Path(policy_path).is_absolute():
            resolved = (self._project_root / policy_path).resolve()
            if resolved.exists():
                if hasattr(controller_cfg, "__setattr__"):
                    controller_cfg.policy_path = str(resolved)
                elif isinstance(controller_cfg, dict):
                    controller_cfg["policy_path"] = str(resolved)

        self.policy = RLPolicyController(controller_cfg)

        # ObservationBuilder
        obs_cfg = {
            "num_actions": int(_cfg_get(robot_cfg, "num_actions", 29)),
            "ang_vel_scale": float(_cfg_get(robot_cfg, "ang_vel_scale", 0.25)),
            "dof_pos_scale": float(_cfg_get(robot_cfg, "dof_pos_scale", 1.0)),
            "dof_vel_scale": float(_cfg_get(robot_cfg, "dof_vel_scale", 0.05)),
            "ankle_idx": list(_cfg_get(robot_cfg, "ankle_idx", [4, 5, 10, 11])),
            "default_dof_pos": list(_cfg_get(robot_cfg, "default_angles")),
        }
        self.obs_builder = TWIST2ObservationBuilder(obs_cfg)

        # Default standing pose (29-DOF)
        self.default_angles = np.asarray(
            _cfg_get(robot_cfg, "default_angles"), dtype=np.float32
        )
        self.num_actions: int = int(_cfg_get(robot_cfg, "num_actions", 29))

        # ---- Mocap mode state ----
        self._last_action: Float32Array = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None

        # ---- Mocap switch safety ----
        mocap_sw = _cfg_get(cfg, "mocap_switch", {})
        self._check_frames: int = int(_cfg_get(mocap_sw, "check_frames", 10))
        self._max_pos_value: float = float(_cfg_get(mocap_sw, "max_position_value", 5.0))

        logger.info(
            "Sim2RealController ready | policy_hz=%.0f | gamepad vx=%.2f vy=%.2f vyaw=%.2f",
            self.policy_hz, self.max_vx, self.max_vy, self.max_vyaw,
        )

    # ------------------------------------------------------------------
    # LocoClient lazy init
    # ------------------------------------------------------------------

    def _get_loco_client(self) -> Any:
        if self._loco_client is not None:
            return self._loco_client
        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import G1LocoClient

            client = G1LocoClient()
            client.SetTimeout(self._loco_timeout)
            client.Init()
            self._loco_client = client
            logger.info("LocoClient initialised")
        except ImportError:
            # Fallback: try generic LocoClient
            from unitree_sdk2py.go2.sport.sport_client import SportClient

            client = SportClient()
            client.SetTimeout(self._loco_timeout)
            client.Init()
            self._loco_client = client
            logger.info("LocoClient (SportClient fallback) initialised")
        return self._loco_client

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop at policy_hz."""
        logger.info("Starting control loop — press Start to begin")
        dt = 1.0 / self.policy_hz

        try:
            while True:
                t0 = time.monotonic()

                # 1. Read remote state
                ls = self.robot.get_lowstate()
                if ls is not None:
                    remote_bytes = bytes(ls.wireless_remote)
                    self.remote.update(remote_bytes)

                # 2. Emergency stop (highest priority)
                if self._check_emergency_stop():
                    if self.mode != RobotMode.DAMPING:
                        logger.warning("EMERGENCY STOP — entering damping mode")
                        self._enter_damping()
                    # Keep sending damping while e-stop held
                    self.robot.set_damping()
                    self._sleep_until(t0, dt)
                    continue

                # 3. Mode transitions
                self._handle_transitions()

                # 4. Execute current mode
                if self.mode == RobotMode.GAMEPAD:
                    self._gamepad_step()
                elif self.mode == RobotMode.MOCAP:
                    self._mocap_step()
                elif self.mode == RobotMode.DAMPING:
                    self.robot.set_damping()

                # 5. Rate control
                self._sleep_until(t0, dt)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — entering damping mode")
            self._enter_damping()
            self.robot.set_damping()

    # ------------------------------------------------------------------
    # Mode execution
    # ------------------------------------------------------------------

    def _gamepad_step(self) -> None:
        """Gamepad mode: joystick → LocoClient.Move()."""
        vx = self.remote.ly * self.max_vx
        vy = self.remote.lx * self.max_vy
        vyaw = self.remote.rx * self.max_vyaw

        client = self._get_loco_client()
        try:
            client.Move(vx, vy, vyaw)
        except Exception as exc:
            logger.error("LocoClient.Move failed: %s", exc)

    def _mocap_step(self) -> None:
        """Mocap mode: UDP BVH → retarget → policy → SDK lowcmd.

        Reuses the SimulationLoop core logic, but:
        - Robot state comes from SDK LowState (not MuJoCo)
        - Actions go to SDK LowCmd as position targets (not MuJoCo torques)
        - No PD decimation inner loop (SDK motor controller runs PD)
        """
        # Get human motion frame
        if not self.udp_provider.is_available():
            logger.warning("UDP provider unavailable — entering damping")
            self._enter_damping()
            return

        try:
            human_frame = self.udp_provider.get_frame()
        except TimeoutError:
            logger.warning("UDP timeout — entering damping")
            self._enter_damping()
            return

        # Retarget → mimic observation (35D)
        retargeted = self.retargeter.retarget(human_frame)
        qpos = self._retarget_to_qpos(retargeted)
        mimic_obs = extract_mimic_obs(
            qpos=qpos,
            last_qpos=self._last_retarget_qpos,
            dt=1.0 / self.policy_hz,
        )

        # Robot state from SDK
        robot_state = self.robot.get_state()

        # Build observation (1402D) → policy inference
        obs = self.obs_builder.build(robot_state, mimic_obs, self._last_action)
        obs = self._adapt_observation_for_policy(obs)
        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)

        # Send position command to real robot
        self.robot.send_positions(target_dof_pos)

        # Update state
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _check_emergency_stop(self) -> bool:
        """LB + RB pressed simultaneously → emergency stop."""
        return self.remote.LB.pressed and self.remote.RB.pressed

    def _enter_damping(self) -> None:
        """Enter damping mode (safe stop)."""
        self.robot.set_damping()
        self.mode = RobotMode.DAMPING
        logger.info("Mode → DAMPING")

    def _handle_transitions(self) -> None:
        """Handle remote-triggered mode transitions."""
        if self.mode == RobotMode.DAMPING:
            if self.remote.start.on_pressed:
                logger.info("Mode → PREPARATION (StandUp)")
                self._enter_preparation()

        elif self.mode == RobotMode.PREPARATION:
            if self.remote.A.on_pressed:
                logger.info("Mode → STANDING → GAMEPAD")
                self.mode = RobotMode.STANDING

        elif self.mode == RobotMode.STANDING:
            # Auto-transition to gamepad
            self.robot.enable_high_level()
            time.sleep(0.3)
            self.mode = RobotMode.GAMEPAD
            logger.info("Mode → GAMEPAD")

        elif self.mode == RobotMode.GAMEPAD:
            if self.remote.Y.on_pressed:
                if self._can_switch_to_mocap():
                    logger.info("Mode → MOCAP (UDP verified)")
                    self._transition_to_mocap()
                else:
                    logger.warning("Cannot switch to MOCAP — UDP check failed")

        elif self.mode == RobotMode.MOCAP:
            if self.remote.X.on_pressed:
                logger.info("Mode → GAMEPAD (from MOCAP)")
                self._transition_to_gamepad()

    def _enter_preparation(self) -> None:
        """StandUp via LocoClient."""
        try:
            client = self._get_loco_client()
            self.robot.enable_high_level()
            time.sleep(0.5)
            client.StandUp()
            self.mode = RobotMode.PREPARATION
        except Exception as exc:
            logger.error("StandUp failed: %s — staying in DAMPING", exc)
            self.mode = RobotMode.DAMPING

    # ------------------------------------------------------------------
    # Mode switching: gamepad ↔ mocap
    # ------------------------------------------------------------------

    def _can_switch_to_mocap(self) -> bool:
        """Verify UDP signal is stable and values are reasonable.

        Checks:
        1. UDP receiver thread alive
        2. At least one frame received
        3. N consecutive frames valid (no NaN, positions in range)
        """
        if not self.udp_provider.is_available():
            logger.warning("Mocap check: UDP provider not available")
            return False

        if not self.udp_provider._frame_ready.is_set():
            logger.warning("Mocap check: no UDP data received yet")
            return False

        # Check N consecutive frames for validity
        valid_count = 0
        for _ in range(self._check_frames + 5):  # allow a few extra attempts
            try:
                frame = self.udp_provider.get_frame()
            except TimeoutError:
                return False

            # Check all bone positions for NaN/Inf and range
            all_valid = True
            for bone_name, (pos, quat) in frame.items():
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    all_valid = False
                    break
                if np.any(np.abs(pos) > self._max_pos_value):
                    all_valid = False
                    break
                if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
                    all_valid = False
                    break

            if all_valid:
                valid_count += 1
            else:
                valid_count = 0  # reset on invalid frame

            if valid_count >= self._check_frames:
                return True

            time.sleep(0.02)  # ~50Hz check rate

        logger.warning("Mocap check: only %d/%d valid frames", valid_count, self._check_frames)
        return False

    def _transition_to_mocap(self) -> None:
        """Switch from gamepad → mocap mode.

        1. Stop LocoClient movement
        2. Switch to low-level control via motion_switcher
        3. Read current joint positions as initial state
        """
        # Stop locomotion
        try:
            client = self._get_loco_client()
            client.Move(0.0, 0.0, 0.0)
        except Exception:
            pass
        time.sleep(0.5)

        # Switch to low-level control
        self.robot.enable_low_level()
        time.sleep(0.3)

        # Read current state as initial
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)

        # Reset observation builder history
        self.obs_builder.reset()

        self.mode = RobotMode.MOCAP

    def _transition_to_gamepad(self) -> None:
        """Switch from mocap → gamepad mode.

        1. Smooth interpolation to default standing pose (2s)
        2. Restore high-level control via motion_switcher
        """
        self._smooth_to_default_pose(duration=2.0)

        # Switch to high-level
        self.robot.enable_high_level()
        time.sleep(0.5)

        self.mode = RobotMode.GAMEPAD

    def _smooth_to_default_pose(self, duration: float = 2.0) -> None:
        """Linear interpolation from current joint positions to default standing pose."""
        state = self.robot.get_state()
        start_pos = state.qpos.copy().astype(np.float32)
        target_pos = self.default_angles.copy()
        steps = max(int(duration * self.policy_hz), 1)
        dt = 1.0 / self.policy_hz

        for i in range(steps):
            alpha = (i + 1) / steps
            interp = start_pos * (1.0 - alpha) + target_pos * alpha
            self.robot.send_positions(interp)
            time.sleep(dt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retarget_to_qpos(self, retargeted: object) -> Float64Array:
        """Convert retarget output to 36D qpos (7D root + 29D joints)."""
        if isinstance(retargeted, tuple) and len(retargeted) == 3:
            base_pos = np.asarray(retargeted[0], dtype=np.float64).reshape(-1)
            base_rot = np.asarray(retargeted[1], dtype=np.float64).reshape(-1)
            joint_pos = np.asarray(retargeted[2], dtype=np.float64).reshape(-1)
            qpos = np.concatenate((base_pos, base_rot, joint_pos))
        else:
            qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)
        if qpos.shape[0] < 36:
            raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= 36)")
        return qpos

    def _adapt_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        """Pad or truncate observation to match policy input dimension."""
        expected = getattr(self.policy, "_expected_obs_dim", None)
        if not isinstance(expected, int) or expected <= 0:
            return obs
        if obs.shape[0] == expected:
            return obs
        if obs.shape[0] > expected:
            return obs[:expected]
        pad = np.zeros(expected - obs.shape[0], dtype=np.float32)
        return np.concatenate((obs, pad), dtype=np.float32)

    @staticmethod
    def _sleep_until(t0: float, dt: float) -> None:
        """Sleep to maintain control frequency."""
        elapsed = time.monotonic() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean shutdown: damping + close resources."""
        logger.info("Shutting down Sim2RealController")
        try:
            self.robot.set_damping()
        except Exception:
            pass
        try:
            self.udp_provider.close()
        except Exception:
            pass
        try:
            self.robot.close()
        except Exception:
            pass
