"""Sim2Real controller -- state machine + dual-mode control loop.

Supports two operating modes for a physical Unitree G1:
- **Gamepad**: LocoClient sends velocity commands (robot stays in "ai" mode)
- **Mocap**: Debug mode + direct LowCmd joint control via RL policy

Prerequisites:
    User must manually activate ai_sport walking via remote controller
    BEFORE starting this script (boot → damping → locked standing → walking).

State machine:
    IDLE ──Start──▶ GAMEPAD ──Y──▶ MOCAP ──X──▶ DAMPING
    Any ──L1+R1──▶ DAMPING ──Start──▶ GAMEPAD (if ai_sport active)
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.controllers.observation import MjlabObservationBuilder
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.udp_bvh_provider import UDPBVHInputProvider
from teleopit.retargeting.core import RetargetingModule
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
    IDLE = "idle"          # Script waiting, robot controlled by remote
    GAMEPAD = "gamepad"    # LocoClient active, joystick walking
    MOCAP = "mocap"        # Debug mode, RL policy, 250Hz LowCmd
    DAMPING = "damping"    # Emergency stop / recovery


class Sim2RealController:
    """G1 real-robot controller -- gamepad/mocap dual mode with state machine.

    Gamepad mode: robot stays in "ai" mode, LocoClient sends velocity commands.
    Mocap mode: enter debug mode, direct LowCmd via RL policy at 250Hz.

    User must activate ai_sport via remote before starting this script.
    The script does NOT attempt to activate ai_sport via SDK.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mode = RobotMode.IDLE

        self.policy_hz: float = float(_cfg_get(cfg, "policy_hz", 50.0))
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # Motion command transition smoothing
        transition_dur = float(_cfg_get(cfg, "transition_duration", 0.0) or 0.0)
        self._qpos_interpolator = QposInterpolator(transition_dur, self.policy_hz)

        # ---- Real robot (SDK) ----
        real_cfg = _cfg_get(cfg, "real_robot")
        self.robot = UnitreeG1Robot(real_cfg)
        self.remote = UnitreeRemote()

        # ---- LocoClient (created in _enter_gamepad) ----
        self._loco_client: Any = None

        # ---- Gamepad speed limits ----
        gp_cfg = _cfg_get(cfg, "gamepad", {})
        self._max_vx: float = float(_cfg_get(gp_cfg, "max_vx", 0.5))
        self._max_vy: float = float(_cfg_get(gp_cfg, "max_vy", 0.3))
        self._max_vyaw: float = float(_cfg_get(gp_cfg, "max_vyaw", 0.5))

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

        # Use provider's human_format (may be auto-adjusted, e.g. hc_mocap_no_toe)
        if hasattr(self.udp_provider, "human_format"):
            human_format = f"bvh_{self.udp_provider.human_format}"
        else:
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

        # Propagate action_scale from robot config only when controller
        # has no explicit value (null). Explicit overrides are preserved.
        if _cfg_get(controller_cfg, "action_scale", None) is None:
            robot_action_scale = _cfg_get(robot_cfg, "action_scale", None)
            if robot_action_scale is not None:
                try:
                    scale_val = list(robot_action_scale)
                except TypeError:
                    scale_val = robot_action_scale
                if hasattr(controller_cfg, "__setattr__"):
                    controller_cfg.action_scale = scale_val
                elif isinstance(controller_cfg, dict):
                    controller_cfg["action_scale"] = scale_val

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
        policy_dim = getattr(self.policy, "_expected_obs_dim", None)

        # ObservationBuilder -- mjlab-aligned path only.
        obs_type = str(_cfg_get(robot_cfg, "obs_builder", "mjlab")).lower()
        if obs_type != "mjlab":
            raise ValueError(
                f"Unsupported robot.obs_builder='{obs_type}'. "
                "TWIST2 policy path is deprecated; use mjlab-aligned policy and set robot.obs_builder=mjlab."
            )
        xml_path = str(_cfg_get(robot_cfg, "xml_path", ""))
        if xml_path and not Path(xml_path).is_absolute():
            candidate = (self._project_root / xml_path).resolve()
            if candidate.exists():
                xml_path = str(candidate)
            else:
                # Fallback: GMR assets live inside the teleopit package
                fallback = self._project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_sim2sim_29dof.xml"
                xml_path = str(fallback.resolve())
        # Real robot does not provide base_pos / base_lin_vel, so sim2real is 154D-only.
        has_state_estimation = bool(_cfg_get(robot_cfg, "has_state_estimation", False))
        if has_state_estimation:
            raise ValueError(
                "Sim2real requires robot.has_state_estimation=false. "
                "The real robot does not provide base_pos/base_lin_vel, so only 154D "
                "mjlab ONNX policies exported from *-NoStateEst tasks are supported."
            )
        if policy_dim is not None and policy_dim != 154:
            raise ValueError(
                f"Sim2real only supports 154D mjlab ONNX policies exported from "
                f"*-NoStateEst tasks; got {policy_dim}D."
            )
        obs_cfg = {
            "num_actions": int(_cfg_get(robot_cfg, "num_actions", 29)),
            "default_dof_pos": list(_cfg_get(robot_cfg, "default_angles")),
            "xml_path": xml_path,
            "anchor_body_name": _cfg_get(robot_cfg, "anchor_body_name", "torso_link"),
            "has_state_estimation": has_state_estimation,
        }
        self.obs_builder = MjlabObservationBuilder(obs_cfg)

        # Startup dim validation: obs_builder must match policy input dim
        builder_dim = self.obs_builder.total_obs_size
        if policy_dim is not None and policy_dim != builder_dim:
            raise ValueError(
                f"Observation dimension mismatch at startup: obs_builder produces {builder_dim}D "
                f"but policy expects {policy_dim}D. Check has_state_estimation={has_state_estimation} "
                "and ensure the ONNX model matches."
            )

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

        # ---- Startup: check mode and warn ----
        self._check_startup_mode()

        logger.info(
            "Sim2RealController ready | mode=IDLE | policy_hz=%.0f | "
            "max_vx=%.1f max_vy=%.1f max_vyaw=%.1f",
            self.policy_hz, self._max_vx, self._max_vy, self._max_vyaw,
        )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _check_startup_mode(self) -> None:
        """Check robot mode at startup and log status (no SDK mode changes)."""
        result = self.robot.check_mode()
        if result is None:
            logger.warning(
                "Could not query motion mode -- MotionSwitcher may not be available"
            )
            return

        mode_name = result.get("name", "") if isinstance(result, dict) else str(result)
        if mode_name:
            logger.info("Startup: robot in '%s' mode -- ready", mode_name)
        else:
            logger.warning(
                "Startup: no active mode detected. "
                "Please activate ai_sport via remote controller "
                "(L1+A → L1+UP) before pressing Start."
            )

    # ------------------------------------------------------------------
    # LocoClient management (fire-and-forget, matching xr_teleoperate)
    # ------------------------------------------------------------------

    def _create_loco_client(self) -> Any:
        """Create LocoClient with fire-and-forget timeout."""
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        client = LocoClient()
        client.SetTimeout(0.0001)  # Fire-and-forget (matching xr_teleoperate)
        client.Init()
        logger.info("LocoClient created (fire-and-forget, timeout=0.0001s)")
        return client

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop at policy_hz."""
        logger.info(
            "Control loop started | mode=IDLE | press Start to enter GAMEPAD"
        )
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
                        logger.warning("EMERGENCY STOP (L1+R1)")
                        self._enter_damping()
                    self._sleep_until(t0, dt)
                    continue

                # 3. Mode transitions
                self._handle_transitions()

                # 4. Execute current mode
                if self.mode == RobotMode.GAMEPAD:
                    self._gamepad_step()
                elif self.mode == RobotMode.MOCAP:
                    self._mocap_step()

                # 5. Rate control
                self._sleep_until(t0, dt)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt -- shutting down")

    # ------------------------------------------------------------------
    # Mode execution
    # ------------------------------------------------------------------

    def _gamepad_step(self) -> None:
        """Gamepad mode: read joystick -> LocoClient.Move(vx, vy, vyaw)."""
        if self._loco_client is None:
            return

        vx = self.remote.ly * self._max_vx
        vy = self.remote.lx * self._max_vy
        vyaw = self.remote.rx * self._max_vyaw

        try:
            self._loco_client.Move(vx, vy, vyaw)
        except Exception:
            pass  # Fire-and-forget -- errors are expected (timeout)

    def _mocap_step(self) -> None:
        """Mocap mode: UDP BVH -> retarget -> policy -> update LowCmd targets."""
        if not self.udp_provider.is_available():
            logger.warning("UDP provider unavailable -- entering damping")
            self._enter_damping()
            return

        try:
            human_frame = self.udp_provider.get_frame()
        except TimeoutError:
            logger.warning("UDP timeout -- entering damping")
            self._enter_damping()
            return

        # Retarget -> mimic observation
        retargeted = self.retargeter.retarget(human_frame)
        qpos = self._retarget_to_qpos(retargeted)
        qpos = self._qpos_interpolator.apply(qpos)

        # Robot state from SDK
        robot_state = self.robot.get_state()

        if qpos.shape[0] < 7 + self.num_actions:
            raise ValueError(
                f"Retargeted qpos too short: {qpos.shape[0]} (need >= {7 + self.num_actions})"
            )
        motion_joint_pos = np.asarray(qpos[7:7 + self.num_actions], dtype=np.float32)
        if self._last_retarget_qpos is None:
            motion_joint_vel = np.zeros((self.num_actions,), dtype=np.float32)
        else:
            prev_joint_pos = np.asarray(self._last_retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
            motion_joint_vel = (motion_joint_pos - prev_joint_pos) * np.float32(self.policy_hz)
        obs = self.obs_builder.build(
            robot_state,
            np.asarray(qpos[:7 + self.num_actions], dtype=np.float32),
            motion_joint_vel,
            self._last_action,
        )
        obs = self._validate_observation_for_policy(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)

        # Update position targets (250Hz thread handles publishing)
        self.robot.send_positions(target_dof_pos)

        # Update state
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _check_emergency_stop(self) -> bool:
        """L1 + R1 pressed simultaneously -> emergency stop."""
        return self.remote.LB.pressed and self.remote.RB.pressed

    def _handle_transitions(self) -> None:
        """Handle remote-triggered mode transitions."""
        if self.mode == RobotMode.IDLE:
            if self.remote.start.on_pressed:
                logger.info("Start pressed (from IDLE)")
                self._enter_gamepad()

        elif self.mode == RobotMode.GAMEPAD:
            if self.remote.Y.on_pressed:
                if self._can_switch_to_mocap():
                    logger.info("Y pressed -> entering MOCAP")
                    self._transition_to_mocap()
                else:
                    logger.warning("Cannot switch to MOCAP -- UDP check failed")

        elif self.mode == RobotMode.MOCAP:
            if self.remote.X.on_pressed:
                logger.info("X pressed -> exiting MOCAP to DAMPING")
                self._enter_damping()

        elif self.mode == RobotMode.DAMPING:
            if self.remote.start.on_pressed:
                logger.info("Start pressed (from DAMPING)")
                self._recover_from_damping()

    # ------------------------------------------------------------------
    # IDLE -> GAMEPAD
    # ------------------------------------------------------------------

    def _enter_gamepad(self) -> None:
        """Enter gamepad mode: verify ai_sport is active, create LocoClient."""
        result = self.robot.check_mode()
        mode_name = ""
        if result is not None:
            mode_name = result.get("name", "") if isinstance(result, dict) else str(result)

        if not mode_name:
            logger.warning(
                "No active mode -- cannot enter GAMEPAD. "
                "Please activate ai_sport via remote controller first."
            )
            return

        logger.info("Mode '%s' active -- creating LocoClient...", mode_name)
        try:
            self._loco_client = self._create_loco_client()
        except Exception as exc:
            logger.error("Failed to create LocoClient: %s", exc)
            return

        self.mode = RobotMode.GAMEPAD
        logger.info("Mode -> GAMEPAD (LocoClient active, joystick controls walking)")

    # ------------------------------------------------------------------
    # GAMEPAD -> MOCAP
    # ------------------------------------------------------------------

    def _can_switch_to_mocap(self) -> bool:
        """Verify UDP signal is stable and values are reasonable."""
        if not self.udp_provider.is_available():
            logger.warning("Mocap check: UDP provider not available")
            return False

        if not self.udp_provider._frame_ready.is_set():
            logger.warning("Mocap check: no UDP data received yet")
            return False

        valid_count = 0
        for _ in range(self._check_frames + 5):
            try:
                frame = self.udp_provider.get_frame()
            except TimeoutError:
                return False

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
                valid_count = 0

            if valid_count >= self._check_frames:
                return True

            time.sleep(0.02)

        logger.warning("Mocap check: only %d/%d valid frames", valid_count, self._check_frames)
        return False

    def _transition_to_mocap(self) -> None:
        """Switch from GAMEPAD -> MOCAP.

        1. Stop walking via LocoClient
        2. Enter debug mode (ReleaseMode loop)
        3. Lock all joints (creates LowCmd publisher + 250Hz thread)
        4. Read current state as initial retarget reference
        """
        # Stop walking
        if self._loco_client is not None:
            try:
                self._loco_client.StopMove()
            except Exception:
                pass
            time.sleep(0.5)
        self._loco_client = None

        # Enter debug mode (release "ai" -> low-level control)
        logger.info("Entering debug mode for low-level control...")
        ok = self.robot.enter_debug_mode()
        if not ok:
            logger.error("Failed to enter debug mode -- staying in GAMEPAD")
            # Recreate LocoClient since we destroyed it
            try:
                self._loco_client = self._create_loco_client()
            except Exception:
                logger.error("Failed to recreate LocoClient -- entering IDLE")
                self.mode = RobotMode.IDLE
            return
        time.sleep(0.5)

        # Lock all joints to current position (prevent collapse)
        logger.info("Locking joints to current position...")
        self.robot.lock_all_joints()
        time.sleep(0.5)

        # Read current state as initial
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._qpos_interpolator.start(init_qpos)

        # Reset observation builder history
        self.obs_builder.reset()

        self.mode = RobotMode.MOCAP
        logger.info("Mode -> MOCAP (debug mode, 250Hz LowCmd publishing)")

    # ------------------------------------------------------------------
    # Emergency stop / damping
    # ------------------------------------------------------------------

    def _enter_damping(self) -> None:
        """Enter damping mode from any state."""
        if self.mode == RobotMode.MOCAP:
            logger.info("DAMPING: sending LowCmd damping (from MOCAP)...")
            self.robot.set_damping()
            time.sleep(0.5)
            # Exit debug mode so user can re-activate ai_sport via remote
            logger.info("DAMPING: exiting debug mode...")
            self.robot.exit_debug_mode()
        elif self._loco_client is not None:
            logger.info("DAMPING: LocoClient.Damp() (from GAMEPAD)...")
            try:
                self._loco_client.Damp()
            except Exception:
                pass

        self._loco_client = None
        self.mode = RobotMode.DAMPING
        logger.info(
            "Mode -> DAMPING (restart robot and re-activate ai_sport "
            "via remote, then press Start)"
        )

    def _recover_from_damping(self) -> None:
        """Attempt recovery from DAMPING: check if ai_sport is active."""
        result = self.robot.check_mode()
        mode_name = ""
        if result is not None:
            mode_name = result.get("name", "") if isinstance(result, dict) else str(result)

        if not mode_name:
            logger.warning(
                "No active mode -- please re-activate ai_sport via "
                "remote controller, then press Start again."
            )
            return

        logger.info("ai_sport active (mode='%s') -- entering GAMEPAD...", mode_name)
        try:
            self._loco_client = self._create_loco_client()
            self.mode = RobotMode.GAMEPAD
            logger.info("Mode -> GAMEPAD (recovered from DAMPING)")
        except Exception as exc:
            logger.error("Failed to create LocoClient: %s", exc)

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

    def _validate_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        """Fail-fast validation for policy input observation dimension."""
        expected = getattr(self.policy, "_expected_obs_dim", None)
        if not isinstance(expected, int) or expected <= 0:
            return obs
        if obs.shape[0] != expected:
            raise ValueError(
                f"Observation dimension mismatch: obs_builder produced {obs.shape[0]}, "
                f"but policy expects {expected}. "
                "Use a matching mjlab-aligned ONNX policy; automatic pad/trim is disabled."
            )
        return obs

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
        """Clean shutdown."""
        logger.info("Shutting down Sim2RealController")
        if self.mode == RobotMode.MOCAP:
            try:
                self.robot.set_damping()
                time.sleep(0.5)
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
