"""Sim2Real controller -- state machine + dual-mode control loop.

Supports two operating modes for a physical Unitree G1:
- **Standing**: RL policy maintains balance with fixed default-pose reference
- **Mocap**: RL policy tracks retargeted motion commands

State machine:
    IDLE ──Start──▶ STANDING ──Y──▶ MOCAP ──X──▶ STANDING
    Any  ──L1+R1──▶ DAMPING  ──Start──▶ STANDING
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.controllers.observation import MjlabObservationBuilder, align_motion_qpos_yaw
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.pico4_provider import Pico4InputProvider
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
    STANDING = "standing"  # Debug mode, RL policy holds default pose
    MOCAP = "mocap"        # Debug mode, RL policy tracks motion commands
    DAMPING = "damping"    # Emergency stop / recovery


class Sim2RealController:
    """G1 real-robot controller -- standing/mocap dual mode with state machine.

    Standing mode: enter debug mode, RL policy maintains balance at default pose.
    Mocap mode: RL policy tracks retargeted motion commands.
    Both modes share the same RL policy inference pipeline.
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

        # ---- Mocap pipeline (reuse existing components) ----
        input_cfg = _cfg_get(cfg, "input")
        controller_cfg = _cfg_get(cfg, "controller")
        robot_cfg = _cfg_get(cfg, "robot")

        # Build input provider based on config
        provider_kind = str(_cfg_get(input_cfg, "provider", "udp_bvh")).lower()

        if provider_kind == "pico4":
            self.input_provider: Pico4InputProvider | UDPBVHInputProvider = Pico4InputProvider(
                human_format=str(_cfg_get(input_cfg, "human_format", "xrobot")),
                timeout=float(_cfg_get(input_cfg, "pico4_timeout", 60.0)),
            )
            human_format = str(_cfg_get(input_cfg, "human_format", "xrobot"))
        else:
            # Default: UDP BVH provider
            ref_bvh = str(_cfg_get(input_cfg, "reference_bvh", ""))
            if ref_bvh and not Path(ref_bvh).is_absolute():
                ref_bvh = str((self._project_root / ref_bvh).resolve())

            self.input_provider = UDPBVHInputProvider(
                reference_bvh=ref_bvh,
                host=str(_cfg_get(input_cfg, "udp_host", "")),
                port=int(_cfg_get(input_cfg, "udp_port", 1118)),
                human_format=str(_cfg_get(input_cfg, "bvh_format", "hc_mocap")),
                timeout=float(_cfg_get(input_cfg, "udp_timeout", 30.0)),
            )

            # Use provider's human_format (may be auto-adjusted, e.g. hc_mocap_no_toe)
            if hasattr(self.input_provider, "human_format"):
                human_format = f"bvh_{self.input_provider.human_format}"
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
            if not candidate.exists():
                raise FileNotFoundError(
                    f"robot.xml_path resolved to {candidate} which does not exist. "
                    "Set an absolute path in robot config."
                )
            xml_path = str(candidate)
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

        # ---- Standing mode reference qpos ----
        self._standing_qpos = np.zeros(36, dtype=np.float64)
        self._standing_qpos[3] = 1.0  # identity quaternion w=1
        self._standing_qpos[7:36] = self.default_angles.astype(np.float64)

        # ---- Policy state (shared by STANDING and MOCAP) ----
        self._last_action: Float32Array = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None

        # ---- Mocap switch safety ----
        mocap_sw = _cfg_get(cfg, "mocap_switch", {})
        self._check_frames: int = int(_cfg_get(mocap_sw, "check_frames", 10))
        self._max_pos_value: float = float(_cfg_get(mocap_sw, "max_position_value", 5.0))

        logger.info(
            "Sim2RealController ready | mode=IDLE | policy_hz=%.0f",
            self.policy_hz,
        )

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop at policy_hz."""
        logger.info(
            "Control loop started | mode=IDLE | press Start to enter STANDING"
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
                if self.mode == RobotMode.STANDING:
                    self._standing_step()
                elif self.mode == RobotMode.MOCAP:
                    self._mocap_step()

                # 5. Rate control
                self._sleep_until(t0, dt)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt -- shutting down")

    # ------------------------------------------------------------------
    # Mode execution
    # ------------------------------------------------------------------

    def _standing_step(self) -> None:
        """Standing mode: feed fixed default-pose reference to RL policy."""
        robot_state = self.robot.get_state()

        # Build standing reference qpos aligned to robot's current yaw
        qpos = self._standing_qpos.copy()
        align_motion_qpos_yaw(
            np.asarray(robot_state.quat, dtype=np.float32), qpos
        )

        # Standing → zero joint velocity reference
        motion_joint_vel = np.zeros(self.num_actions, dtype=np.float32)

        obs = self.obs_builder.build(
            robot_state,
            np.asarray(qpos[:7 + self.num_actions], dtype=np.float32),
            motion_joint_vel,
            self._last_action,
        )
        obs = self._validate_observation_for_policy(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        self.robot.send_positions(target_dof_pos)

        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()

    def _mocap_step(self) -> None:
        """Mocap mode: input provider -> retarget -> policy -> update LowCmd targets."""
        if not self.input_provider.is_available():
            logger.warning("Input provider unavailable -- entering damping")
            self._enter_damping()
            return

        try:
            human_frame = self.input_provider.get_frame()
        except (TimeoutError, RuntimeError):
            logger.warning("Input provider error -- entering damping")
            self._enter_damping()
            return

        # Retarget -> mimic observation
        retargeted = self.retargeter.retarget(human_frame)
        qpos = self._retarget_to_qpos(retargeted)
        qpos = self._qpos_interpolator.apply(qpos)

        # Robot state from SDK
        robot_state = self.robot.get_state()

        # Align motion root yaw to robot's current yaw heading.
        align_motion_qpos_yaw(np.asarray(robot_state.quat, dtype=np.float32), qpos)

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
                self._enter_standing()

        elif self.mode == RobotMode.STANDING:
            if self.remote.Y.on_pressed:
                if self._can_switch_to_mocap():
                    logger.info("Y pressed -> entering MOCAP")
                    self._transition_to_mocap()
                else:
                    logger.warning("Cannot switch to MOCAP -- input check failed")

        elif self.mode == RobotMode.MOCAP:
            if self.remote.X.on_pressed:
                logger.info("X pressed -> returning to STANDING")
                self._enter_standing()

        elif self.mode == RobotMode.DAMPING:
            if self.remote.start.on_pressed:
                logger.info("Start pressed (from DAMPING)")
                self._enter_standing()

    # ------------------------------------------------------------------
    # Enter STANDING (from IDLE, MOCAP, or DAMPING)
    # ------------------------------------------------------------------

    def _enter_standing(self) -> None:
        """Enter standing mode: debug mode + RL policy holds default pose.

        Works from IDLE, MOCAP (already in debug mode), or DAMPING.
        """
        already_in_debug = self.mode in (RobotMode.STANDING, RobotMode.MOCAP)

        if not already_in_debug:
            logger.info("Entering debug mode...")
            ok = self.robot.enter_debug_mode()
            if not ok:
                logger.error("Failed to enter debug mode -- staying in %s", self.mode.value)
                return
            time.sleep(0.5)

        # Lock joints to current position (prevent collapse during init)
        logger.info("Locking joints to current position...")
        self.robot.lock_all_joints()
        time.sleep(0.3)

        # Initialize policy state
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_builder.reset()

        self.mode = RobotMode.STANDING
        logger.info("Mode -> STANDING (RL policy maintaining balance at default pose)")

    # ------------------------------------------------------------------
    # STANDING -> MOCAP
    # ------------------------------------------------------------------

    def _can_switch_to_mocap(self) -> bool:
        """Verify input signal is stable and values are reasonable."""
        if not self.input_provider.is_available():
            logger.warning("Mocap check: input provider not available")
            return False

        # For UDP BVH provider, also check if initial data has been received
        if hasattr(self.input_provider, "_frame_ready"):
            if not self.input_provider._frame_ready.is_set():
                logger.warning("Mocap check: no data received yet")
                return False

        # For Pico4 provider, check SDK data availability before calling
        # get_frame() to avoid blocking the control loop for up to
        # pico4_timeout on first frame.
        if hasattr(self.input_provider, "_xrt"):
            if not self.input_provider._xrt.is_body_data_available():
                logger.warning("Mocap check: Pico4 body data not available yet")
                return False

        valid_count = 0
        for _ in range(self._check_frames + 5):
            try:
                frame = self.input_provider.get_frame()
            except (TimeoutError, RuntimeError):
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
        """Switch from STANDING -> MOCAP.

        Already in debug mode from STANDING, so just read current state
        and start interpolation for smooth transition.
        """
        # Read current state as initial reference for interpolation
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
        logger.info("Mode -> MOCAP (tracking motion commands)")

    # ------------------------------------------------------------------
    # Emergency stop / damping
    # ------------------------------------------------------------------

    def _enter_damping(self) -> None:
        """Enter damping mode from any state."""
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP):
            logger.info("DAMPING: sending LowCmd damping...")
            self.robot.set_damping()
            time.sleep(0.5)
            logger.info("DAMPING: exiting debug mode...")
            self.robot.exit_debug_mode()

        self.mode = RobotMode.DAMPING
        logger.info("Mode -> DAMPING (press Start to re-enter STANDING)")

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
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP):
            try:
                self.robot.set_damping()
                time.sleep(0.5)
            except Exception:
                pass
            try:
                self.robot.exit_debug_mode()
            except Exception:
                pass
        try:
            self.input_provider.close()
        except Exception:
            pass
        try:
            self.robot.close()
        except Exception:
            pass
