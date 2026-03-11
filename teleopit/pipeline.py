from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from teleopit.bus.in_process import InProcessBus
from teleopit.controllers.observation import MjlabObservationBuilder
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs import BVHInputProvider, UDPBVHInputProvider
from teleopit.recording.hdf5_recorder import HDF5Recorder
from teleopit.retargeting.core import RetargetingModule
from teleopit.robots.mujoco_robot import MuJoCoRobot
from teleopit.sim.loop import SimulationLoop


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        return default if value is None else value
    return getattr(cfg, key, default)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
        return
    setattr(cfg, key, value)


def _parse_viewers(cfg: Any) -> set[str]:
    """Parse viewers config into a set of viewer names.

    Supports the new ``viewers`` key (comma-separated string or the
    special values ``"all"`` / ``"none"``) as well as the legacy
    ``viewer: true/false`` boolean.
    """
    viewers_raw = _cfg_get(cfg, "viewers", None)
    if viewers_raw is not None:
        # Support Hydra list syntax (e.g. [retarget,sim2sim])
        if hasattr(viewers_raw, "__iter__") and not isinstance(viewers_raw, str):
            return {str(v).strip().lower() for v in viewers_raw if str(v).strip()}
        s = str(viewers_raw).strip().lower()
        if s == "all":
            return {"bvh", "retarget", "sim2sim"}
        if s in ("none", "false", ""):
            return set()
        # Strip surrounding quotes/brackets from Hydra
        s = s.strip("'\"[]")
        return {v.strip() for v in s.split(",") if v.strip()}

    # Backward compat: legacy ``viewer: true/false``
    legacy = _cfg_get(cfg, "viewer", None)
    if legacy is not None:
        return {"sim2sim"} if bool(legacy) else set()

    return set()


class _LoopingInputProvider:
    def __init__(self, provider: BVHInputProvider) -> None:
        self._provider = provider

    def get_frame(self) -> dict[str, tuple[Any, Any]]:
        if not self._provider.is_available():
            self._provider.reset()
        return self._provider.get_frame()

    def is_available(self) -> bool:
        return True

    @property
    def fps(self) -> int:
        return self._provider.fps

    @property
    def bone_names(self) -> list[str]:
        return self._provider.bone_names

    @property
    def bone_parents(self) -> Any:
        return self._provider.bone_parents


class TeleopPipeline:
    def __init__(self, cfg: DictConfig | dict[str, Any]) -> None:
        self.cfg = cfg
        self._project_root = Path(__file__).resolve().parent.parent

        self._prepare_cfg_paths()

        robot_cfg = cast(Any, _cfg_get(cfg, "robot"))
        controller_cfg = cast(Any, _cfg_get(cfg, "controller"))
        input_cfg = cast(Any, _cfg_get(cfg, "input"))

        if robot_cfg is None or controller_cfg is None or input_cfg is None:
            raise ValueError("cfg must include robot/controller/input sections")

        self.robot = MuJoCoRobot(robot_cfg)

        # Ensure controller has default_dof_pos from robot config
        if _cfg_get(controller_cfg, "default_dof_pos", None) is None:
            default_angles = _cfg_get(robot_cfg, "default_angles", None)
            if default_angles is not None:
                _cfg_set(controller_cfg, "default_dof_pos", list(default_angles))

        # Propagate action_scale from robot config only when controller
        # has no explicit value (null). Explicit overrides (e.g. 0.0 in tests)
        # are preserved.
        if _cfg_get(controller_cfg, "action_scale", None) is None:
            robot_action_scale = _cfg_get(robot_cfg, "action_scale", None)
            if robot_action_scale is not None:
                try:
                    _cfg_set(controller_cfg, "action_scale", list(robot_action_scale))
                except TypeError:
                    _cfg_set(controller_cfg, "action_scale", robot_action_scale)

        self.controller = RLPolicyController(controller_cfg)
        self.obs_builder = self._build_obs_builder(robot_cfg)
        self.bus = InProcessBus()
        self.input_provider = self._build_input_provider(input_cfg)

        # Use provider's human_format (may be auto-adjusted, e.g. hc_mocap_no_toe)
        if hasattr(self.input_provider, "human_format"):
            human_format = f"bvh_{self.input_provider.human_format}"
        else:
            human_format = _cfg_get(input_cfg, "human_format", None)
            if not human_format or str(human_format) == "null":
                bvh_format = str(_cfg_get(input_cfg, "bvh_format", "lafan1"))
                human_format = f"bvh_{bvh_format}"

        self.retargeter = RetargetingModule(
            robot_name=str(_cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(human_format),
            actual_human_height=float(_cfg_get(input_cfg, "human_height", 1.75)),
        )

        sim_cfg = {
            "policy_hz": float(_cfg_get(cfg, "policy_hz", 50.0)),
            "pd_hz": float(_cfg_get(cfg, "pd_hz", 1000.0)),
        }
        viewer_set = _parse_viewers(cfg)
        self.loop = SimulationLoop(
            cast(Any, self.robot),
            cast(Any, self.controller),
            cast(Any, self.obs_builder),
            cast(Any, self.bus),
            sim_cfg,
            viewers=viewer_set,
        )

    def run(self, num_steps: int, record: bool = False) -> dict[str, float | int | str]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative (0 = infinite)")

        if not record:
            return dict(self.loop.run(cast(Any, self.input_provider), cast(Any, self.retargeter), num_steps=num_steps))

        rec_cfg = cast(Any, _cfg_get(self.cfg, "recording", {}))
        output_path = Path(str(_cfg_get(rec_cfg, "output_path", "teleop_session.h5"))).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with HDF5Recorder(output_path) as recorder:
            result = self.loop.run(
                cast(Any, self.input_provider),
                cast(Any, self.retargeter),
                num_steps=num_steps,
                recorder=cast(Any, recorder),
            )

        result_with_path: dict[str, float | int | str] = dict(result)
        result_with_path["record_path"] = str(output_path)
        return result_with_path

    def _build_input_provider(self, input_cfg: Any) -> Any:
        provider_kind = str(_cfg_get(input_cfg, "provider", "bvh")).lower()

        if provider_kind == "udp_bvh":
            ref_bvh = str(_cfg_get(input_cfg, "reference_bvh", ""))
            if not ref_bvh:
                raise ValueError("input.reference_bvh must be set for udp_bvh provider")
            ref_path = Path(ref_bvh)
            if not ref_path.is_absolute():
                ref_path = (self._project_root / ref_path).resolve()
            return UDPBVHInputProvider(
                reference_bvh=str(ref_path),
                host=str(_cfg_get(input_cfg, "udp_host", "")),
                port=int(_cfg_get(input_cfg, "udp_port", 1118)),
                human_format=str(_cfg_get(input_cfg, "bvh_format", "hc_mocap")),
                timeout=float(_cfg_get(input_cfg, "udp_timeout", 30.0)),
            )

        bvh_path = str(_cfg_get(input_cfg, "bvh_file", ""))
        if not bvh_path:
            raise ValueError("input.bvh_file must be set")
        provider = BVHInputProvider(bvh_path=bvh_path, human_format=str(_cfg_get(input_cfg, "bvh_format", "lafan1")))

        if provider_kind == "vr_stub":
            return _LoopingInputProvider(provider)
        return provider

    def _build_obs_builder(self, robot_cfg: Any) -> Any:
        """Build observation builder. Only mjlab tracking observations are supported."""
        obs_type = str(_cfg_get(robot_cfg, "obs_builder", "mjlab")).lower()
        if obs_type != "mjlab":
            raise ValueError(
                f"Unsupported robot.obs_builder='{obs_type}'. "
                "TWIST2 policy path is deprecated; use mjlab-aligned policy and set robot.obs_builder=mjlab."
            )
        xml_path = str(_cfg_get(robot_cfg, "xml_path", ""))
        obs_cfg = {
            "num_actions": int(_cfg_get(robot_cfg, "num_actions")),
            "default_dof_pos": list(_cfg_get(robot_cfg, "default_angles")),
            "xml_path": xml_path,
            "anchor_body_name": _cfg_get(robot_cfg, "anchor_body_name", "torso_link"),
        }
        return MjlabObservationBuilder(obs_cfg)

    def _prepare_cfg_paths(self) -> None:
        robot_cfg = cast(Any, _cfg_get(self.cfg, "robot"))
        controller_cfg = cast(Any, _cfg_get(self.cfg, "controller"))
        input_cfg = cast(Any, _cfg_get(self.cfg, "input"))

        if robot_cfg is not None:
            xml_path = Path(str(_cfg_get(robot_cfg, "xml_path", "")))
            if not xml_path.is_absolute():
                candidate = (self._project_root / xml_path).resolve()
                if candidate.exists():
                    _cfg_set(robot_cfg, "xml_path", str(candidate))
                else:
                    fallback = self._project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_sim2sim_29dof.xml"
                    _cfg_set(robot_cfg, "xml_path", str(fallback.resolve()))

        if controller_cfg is not None:
            policy = str(_cfg_get(controller_cfg, "policy_path", ""))
            if not policy or policy == "None":
                raise ValueError(
                    "controller.policy_path must be set to an ONNX exported from train_mimic checkpoint. "
                    "Deprecated TWIST2 default policy is removed."
                )
            policy_path = Path(policy).expanduser()
            if not policy_path.is_absolute():
                policy_path = (self._project_root / policy_path).resolve()
                _cfg_set(controller_cfg, "policy_path", str(policy_path))

        if input_cfg is not None:
            provider_kind = str(_cfg_get(input_cfg, "provider", "bvh")).lower()
            if provider_kind != "udp_bvh":
                bvh_file = str(_cfg_get(input_cfg, "bvh_file", ""))
                if not bvh_file or bvh_file == "None":
                    default_bvh = self._project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "xsens_bvh_test" / "251021_04_boxing_120Hz_cm_3DsMax.bvh"
                    _cfg_set(input_cfg, "bvh_file", str(default_bvh.resolve()))
                    if _cfg_get(input_cfg, "bvh_format", None) is None:
                        _cfg_set(input_cfg, "bvh_format", "lafan1")
                    if _cfg_get(input_cfg, "human_format", None) is None:
                        _cfg_set(input_cfg, "human_format", "bvh_xsens")
