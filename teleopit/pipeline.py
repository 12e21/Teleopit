from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from teleopit.bus.in_process import InProcessBus
from teleopit.controllers.observation import TWIST2ObservationBuilder
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs import BVHInputProvider
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


class _LoopingInputProvider:
    def __init__(self, provider: BVHInputProvider) -> None:
        self._provider = provider

    def get_frame(self) -> dict[str, tuple[Any, Any]]:
        if not self._provider.is_available():
            self._provider.reset()
        return self._provider.get_frame()

    def is_available(self) -> bool:
        return True


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
        self.controller = RLPolicyController(controller_cfg)
        self.obs_builder = TWIST2ObservationBuilder(self._build_obs_cfg(robot_cfg))
        self.bus = InProcessBus()
        self.input_provider = self._build_input_provider(input_cfg)
        self.retargeter = RetargetingModule(
            robot_name=str(_cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(_cfg_get(input_cfg, "human_format", "bvh_lafan1")),
            actual_human_height=float(_cfg_get(input_cfg, "human_height", 1.75)),
        )

        sim_cfg = {
            "policy_hz": float(_cfg_get(cfg, "policy_hz", 50.0)),
            "pd_hz": float(_cfg_get(cfg, "pd_hz", 1000.0)),
        }
        self.loop = SimulationLoop(
            cast(Any, self.robot),
            cast(Any, self.controller),
            cast(Any, self.obs_builder),
            cast(Any, self.bus),
            sim_cfg,
        )

    def run(self, num_steps: int, record: bool = False) -> dict[str, float | int | str]:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

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
        bvh_path = str(_cfg_get(input_cfg, "bvh_file", ""))
        if not bvh_path:
            raise ValueError("input.bvh_file must be set")
        provider = BVHInputProvider(bvh_path=bvh_path, human_format=str(_cfg_get(input_cfg, "bvh_format", "lafan1")))

        if provider_kind == "vr_stub":
            return _LoopingInputProvider(provider)
        return provider

    def _build_obs_cfg(self, robot_cfg: Any) -> dict[str, Any]:
        return {
            "num_actions": int(_cfg_get(robot_cfg, "num_actions")),
            "ang_vel_scale": float(_cfg_get(robot_cfg, "ang_vel_scale", 0.25)),
            "dof_pos_scale": float(_cfg_get(robot_cfg, "dof_pos_scale", 1.0)),
            "dof_vel_scale": float(_cfg_get(robot_cfg, "dof_vel_scale", 0.05)),
            "ankle_idx": list(_cfg_get(robot_cfg, "ankle_idx", [4, 5, 10, 11])),
            "default_dof_pos": list(_cfg_get(robot_cfg, "default_angles")),
        }

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
                    fallback = self._project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
                    _cfg_set(robot_cfg, "xml_path", str(fallback.resolve()))

        if controller_cfg is not None:
            policy = str(_cfg_get(controller_cfg, "policy_path", ""))
            if not policy or policy == "None":
                default_policy = self._project_root.parent / "TWIST2" / "assets" / "ckpts" / "twist2_1017_20k.onnx"
                _cfg_set(controller_cfg, "policy_path", str(default_policy.resolve()))

        if input_cfg is not None:
            bvh_file = str(_cfg_get(input_cfg, "bvh_file", ""))
            if not bvh_file or bvh_file == "None":
                default_bvh = self._project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "xsens_bvh_test" / "251021_04_boxing_120Hz_cm_3DsMax.bvh"
                _cfg_set(input_cfg, "bvh_file", str(default_bvh.resolve()))
                if _cfg_get(input_cfg, "bvh_format", None) is None:
                    _cfg_set(input_cfg, "bvh_format", "lafan1")
                if _cfg_get(input_cfg, "human_format", None) is None:
                    _cfg_set(input_cfg, "human_format", "bvh_xsens")
