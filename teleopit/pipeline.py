from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from teleopit.bus.in_process import InProcessBus
from teleopit.controllers.observation import MjlabObservationBuilder
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs import BVHInputProvider, Pico4InputProvider, UDPBVHInputProvider
from teleopit.recording.hdf5_recorder import HDF5Recorder
from teleopit.retargeting.core import RetargetingModule
from teleopit.robots.mujoco_robot import MuJoCoRobot
from teleopit.runtime.factory import build_inference_components
from teleopit.sim.loop import SimulationLoop


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        return default if value is None else value
    return getattr(cfg, key, default)


class TeleopPipeline:
    def __init__(self, cfg: DictConfig | dict[str, Any]) -> None:
        self.cfg = cfg
        self._project_root = Path(__file__).resolve().parent.parent
        components = build_inference_components(
            cfg,
            self._project_root,
            robot_cls=MuJoCoRobot,
            controller_cls=RLPolicyController,
            obs_builder_cls=MjlabObservationBuilder,
            bvh_input_cls=BVHInputProvider,
            pico4_input_cls=Pico4InputProvider,
            udp_bvh_input_cls=UDPBVHInputProvider,
            retargeter_cls=RetargetingModule,
        )

        self.robot = components.robot
        self.controller = components.controller
        self.obs_builder = components.obs_builder
        self.input_provider = components.input_provider
        self.retargeter = components.retargeter
        self.bus = InProcessBus()
        self.loop = SimulationLoop(
            cast(Any, self.robot),
            cast(Any, self.controller),
            cast(Any, self.obs_builder),
            cast(Any, self.bus),
            components.sim_cfg,
            viewers=components.viewers,
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
