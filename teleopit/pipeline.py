from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig

from teleopit.bus.in_process import InProcessBus
from teleopit.controllers.observation import VelCmdObservationBuilder
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs import BVHInputProvider, Pico4InputProvider
from teleopit.inputs.pico_video import PicoVideoRuntime, parse_pico_video_config
from teleopit.retargeting.core import RetargetingModule
from teleopit.robots.mujoco_robot import MuJoCoRobot
from teleopit.runtime.assets import PROJECT_ROOT
from teleopit.runtime.common import cfg_get
from teleopit.runtime.console import PlainConsole
from teleopit.runtime.factory import build_inference_components
from teleopit.sim.loop import SimulationLoop


class TeleopPipeline:
    def __init__(self, cfg: DictConfig | dict[str, Any], *, console: PlainConsole | None = None) -> None:
        self.cfg = cfg
        self._project_root = PROJECT_ROOT
        components = build_inference_components(
            cfg,
            self._project_root,
            robot_cls=MuJoCoRobot,
            controller_cls=RLPolicyController,
            obs_builder_cls=VelCmdObservationBuilder,
            bvh_input_cls=BVHInputProvider,
            pico4_input_cls=Pico4InputProvider,
            retargeter_cls=RetargetingModule,
        )

        self.robot = components.robot
        self.controller = components.controller
        self.obs_builder = components.obs_builder
        self.input_provider = components.input_provider
        self.retargeter = components.retargeter
        self.bus = InProcessBus()
        input_cfg = cfg_get(self.cfg, "input", {})
        self.video_runtime = PicoVideoRuntime(
            provider=self.input_provider,
            config=parse_pico_video_config(input_cfg),
            mode="sim2sim",
            robot=self.robot,
        )
        self.loop = SimulationLoop(
            cast(Any, self.robot),
            cast(Any, self.controller),
            cast(Any, self.obs_builder),
            cast(Any, self.bus),
            components.sim_cfg,
            viewers=components.viewers,
            video_runtime=self.video_runtime,
            console=console,
        )

    def run(self, num_steps: int) -> dict[str, float | int | str]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative (0 = infinite)")

        self.controller.reset()
        return dict(self.loop.run(cast(Any, self.input_provider), cast(Any, self.retargeter), num_steps=num_steps))
