from __future__ import annotations

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf


def _install_render_sim_fakes(monkeypatch, *, builtin_pd: bool, provider_fps: int) -> dict[str, object]:
    render_sim_path = Path(__file__).resolve().parent.parent / "scripts" / "render_sim.py"
    spec = spec_from_file_location("test_render_sim_module", render_sim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load render_sim module from {render_sim_path}")
    render_sim = module_from_spec(spec)
    spec.loader.exec_module(render_sim)
    teleopit_pipeline = import_module("teleopit.pipeline")
    retarget_core = import_module("teleopit.retargeting.core")

    captured: dict[str, object] = {
        "aligned_root_xy": [],
        "set_actions": [],
        "controller_calls": 0,
        "step_count": 0,
    }

    num_actions = 1

    class FakeRobot:
        def __init__(self) -> None:
            self._builtin_pd = builtin_pd
            self.model = SimpleNamespace(
                vis=SimpleNamespace(global_=SimpleNamespace(offwidth=0, offheight=0))
            )
            self.data = SimpleNamespace(qpos=np.zeros(7 + num_actions, dtype=np.float64))

        def get_state(self) -> SimpleNamespace:
            return SimpleNamespace(
                qpos=np.zeros(num_actions, dtype=np.float32),
                qvel=np.zeros(num_actions, dtype=np.float32),
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                base_pos=np.array([7.0, 8.0, 0.76], dtype=np.float32),
            )

        def set_action(self, action: np.ndarray) -> None:
            cast_action = np.asarray(action, dtype=np.float32).copy()
            captured["set_actions"].append(cast_action)

        def step(self) -> None:
            captured["step_count"] = int(captured["step_count"]) + 1

    class FakeController:
        def compute_action(self, obs: np.ndarray) -> np.ndarray:
            captured["controller_calls"] = int(captured["controller_calls"]) + 1
            return np.array([2.5], dtype=np.float32)

    class FakeInputProvider:
        def __init__(self) -> None:
            self._frames = [
                {"pelvis": (np.array([float(i), 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))}
                for i in range(6)
            ]
            self.fps = provider_fps

        def __len__(self) -> int:
            return len(self._frames)

    class FakeRetargeter:
        def retarget(self, human_frame: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
            return np.array([1.0, 2.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float64)

    class FakeLoop:
        def __init__(self) -> None:
            self.decimation = 2
            self._num_actions = num_actions
            self._kps = np.array([100.0], dtype=np.float32)
            self._kds = np.array([0.0], dtype=np.float32)
            self._torque_limits = np.array([1000.0], dtype=np.float32)
            self._last_action = np.zeros(num_actions, dtype=np.float32)
            self._last_retarget_qpos = None

        def _retarget_to_qpos(self, retargeted: np.ndarray) -> np.ndarray:
            return np.asarray(retargeted, dtype=np.float64).copy()

        def _build_observation(
            self,
            state: object,
            mimic_obs: np.ndarray,
            last_action: np.ndarray,
            retarget_qpos: np.ndarray,
        ) -> np.ndarray:
            captured["aligned_root_xy"].append(np.asarray(retarget_qpos[:2], dtype=np.float64).copy())
            return np.array([0.0], dtype=np.float32)

        def _validate_observation_for_policy(self, obs: np.ndarray) -> np.ndarray:
            return obs

        def _compute_target_dof_pos(self, action: np.ndarray) -> np.ndarray:
            return np.array([3.0], dtype=np.float32)

    class FakePipeline:
        def __init__(self) -> None:
            self.robot = FakeRobot()
            self.controller = FakeController()
            self.input_provider = FakeInputProvider()
            self.retargeter = FakeRetargeter()
            self.loop = FakeLoop()

    class FakeRenderer:
        def __init__(self, model: object, height: int, width: int) -> None:
            self.model = model
            self.height = height
            self.width = width

        def update_scene(self, data: object, camera: object) -> None:
            return None

        def render(self) -> np.ndarray:
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    fake_pipeline = FakePipeline()

    def fake_load_configs(
        bvh_path: str,
        project_root: Path,
        bvh_format: str = "lafan1",
        policy_path: str | None = None,
    ) -> dict[str, object]:
        return {
            "robot": OmegaConf.create({"xml_path": "unused.xml"}),
            "controller": OmegaConf.create({}),
            "input": OmegaConf.create({}),
            "policy_hz": 50.0,
            "pd_hz": 100.0,
        }

    monkeypatch.setattr(render_sim, "_load_configs", fake_load_configs)
    monkeypatch.setattr(render_sim, "_make_camera", lambda: SimpleNamespace(lookat=np.zeros(3, dtype=np.float64)))
    monkeypatch.setattr(render_sim, "_write_video", lambda frames, path, fps: None)
    monkeypatch.setattr(render_sim.mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(teleopit_pipeline, "TeleopPipeline", lambda cfg: fake_pipeline)
    monkeypatch.setattr(retarget_core, "extract_mimic_obs", lambda qpos, last_qpos, dt: np.zeros(1, dtype=np.float32))
    captured["render_sim"] = render_sim
    return captured


def test_render_sim2sim_uses_provider_fps_aligns_root_and_respects_builtin_pd(monkeypatch, tmp_path: Path) -> None:
    captured = _install_render_sim_fakes(monkeypatch, builtin_pd=True, provider_fps=30)
    render_sim = captured["render_sim"]

    render_sim.render_sim2sim(
        bvh_path=tmp_path / "clip.bvh",
        output_path=tmp_path / "out.mp4",
        width=64,
        height=64,
        fps=60,
        bvh_format="lafan1",
        policy_path="policy.onnx",
    )

    assert captured["controller_calls"] == 10
    assert len(captured["set_actions"]) == 10
    assert all(np.allclose(action, [3.0]) for action in captured["set_actions"])
    assert all(np.allclose(xy, [7.0, 8.0]) for xy in captured["aligned_root_xy"])
    assert captured["step_count"] == 20


def test_render_sim2sim_external_pd_still_sends_torque(monkeypatch, tmp_path: Path) -> None:
    captured = _install_render_sim_fakes(monkeypatch, builtin_pd=False, provider_fps=30)
    render_sim = captured["render_sim"]

    render_sim.render_sim2sim(
        bvh_path=tmp_path / "clip.bvh",
        output_path=tmp_path / "out.mp4",
        width=64,
        height=64,
        fps=60,
        bvh_format="lafan1",
        policy_path="policy.onnx",
    )

    assert len(captured["set_actions"]) == 20
    assert all(np.allclose(action, [300.0]) for action in captured["set_actions"])
    assert captured["step_count"] == 20
