from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import cfg_get, cfg_set, normalize_path_in_cfg, parse_viewers, require_section


@dataclass(frozen=True)
class InferenceComponents:
    robot: Any
    controller: Any
    obs_builder: Any
    input_provider: Any
    retargeter: Any
    sim_cfg: dict[str, float | bool]
    viewers: set[str]


@dataclass(frozen=True)
class MocapComponents:
    input_provider: Any
    retargeter: Any
    controller: Any
    obs_builder: Any


class _LoopingInputProvider:
    def __init__(self, provider: Any) -> None:
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


def build_simulation_cfg(cfg: Any) -> dict[str, float | bool]:
    return {
        "policy_hz": float(cfg_get(cfg, "policy_hz", 50.0)),
        "pd_hz": float(cfg_get(cfg, "pd_hz", 1000.0)),
        "transition_duration": float(cfg_get(cfg, "transition_duration", 0.0) or 0.0),
        "realtime": bool(cfg_get(cfg, "realtime", False)),
    }


def propagate_controller_defaults(controller_cfg: Any, robot_cfg: Any) -> None:
    if cfg_get(controller_cfg, "default_dof_pos", None) is None:
        default_angles = cfg_get(robot_cfg, "default_angles", None)
        if default_angles is not None:
            cfg_set(controller_cfg, "default_dof_pos", list(default_angles))

    if cfg_get(controller_cfg, "action_scale", None) is None:
        robot_action_scale = cfg_get(robot_cfg, "action_scale", None)
        if robot_action_scale is not None:
            try:
                cfg_set(controller_cfg, "action_scale", list(robot_action_scale))
            except TypeError:
                cfg_set(controller_cfg, "action_scale", robot_action_scale)


def _prepare_policy_paths(robot_cfg: Any, controller_cfg: Any, project_root: Path) -> None:
    normalize_path_in_cfg(
        robot_cfg,
        "xml_path",
        base_dir=project_root,
        required=True,
        missing_message="robot.xml_path must be set",
    )
    normalize_path_in_cfg(
        controller_cfg,
        "policy_path",
        base_dir=project_root,
        required=True,
        missing_message=(
            "controller.policy_path must be set to an ONNX exported from train_mimic "
            "checkpoint. Deprecated TWIST2 default policy is removed."
        ),
    )


def _build_policy_components(
    *,
    robot_cfg: Any,
    controller_cfg: Any,
    project_root: Path,
    controller_cls: type[Any],
    obs_builder_cls: type[Any],
    sim2real: bool,
) -> tuple[Any, Any]:
    _prepare_policy_paths(robot_cfg, controller_cfg, project_root)
    propagate_controller_defaults(controller_cfg, robot_cfg)

    controller = controller_cls(controller_cfg)
    obs_type = str(cfg_get(robot_cfg, "obs_builder", "mjlab")).lower()
    if obs_type != "mjlab":
        raise ValueError(
            f"Unsupported robot.obs_builder='{obs_type}'. "
            "TWIST2 policy path is deprecated; use mjlab-aligned policy and set "
            "robot.obs_builder=mjlab."
        )

    has_state_estimation = bool(cfg_get(robot_cfg, "has_state_estimation", False))
    policy_dim = getattr(controller, "_expected_obs_dim", None)

    if sim2real:
        if has_state_estimation:
            raise ValueError(
                "Sim2real requires robot.has_state_estimation=false. The real robot does "
                "not provide base_pos/base_lin_vel, so only 154D mjlab ONNX policies "
                "exported from *-NoStateEst tasks are supported."
            )
        if policy_dim is not None and policy_dim != 154:
            raise ValueError(
                f"Sim2real only supports 154D mjlab ONNX policies exported from "
                f"*-NoStateEst tasks; got {policy_dim}D."
            )

    obs_cfg = {
        "num_actions": int(cfg_get(robot_cfg, "num_actions")),
        "default_dof_pos": list(cfg_get(robot_cfg, "default_angles")),
        "xml_path": str(cfg_get(robot_cfg, "xml_path")),
        "anchor_body_name": cfg_get(robot_cfg, "anchor_body_name", "torso_link"),
        "has_state_estimation": has_state_estimation,
    }
    obs_builder = obs_builder_cls(obs_cfg)

    builder_dim = getattr(obs_builder, "total_obs_size", None)
    if policy_dim is not None and builder_dim is not None and policy_dim != builder_dim:
        raise ValueError(
            f"Observation dimension mismatch at startup: obs_builder produces {builder_dim}D "
            f"but policy expects {policy_dim}D. Check has_state_estimation={has_state_estimation} "
            "and ensure the ONNX model matches."
        )
    return controller, obs_builder


def _prepare_input_cfg(input_cfg: Any, project_root: Path, *, sim2real: bool) -> str:
    provider_kind = str(cfg_get(input_cfg, "provider", "udp_bvh" if sim2real else "bvh")).lower()
    if provider_kind in ("bvh", "vr_stub"):
        normalize_path_in_cfg(
            input_cfg,
            "bvh_file",
            base_dir=project_root,
            required=True,
            missing_message="input.bvh_file must be set for offline BVH input",
        )
    elif provider_kind == "udp_bvh":
        normalize_path_in_cfg(
            input_cfg,
            "reference_bvh",
            base_dir=project_root,
            required=True,
            missing_message="input.reference_bvh must be set for udp_bvh provider",
        )
    elif provider_kind != "pico4":
        raise ValueError(
            f"Unsupported input.provider='{provider_kind}'. "
            "Supported providers are bvh, vr_stub, udp_bvh, pico4."
        )

    if sim2real and provider_kind not in ("udp_bvh", "pico4"):
        raise ValueError(
            f"Sim2real only supports udp_bvh or pico4 input providers; got '{provider_kind}'."
        )
    return provider_kind


def _build_input_provider(
    *,
    input_cfg: Any,
    provider_kind: str,
    bvh_input_cls: type[Any],
    pico4_input_cls: type[Any],
    udp_bvh_input_cls: type[Any],
) -> Any:
    if provider_kind == "pico4":
        return pico4_input_cls(
            human_format=str(cfg_get(input_cfg, "human_format", "xrobot")),
            timeout=float(cfg_get(input_cfg, "pico4_timeout", 60.0)),
        )

    if provider_kind == "udp_bvh":
        return udp_bvh_input_cls(
            reference_bvh=str(cfg_get(input_cfg, "reference_bvh")),
            host=str(cfg_get(input_cfg, "udp_host", "")),
            port=int(cfg_get(input_cfg, "udp_port", 1118)),
            human_format=str(cfg_get(input_cfg, "bvh_format", "hc_mocap")),
            timeout=float(cfg_get(input_cfg, "udp_timeout", 30.0)),
        )

    provider = bvh_input_cls(
        bvh_path=str(cfg_get(input_cfg, "bvh_file")),
        human_format=str(cfg_get(input_cfg, "bvh_format", "lafan1")),
    )
    if provider_kind == "vr_stub":
        return _LoopingInputProvider(provider)
    return provider


def _resolve_human_format(input_cfg: Any, input_provider: Any) -> str:
    if hasattr(input_provider, "human_format"):
        provider_format = input_provider.human_format
        provider_kind = str(cfg_get(input_cfg, "provider", "bvh")).lower()
        if provider_kind == "pico4":
            return str(provider_format)
        return f"bvh_{provider_format}"

    human_format = cfg_get(input_cfg, "human_format", None)
    if human_format and str(human_format) != "null":
        return str(human_format)

    if str(cfg_get(input_cfg, "provider", "bvh")).lower() == "pico4":
        return str(cfg_get(input_cfg, "human_format", "xrobot"))
    return f"bvh_{cfg_get(input_cfg, 'bvh_format', 'lafan1')}"


def _build_retargeter(input_cfg: Any, input_provider: Any, retargeter_cls: type[Any]) -> Any:
    return retargeter_cls(
        robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
        human_format=_resolve_human_format(input_cfg, input_provider),
        actual_human_height=float(cfg_get(input_cfg, "human_height", 1.75)),
    )


def build_inference_components(
    cfg: Any,
    project_root: Path,
    *,
    robot_cls: type[Any],
    controller_cls: type[Any],
    obs_builder_cls: type[Any],
    bvh_input_cls: type[Any],
    pico4_input_cls: type[Any],
    udp_bvh_input_cls: type[Any],
    retargeter_cls: type[Any],
) -> InferenceComponents:
    robot_cfg = require_section(cfg, "robot")
    controller_cfg = require_section(cfg, "controller")
    input_cfg = require_section(cfg, "input")

    provider_kind = _prepare_input_cfg(input_cfg, project_root, sim2real=False)
    controller, obs_builder = _build_policy_components(
        robot_cfg=robot_cfg,
        controller_cfg=controller_cfg,
        project_root=project_root,
        controller_cls=controller_cls,
        obs_builder_cls=obs_builder_cls,
        sim2real=False,
    )
    robot = robot_cls(robot_cfg)
    input_provider = _build_input_provider(
        input_cfg=input_cfg,
        provider_kind=provider_kind,
        bvh_input_cls=bvh_input_cls,
        pico4_input_cls=pico4_input_cls,
        udp_bvh_input_cls=udp_bvh_input_cls,
    )
    retargeter = _build_retargeter(input_cfg, input_provider, retargeter_cls)
    return InferenceComponents(
        robot=robot,
        controller=controller,
        obs_builder=obs_builder,
        input_provider=input_provider,
        retargeter=retargeter,
        sim_cfg=build_simulation_cfg(cfg),
        viewers=parse_viewers(cfg),
    )


def build_sim2real_mocap_components(
    cfg: Any,
    project_root: Path,
    *,
    controller_cls: type[Any],
    obs_builder_cls: type[Any],
    pico4_input_cls: type[Any],
    udp_bvh_input_cls: type[Any],
    retargeter_cls: type[Any],
) -> MocapComponents:
    robot_cfg = require_section(cfg, "robot")
    controller_cfg = require_section(cfg, "controller")
    input_cfg = require_section(cfg, "input")

    provider_kind = _prepare_input_cfg(input_cfg, project_root, sim2real=True)
    controller, obs_builder = _build_policy_components(
        robot_cfg=robot_cfg,
        controller_cfg=controller_cfg,
        project_root=project_root,
        controller_cls=controller_cls,
        obs_builder_cls=obs_builder_cls,
        sim2real=True,
    )
    input_provider = _build_input_provider(
        input_cfg=input_cfg,
        provider_kind=provider_kind,
        bvh_input_cls=object,
        pico4_input_cls=pico4_input_cls,
        udp_bvh_input_cls=udp_bvh_input_cls,
    )
    retargeter = _build_retargeter(input_cfg, input_provider, retargeter_cls)
    return MocapComponents(
        input_provider=input_provider,
        retargeter=retargeter,
        controller=controller,
        obs_builder=obs_builder,
    )
