from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray


class _OrtValue(Protocol):
    name: str
    shape: Sequence[object]


class _OrtSession(Protocol):
    def get_inputs(self) -> Sequence[_OrtValue]: ...

    def get_outputs(self) -> Sequence[_OrtValue]: ...

    def run(self, output_names: Sequence[str], feed: dict[str, NDArray[np.float32]]) -> Sequence[object]: ...


class RLPolicyController:
    def __init__(self, cfg: object) -> None:
        self._session: _OrtSession
        self._input_name: str
        self._output_name: str
        self._expected_obs_dim: int | None
        self.action_scale: NDArray[np.float32]
        self.default_dof_pos: NDArray[np.float32]
        self.clip_range: tuple[float, float]

        policy_path = Path(str(self._cfg_get(cfg, "policy_path", ""))).expanduser()
        if not policy_path.is_file():
            raise FileNotFoundError(f"ONNX policy file not found: {policy_path}")
        if "twist2_1017_20k.onnx" in str(policy_path):
            raise ValueError(
                f"Deprecated policy is not supported: {policy_path}. "
                "Use ONNX exported from train_mimic checkpoint (mjlab-aligned)."
            )

        try:
            ort = importlib.import_module("onnxruntime")
        except ModuleNotFoundError as exc:
            raise ImportError("onnxruntime is required for RLPolicyController") from exc
        session_ctor = cast(object, getattr(ort, "InferenceSession"))
        providers_fn = cast(object, getattr(ort, "get_available_providers"))
        if not callable(session_ctor) or not callable(providers_fn):
            raise ImportError("onnxruntime missing required API")

        providers = self._select_providers(cast(Callable[[], Sequence[str]], providers_fn), str(self._cfg_get(cfg, "device", "auto")))
        self._session = cast(_OrtSession, session_ctor(str(policy_path), providers=providers))
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        self._expected_obs_dim = self._extract_feature_dim(self._session.get_inputs()[0].shape)
        _SUPPORTED_OBS_DIMS = {154, 160}
        if self._expected_obs_dim is not None and self._expected_obs_dim not in _SUPPORTED_OBS_DIMS:
            raise ValueError(
                f"Unsupported policy input dimension: {self._expected_obs_dim}. "
                f"Supported dimensions: {sorted(_SUPPORTED_OBS_DIMS)} "
                "(mjlab-aligned policies exported from train_mimic)."
            )

        raw_scale = self._cfg_get(cfg, "action_scale", None)
        self.action_scale = np.asarray(
            raw_scale if raw_scale is not None else 1.0,
            dtype=np.float32,
        )
        self.default_dof_pos = np.asarray(
            self._cfg_get(cfg, "default_dof_pos", self._cfg_get(cfg, "default_angles", [])),
            dtype=np.float32,
        )
        self.clip_range = self._normalize_clip_range(self._cfg_get(cfg, "clip_range", (-10.0, 10.0)))

    def compute_action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            if self._expected_obs_dim is not None and obs.shape[0] != self._expected_obs_dim:
                raise ValueError(
                    f"Observation dimension mismatch: expected {self._expected_obs_dim}, got {obs.shape[0]}"
                )
            obs = obs[np.newaxis, :]
        elif obs.ndim == 2 and obs.shape[0] == 1:
            if self._expected_obs_dim is not None and obs.shape[1] != self._expected_obs_dim:
                raise ValueError(
                    f"Observation dimension mismatch: expected {self._expected_obs_dim}, got {obs.shape[1]}"
                )
        else:
            raise ValueError(f"Observation must be shape (obs_dim,) or (1, obs_dim), got {obs.shape}")

        raw_action = np.asarray(
            self._session.run([self._output_name], {self._input_name: obs})[0],
            dtype=np.float32,
        ).reshape(-1)
        return raw_action

    def get_target_dof_pos(self, raw_action: NDArray[np.float32]) -> NDArray[np.float32]:
        scaled_action = self._clip_and_scale(np.asarray(raw_action, dtype=np.float32).reshape(-1))
        if self.default_dof_pos.size == 0:
            return scaled_action
        if scaled_action.shape[0] != self.default_dof_pos.shape[0]:
            raise ValueError(f"Action/default_dof_pos size mismatch: {scaled_action.shape[0]} vs {self.default_dof_pos.shape[0]}")
        return scaled_action + self.default_dof_pos

    def reset(self) -> None:
        return None

    def _clip_and_scale(self, raw_action: NDArray[np.float32]) -> NDArray[np.float32]:
        clipped = np.clip(raw_action, self.clip_range[0], self.clip_range[1])
        return np.asarray(clipped * self.action_scale, dtype=np.float32)

    @staticmethod
    def _cfg_get(cfg: object, key: str, default: object) -> object:
        if hasattr(cfg, "get"):
            value = cast(object, getattr(cfg, "get")(key))
            return default if value is None else value
        return cast(object, getattr(cfg, key, default))

    @staticmethod
    def _normalize_clip_range(raw: object) -> tuple[float, float]:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != 2:
                raise ValueError(f"clip_range sequence must contain exactly two values, got {raw}")
            low_raw = raw[0]
            high_raw = raw[1]
            if not isinstance(low_raw, (int, float)) or not isinstance(high_raw, (int, float)):
                raise ValueError(f"clip_range values must be numeric, got {raw}")
            low, high = float(low_raw), float(high_raw)
        else:
            if not isinstance(raw, (int, float)):
                raise ValueError(f"clip_range must be numeric or 2-length sequence, got {raw}")
            limit = float(raw)
            low, high = -abs(limit), abs(limit)
        if low > high:
            raise ValueError(f"clip_range lower bound must be <= upper bound, got ({low}, {high})")
        return low, high

    @staticmethod
    def _extract_feature_dim(shape: Sequence[object]) -> int | None:
        if len(shape) < 2:
            return None
        feature_dim = shape[-1]
        if isinstance(feature_dim, int):
            return feature_dim
        if isinstance(feature_dim, float):
            return int(feature_dim)
        if isinstance(feature_dim, str):
            try:
                return int(feature_dim)
            except ValueError:
                return None
        return None

    @staticmethod
    def _select_providers(get_available_providers: object, device: str) -> list[str]:
        if not callable(get_available_providers):
            return ["CPUExecutionProvider"]
        available = set(cast(Sequence[str], get_available_providers()))
        requested = device.lower()

        prefer_cuda = requested.startswith("cuda") or requested == "auto"
        providers: list[str] = []
        if prefer_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
