from __future__ import annotations

from pathlib import Path
from typing import Any


_VALID_VIEWERS = frozenset({"bvh", "retarget", "sim2sim"})


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        return default if value is None else value
    return getattr(cfg, key, default)


def cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
        return
    if hasattr(cfg, "__setitem__"):
        try:
            cfg[key] = value
            return
        except Exception:
            pass
    setattr(cfg, key, value)


def parse_viewers(cfg: Any) -> set[str]:
    if cfg_get(cfg, "viewer", None) is not None:
        raise ValueError("Legacy config key 'viewer' has been removed. Use 'viewers' instead.")

    viewers_raw = cfg_get(cfg, "viewers", None)
    if viewers_raw is None:
        return set()

    if hasattr(viewers_raw, "__iter__") and not isinstance(viewers_raw, str):
        viewers = {str(v).strip().lower() for v in viewers_raw if str(v).strip()}
    else:
        raw = str(viewers_raw).strip().lower()
        if raw == "all":
            return set(_VALID_VIEWERS)
        if raw in ("none", "false", ""):
            return set()
        raw = raw.strip("'\"[]")
        viewers = {token.strip() for token in raw.split(",") if token.strip()}

    invalid = viewers.difference(_VALID_VIEWERS)
    if invalid:
        valid = ", ".join(sorted(_VALID_VIEWERS))
        raise ValueError(
            f"Unsupported viewers {sorted(invalid)}. Use a subset of [{valid}] or 'all'/'none'."
        )
    return viewers


def require_section(cfg: Any, key: str) -> Any:
    section = cfg_get(cfg, key, None)
    if section is None:
        raise ValueError(f"cfg must include a '{key}' section")
    return section


def normalize_path_in_cfg(
    cfg: Any,
    key: str,
    *,
    base_dir: Path,
    required: bool = False,
    must_exist: bool = True,
    missing_message: str | None = None,
) -> Path | None:
    raw = str(cfg_get(cfg, key, "")).strip()
    if not raw or raw == "None":
        if required:
            raise ValueError(missing_message or f"{key} must be set")
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{key} resolved to {path} which does not exist.")

    cfg_set(cfg, key, str(path))
    return path
