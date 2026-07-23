from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root() -> Path:
    raw = os.environ.get("TELEOPIT_PROJECT_ROOT")
    if raw is not None and raw.strip():
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise ValueError("TELEOPIT_PROJECT_ROOT must be an absolute path")
        return candidate.resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = resolve_project_root()
ROBOT_ASSETS_ROOT = PROJECT_ROOT / "assets" / "robots"
GMR_ASSETS_ROOT = PROJECT_ROOT / "teleopit" / "retargeting" / "gmr" / "assets"
UNITREE_G1_XML = ROBOT_ASSETS_ROOT / "unitree_g1" / "g1_29dof.xml"
UNITREE_G1_DEX3_XML = ROBOT_ASSETS_ROOT / "unitree_g1" / "g1_29dof_dex3.xml"
UNITREE_G1_AVP_O6_XML = ROBOT_ASSETS_ROOT / "unitree_g1" / "g1_29dof_avp_o6.xml"
UNITREE_G1_MJLAB_XML = UNITREE_G1_XML


def missing_gmr_assets_message(path: str | Path, *, label: str = "Required asset") -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return (
        f"{label} not found: {resolved}\n"
        "Set TELEOPIT_PROJECT_ROOT before starting Python to use an external resource root, or\n"
        "Download the external robot assets with:\n"
        "  python scripts/setup/download_assets.py --only robots gmr"
    )
