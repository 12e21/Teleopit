#!/usr/bin/env python3
"""Convert G1 URDF to USD using Isaac Lab's official UrdfConverter."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_DIR.parent
ASSET_PATH = (TRAIN_ROOT / "assets" / "g1" / "g1_custom_collision_29dof.urdf").resolve()
USD_DIR = TRAIN_ROOT / "assets" / "g1" / "usd"
USD_FILE_NAME = "g1_29dof.usd"


parser = argparse.ArgumentParser(description="Convert G1 URDF to USD with Isaac Lab UrdfConverter")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main() -> None:
    if not ASSET_PATH.is_file():
        raise FileNotFoundError(f"URDF not found: {ASSET_PATH}")

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=str(ASSET_PATH),
        usd_dir=str(USD_DIR),
        usd_file_name=USD_FILE_NAME,
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="none",
        ),
    )

    converter = UrdfConverter(urdf_converter_cfg)
    print(f"URDF: {ASSET_PATH}")
    print(f"USD: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
