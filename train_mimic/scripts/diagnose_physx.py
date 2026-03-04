#!/usr/bin/env python3

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAttributeAccessIssue=false, reportOptionalCall=false, reportUnusedCallResult=false, reportMissingTypeStubs=false, reportUnusedImport=false

import os
import traceback

_ = os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import train_mimic.envs  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main() -> None:
    task_name = "Isaac-G1-Mimic-v0"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 4

    env = gym.make(task_name, cfg=env_cfg)
    try:
        base_env = env.unwrapped
        _ = env.reset()
        print("DIAG: reset_complete=True", flush=True)

        view = base_env.robot.root_physx_view

        def _safe_call(name: str):
            fn = getattr(view, name, None)
            if not callable(fn):
                return f"UNAVAILABLE: no attribute {name}"
            try:
                value = fn()
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().tolist()
                return value
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"

        stiffnesses = _safe_call("get_dof_stiffnesses")
        dampings = _safe_call("get_dof_dampings")
        pos_iters = _safe_call("get_solver_position_iteration_counts")
        vel_iters = _safe_call("get_solver_velocity_iteration_counts")

        print(f"DIAG: stiffness={stiffnesses}", flush=True)
        print(f"DIAG: damping={dampings}", flush=True)
        print(f"DIAG: solver_position_iters={pos_iters}", flush=True)
        print(f"DIAG: solver_velocity_iters={vel_iters}", flush=True)

        actions = torch.zeros((base_env.num_envs, base_env.cfg.num_actions), device=base_env.device)
        _ = env.step(actions)

        torque_max = torch.max(base_env.torques).item()
        torque_min = torch.min(base_env.torques).item()
        print(f"DIAG: torque_max={torque_max}", flush=True)
        print(f"DIAG: torque_min={torque_min}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"DIAG: exception={exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
