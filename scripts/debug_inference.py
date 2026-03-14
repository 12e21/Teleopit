"""Diagnostic script: dump observation/action details for first N inference steps.

Usage:
    python scripts/debug_inference.py \
        controller.policy_path=ckpt/model_10000_loose_4096.onnx \
        input.bvh_file=data/lafan1/dance1_subject2.bvh
"""
from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline
from teleopit.runtime.cli import validate_policy_path


OBS_SLICES = {
    "command (motion_joint_pos)": (0, 29),
    "command (motion_joint_vel)": (29, 58),
    "motion_anchor_pos_b": (58, 61),
    "motion_anchor_ori_b": (61, 67),
    "base_lin_vel": (67, 70),
    "base_ang_vel": (70, 73),
    "joint_pos_rel": (73, 102),
    "joint_vel": (102, 131),
    "last_action": (131, 160),
}

JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
    "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
    "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
]


def _fmt_arr(a: np.ndarray) -> str:
    return f"min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}  std={a.std():.4f}"


def _print_joint_array(name: str, arr: np.ndarray) -> None:
    print(f"\n  {name} (29D):")
    for i, (jn, v) in enumerate(zip(JOINT_NAMES, arr)):
        print(f"    [{i:2d}] {jn:20s} = {v:+.6f}")


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="default")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "debug_inference.py")
    pipeline = TeleopPipeline(cfg)

    ctrl = pipeline.controller
    print("=" * 70)
    print("CONTROLLER CONFIG DIAGNOSTIC")
    print("=" * 70)

    action_scale = np.asarray(ctrl.action_scale)
    default_dof_pos = np.asarray(ctrl.default_dof_pos)

    print(f"\naction_scale shape: {action_scale.shape}")
    if action_scale.ndim == 0:
        print(f"  SCALAR: {action_scale.item():.6f}  (WARNING: should be per-joint array!)")
    else:
        _print_joint_array("action_scale", action_scale)

    _print_joint_array("default_dof_pos", default_dof_pos)

    print(f"\nclip_range: {ctrl.clip_range}")

    # Print PD gains and torque limits from the loop
    from teleopit.sim.loop import SimulationLoop

    loop: SimulationLoop = pipeline.loop

    kps = np.asarray(loop._kps)
    kds = np.asarray(loop._kds)
    torque_lim = np.asarray(loop._torque_limits)

    _print_joint_array("kps (PD stiffness)", kps)
    _print_joint_array("kds (PD damping)", kds)
    _print_joint_array("torque_limits", torque_lim)
    retargeter = pipeline.retargeter
    input_provider = pipeline.input_provider

    print("\n" + "=" * 70)
    print("INFERENCE STEP-BY-STEP DIAGNOSTIC (5 steps)")
    print("=" * 70)

    num_debug_steps = int(cfg.get("num_steps", 5))

    # Monkey-patch the loop to capture observation + action at each step
    orig_build = loop._build_observation
    orig_target = loop._compute_target_dof_pos
    step_data: list[dict] = []

    def patched_build(state, mimic_obs, last_action, retarget_qpos):
        obs = orig_build(state, mimic_obs, last_action, retarget_qpos)
        root_h = float(pipeline.robot.data.qpos[2]) if hasattr(pipeline.robot, "data") else 0.0
        step_data.append({"obs": obs.copy(), "last_action": last_action.copy(), "root_h": root_h})
        return obs

    def patched_target(action):
        target = orig_target(action)
        if step_data:
            step_data[-1]["raw_action"] = action.copy()
            step_data[-1]["target_dof_pos"] = target.copy()
            step_data[-1]["scaled_action"] = (target - default_dof_pos).copy()
        return target

    loop._build_observation = patched_build
    loop._compute_target_dof_pos = patched_target

    result = pipeline.run(num_steps=num_debug_steps, record=False)

    for i, sd in enumerate(step_data):
        print(f"\n{'─' * 60}")
        print(f"Step {i}")
        print(f"{'─' * 60}")

        obs = sd["obs"]
        print(f"\nObservation (160D) breakdown:")
        for name, (s, e) in OBS_SLICES.items():
            seg = obs[s:e]
            print(f"  [{s:3d}:{e:3d}] {name:30s}  {_fmt_arr(seg)}")

        if "raw_action" in sd:
            print(f"\nRaw ONNX action:   {_fmt_arr(sd['raw_action'])}")
            print(f"Scaled action:     {_fmt_arr(sd['scaled_action'])}")
            print(f"Target dof pos:    {_fmt_arr(sd['target_dof_pos'])}")

            if i == 0:
                _print_joint_array("raw_action", sd["raw_action"])
                _print_joint_array("target_dof_pos", sd["target_dof_pos"])

    # Print root height timeline
    print(f"\n{'=' * 70}")
    print("ROOT HEIGHT TIMELINE")
    print(f"{'=' * 70}")
    for i, sd in enumerate(step_data):
        rh = sd.get("root_h", 0.0)
        t = i * (1.0 / 50.0)
        bar = "#" * max(0, int(rh * 40))
        status = "OK" if rh > 0.5 else "FALLEN"
        if i % 10 == 0 or rh < 0.5:
            print(f"  step {i:4d} t={t:5.2f}s  root_h={rh:6.3f}  {bar}  {status}")

    print(f"\nLoop result: {result}")


if __name__ == "__main__":
    main()
