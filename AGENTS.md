# AGENTS.md

## Project Overview

Teleopit is a lightweight, extensible, self-contained humanoid robot whole-body teleoperation framework. It integrates GMR (General Motion Retargeting) and supports TWIST2-compatible RL policy inference via ONNX.

Language: Python 3.10+
Package: `teleopit` (installed via `pip install -e .`)
Config: Hydra/OmegaConf YAML files in `teleopit/configs/`

## Architecture

```
InputProvider (BVH/VR) → Retargeter (GMR) → ObservationBuilder (1402D) → Controller (ONNX RL) → Robot (MuJoCo + PD)
```

Module-internal isolation: all modules run in-process, communicate via `InProcessBus` (zero-copy). Core interfaces defined as `typing.Protocol` in `teleopit/interfaces.py`.

## Directory Structure

```
teleopit/                 # Core package
├── interfaces.py         # Protocol definitions: Robot, Controller, InputProvider, Retargeter, etc.
├── pipeline.py           # TeleopPipeline — assembles and runs the full pipeline
├── bus/                  # InProcessBus message pub/sub
├── configs/              # Hydra YAML configs
│   ├── default.yaml      # Top-level config composing robot + controller + input
│   ├── robot/g1.yaml     # G1 robot: XML path, PD gains, default angles, action dims
│   ├── controller/rl_policy.yaml
│   └── input/bvh.yaml
├── controllers/
│   ├── rl_policy.py      # RLPolicyController — ONNX inference, returns RAW action (no scaling)
│   └── observation.py    # TWIST2ObservationBuilder — 1402D obs (127×11 history + 35 mimic)
├── inputs/
│   └── bvh_provider.py   # BVHInputProvider — parses lafan1/hc_mocap BVH formats
├── retargeting/
│   ├── core.py           # RetargetingModule + extract_mimic_obs()
│   └── gmr/              # Self-contained GMR (assets, IK solver, 17+ robot configs)
│       └── assets/unitree_g1/
│           ├── g1_mocap_29dof.xml    # Kinematic retarget only (has ctrlrange bug)
│           └── g1_sim2sim_29dof.xml  # Sim2sim with correct actuator limits
├── robots/
│   └── mujoco_robot.py   # MuJoCoRobot — MuJoCo sim wrapper
├── sim/
│   └── loop.py           # SimulationLoop — PD control at 1000Hz, policy at 50Hz
└── recording/            # HDF5Recorder
scripts/
├── run_sim.py            # Run teleoperation pipeline
├── render_sim.py         # Render single BVH → 3 videos (bvh skeleton, retarget, sim2sim), supports --format flag
└── render_all_lafan1.sh  # Batch render all data/lafan1/*.bvh
tests/                    # 78 pytest tests
data/                     # BVH motion data (gitignored)
├── lafan1/               # 77 BVH files, 30fps, 22 joints — working
├── hc_mocap/             # hc_mocap BVH files, 60fps, 50 joints, tab-separated, meters
└── lafan1-resolved/      # 77 BVH files, 60fps, 75 joints — retarget BROKEN (different skeleton)
outputs/                  # Rendered videos (gitignored)
```

## Key Technical Details

### Sim2Sim Pipeline
- Policy runs at 50Hz, PD control at 1000Hz (decimation=20), sim_dt=0.001
- Action flow: `compute_action()` returns RAW action → `get_target_dof_pos()` applies clip [-10,10] + scale 0.5 ONCE
- Must use `g1_sim2sim_29dof.xml` for sim2sim (not `g1_mocap_29dof.xml` which clamps torques to ±1 Nm)

### TWIST2 Observation
- 1402D vector: 127 features × 11 timesteps (history) + 35 mimic features
- Mimic obs from `extract_mimic_obs()`: target joint positions + velocities

### GMR Retargeting
- Self-contained in `teleopit/retargeting/gmr/` with all assets
- Supports lafan1-format BVH (22 joints, 30fps, centimeters, space-separated)
- Supports hc_mocap-format BVH (50 joints, 60fps→30fps downsampled, meters, tab-separated)
- lafan1-resolved format (75 joints, 6 channels) requires an adapter (not yet implemented)
- IK configs per format: `bvh_lafan1_to_g1.json`, `bvh_hc_mocap_to_g1.json`

### PD Gains (G1 robot, from g1.yaml)
- Most joints: kp varies by joint (see config)
- Wrist joints: kp=4.0, kd=0.2 (matches TWIST2 reference)

## Development

### Commit Policy
- **不要自动提交代码**。commit 必须由用户主动发起，确保测试通过后再提交。
- 使用 git 默认 user 作为 commit 作者。
- **完成大 feature 后必须更新文档**：当一个完整功能（如新格式支持、新模块）实现并测试通过后，需同步更新 AGENTS.md 和 README.md 中的相关说明（目录结构、使用方法、技术细节等），随代码一起提交。

```bash
pip install -e .           # Install in dev mode
pytest tests/ -v           # Run tests (78 tests)
```

### Rendering Videos
```bash
# Single BVH — lafan1 format (default)
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/lafan1/dance1_subject2.bvh

# Single BVH — hc_mocap format
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/motion_corrected_v2.bvh --format hc_mocap

# All lafan1 BVH files (skips already-rendered)
bash scripts/render_all_lafan1.sh --max_seconds 30
```

## Known Issues

1. **lafan1-resolved retarget broken**: Different BVH skeleton format (75 joints vs 22). Needs adapter layer to map joints. User deferred fix.
2. **g1_mocap_29dof.xml ctrlrange**: Has `ctrlrange="-1 1"` that clamps all torques. Only use for kinematic retarget visualization, never for sim2sim.

## Training Package (teleopit_train/)

Training code migrated from TWIST2, integrated as a separate package with one-way dependency (train → inference).

### Directory Structure

```
teleopit_train/               # Training package (pip install -e '.[train]')
├── __init__.py               # Version 0.1.0, TELEOPIT_TRAIN_ROOT_DIR
├── configs/                  # Dataset manifests
│   └── twist2_dataset.yaml   # TWIST2 motion dataset (1.9MB, 496 BVH files)
├── envs/                     # Isaac Lab environments
│   ├── g1_mimic_env.py       # G1MimicEnv with 8 tracking error functions
│   └── g1_mimic_cfg.py       # G1MimicEnvCfg, G1MimicPPORunnerCfg
├── scripts/                  # Training, export, evaluation
│   ├── train.py              # Isaac Lab training entry point
│   ├── save_onnx.py          # ONNX export
│   └── benchmark.py          # Policy evaluation with tracking errors
├── rsl_rl/                   # RL algorithms (PPO, DAgger)
├── pose/                     # Motion library (poselib)
└── assets/g1/                # Isaac Lab URDF + meshes
```

### Environment Setup & Training

See [docs/training.md](docs/training.md) for full setup and training instructions.
See [docs/assets.md](docs/assets.md) for USD/URDF asset management.
See [docs/troubleshooting.md](docs/troubleshooting.md) for known issues.

Quick reference:
 Conda env: `teleopit_isaaclab` (Python 3.11, Isaac Sim 5.1.0, Isaac Lab v2.3.2)
 Train: `python teleopit_train/scripts/train.py --task Isaac-G1-Mimic-v0 --num_envs 4096 --max_iterations 30000 --headless`
 Export: `python teleopit_train/scripts/save_onnx.py --checkpoint <path> --output policy.onnx`
 USD convert: `OMNI_KIT_ACCEPT_EULA=YES python teleopit_train/scripts/convert_urdf_isaaclab.py --headless`

### Key Technical Details

 **Isaac Lab DirectRLEnv**: Uses Isaac Lab's DirectRLEnv base class for GPU-accelerated simulation
 **Config system**: Python class-based (not Hydra) — nested inheritance chain
 **Training task**: `Isaac-G1-Mimic-v0` (registered in task_registry)
 **Checkpoint format**: `logs/rsl_rl/{experiment}/{run}/model_{iter}.pt`
 **Network**: ActorCriticFuture — Conv1D history encoder + MLP future encoder + [512,512,256,128] actor
 **One-way dependency**: teleopit_train imports from teleopit, never the reverse
 **USD asset**: Generated by Isaac Lab UrdfConverter (NOT custom pxr converter)
 **PhysX config**: max_depenetration_velocity=10.0, solver iterations 8/4, zero ImplicitActuator gains
 **Evaluation**: Use `benchmark.py` to compute 8 tracking errors on trained policies
 **Tracking errors**: joint_dof, joint_vel, root_translation, root_rotation, root_vel, root_ang_vel, keybody_pos, feet_slip
 **Wandb logging**: Episode metrics (rewards + errors) logged via `self.extras["episode"]` in `_reset_idx`