# AGENTS.md

## Project Overview

Teleopit is a lightweight, extensible, self-contained humanoid robot whole-body teleoperation framework. It integrates GMR (General Motion Retargeting) and supports TWIST2-compatible RL policy inference via ONNX.

Language: Python 3.10+
Package: `teleopit` (installed via `pip install -e .`)
Config: Hydra/OmegaConf YAML files in `teleopit/configs/`

## Architecture

```
InputProvider (BVH file / UDP realtime / VR) → Retargeter (GMR) → ObservationBuilder (1402D) → Controller (ONNX RL) → Robot (MuJoCo + PD)
```

Module-internal isolation: all modules run in-process, communicate via `InProcessBus` (zero-copy). Core interfaces defined as `typing.Protocol` in `teleopit/interfaces.py`.

## Directory Structure

```
teleopit/                 # Core package
├── interfaces.py         # Protocol definitions: Robot, Controller, InputProvider, Retargeter, etc.
├── pipeline.py           # TeleopPipeline — assembles and runs the full pipeline
├── bus/                  # InProcessBus message pub/sub
├── configs/              # Hydra YAML configs
│   ├── default.yaml      # Offline sim top-level config: viewers, policy_hz, pd_hz
│   ├── online.yaml       # Online sim2sim top-level config: realtime=true, num_steps=0
│   ├── robot/g1.yaml     # G1 robot: XML path, PD gains, default angles, action dims
│   ├── controller/rl_policy.yaml
│   ├── input/bvh.yaml    # Offline BVH file input
│   └── input/udp_bvh.yaml # UDP realtime BVH input (reference_bvh, port, timeout)
├── controllers/
│   ├── rl_policy.py      # RLPolicyController — ONNX inference, returns RAW action (no scaling)
│   └── observation.py    # TWIST2ObservationBuilder — 1402D obs (127×11 history + 35 mimic)
├── inputs/
│   ├── bvh_provider.py       # BVHInputProvider — offline BVH file; exposes fps, bone_names, bone_parents
│   └── udp_bvh_provider.py   # UDPBVHInputProvider — realtime UDP BVH; daemon receiver thread
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
├── run_sim.py            # Offline BVH sim2sim pipeline
├── run_online_sim.py     # Online realtime UDP sim2sim (uses online.yaml)
├── send_bvh_udp.py       # UDP BVH test sender (--bvh, --loop, --fps, --downsample)
├── render_sim.py         # Render single BVH → 3 videos (bvh skeleton, retarget, sim2sim), supports --format flag
├── render_all_lafan1.sh  # Batch render all data/lafan1/*.bvh
└── compute_ik_offsets.py # Compute IK quaternion offsets for new BVH formats (see IK Offset Calibration)
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
- Action flow: `compute_action()` returns RAW action → `get_target_dof_pos()` applies clip [-10,10] + scale 0.5 + default_dof_pos
- Must use `g1_sim2sim_29dof.xml` for sim2sim (not `g1_mocap_29dof.xml` which clamps torques to ±1 Nm)

### Multi-Viewer Support
`SimulationLoop` supports three simultaneous viewer windows controlled by the `viewers` config:

```bash
python scripts/run_sim.py viewers=sim2sim                  # default: physics sim only
python scripts/run_sim.py 'viewers=[bvh,retarget,sim2sim]'  # all three windows
python scripts/run_sim.py viewers=all                      # same as above
python scripts/run_sim.py 'viewers=[retarget,sim2sim]'     # skip BVH skeleton
python scripts/run_sim.py viewers=none                     # headless
```

| Viewer | Backend | Update method |
|--------|---------|---------------|
| **sim2sim** | MuJoCo (subprocess) | Reads post-physics `robot.data.qpos` via shared memory → `mj_forward` |
| **retarget** | MuJoCo (subprocess) | Reads retarget `qpos` via shared memory → `mj_forward` → foot Z correction |
| **bvh** | matplotlib 3D (subprocess) | Reads bone positions via shared memory → scatter + line plot (matches `render_sim.py`) |

- All viewers run in **separate subprocesses** (GLFW/GLX only supports one window per process)
- Data exchange: main process writes qpos / mocap data to `multiprocessing.Array` shared memory
- BVH viewer is created lazily in `run()` (needs `bone_names`/`bone_parents` from `InputProvider`)
- Retarget viewer foot Z correction uses `left_ankle_roll_link` / `right_ankle_roll_link` body IDs
- Simulation breaks when **all** active viewer windows are closed
- Backward compatible: `+viewer=true` maps to `viewers=sim2sim`, `+viewer=false` maps to `viewers=none`
- Hydra quoting: multi-viewer values with commas need shell quotes, e.g. `'viewers=[retarget,sim2sim]'`

**default_dof_pos 传递（关键）**：RL policy 输出的 action 是相对于默认站姿的**偏移量**，目标关节角计算公式为：

```
target_dof_pos = clip(action, -10, 10) × action_scale + default_dof_pos
```

`default_dof_pos` 来自 `robot/g1.yaml` 的 `default_angles`（膝盖微屈 0.4 rad、肘部弯曲 1.2 rad 等）。`TeleopPipeline` 在初始化时自动将 `robot_cfg.default_angles` 传递给 `controller_cfg.default_dof_pos`。若此传递缺失，target 会缺少站姿偏移（膝盖 0→伸直、肘部 0→伸直），机器人无法维持平衡。

### Online Sim2Sim (UDP 实时输入)

支持通过 UDP 接收实时动捕 BVH 数据，每个 UDP 包 = 一行 BVH motion data（159 floats for hc_mocap），接收频率约 30Hz。

```bash
# Terminal 1: start online sim
python scripts/run_online_sim.py viewers=sim2sim

# Terminal 2: send test data
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

**`UDPBVHInputProvider` 关键设计**：
- 从 `reference_bvh` 解析骨骼元数据（bone_names, parents, offsets, euler_order, channels）
- 后台 daemon 线程 `recvfrom` → 逐帧处理（`process_single_bvh_frame()`）
- `_latest_frame` 通过 `threading.Lock` 保护，`_frame_ready` 通过 `threading.Event` 阻塞首帧
- `get_frame()` 始终返回最新帧（无内部计数器），与 `SimulationLoop` 的 `bvh_idx` 兼容
- `is_available()` 持续返回 True → 循环不因 input 耗尽退出
- `fps=30`（固定），`human_height` 从 reference BVH frame-0 FK 计算

**`SimulationLoop` 变更**：
- 新增 `realtime` config flag：即使无 viewer 也进行 wall-clock 限速
- `num_steps=0` 表示无限循环（`max_steps = 2**63`）
- 添加 `KeyboardInterrupt` 处理，Ctrl+C 优雅退出

### 帧率对齐
- BVH 输入帧率（如 hc_mocap 30fps）可能与 policy 频率（50Hz）不同
- `SimulationLoop` 按时间对齐：`bvh_idx = int(policy_time × input_fps)`，多个 policy step 复用同一 BVH 帧
- `bvh_provider.py` 的 `fps` 属性返回降采样后的实际帧率（hc_mocap 60→30fps）

### TWIST2 Observation
- 1402D vector: 127 features × 11 timesteps (history) + 35 mimic features
- Mimic obs from `extract_mimic_obs()`: target joint positions + velocities

### GMR Retargeting
- Self-contained in `teleopit/retargeting/gmr/` with all assets
- Supports lafan1-format BVH (22 joints, 30fps, centimeters, space-separated)
- Supports hc_mocap-format BVH (50 joints, 60fps→30fps downsampled, meters, tab-separated)
- lafan1-resolved format (75 joints, 6 channels) requires an adapter (not yet implemented)
- IK configs per format: `bvh_lafan1_to_g1.json`, `bvh_hc_mocap_to_g1.json`

### IK Offset Calibration

IK config 中每个 (robot_body, human_bone) 对的第 5 个元素是四元数偏移 `R_offset`（w,x,y,z 标量在前）。重定向时的应用公式为：

```
R_result = R_human * R_offset        （scipy Rotation 乘法，即 Hamilton 积）
```

校准公式：

```
R_offset = R_human_tpose^{-1} * R_robot_tpose
```

**关键注意事项 — 根节点朝向对齐**：计算 `R_robot_tpose` 时，必须先将机器人根节点旋转到与 BVH 人体朝向一致。G1 默认朝 +X，hc_mocap BVH 人体朝 -Y（Z-up），需要给根节点设置 -90° Z 轴旋转。若缺少此步骤，所有偏移量会有 ~90° 的系统性偏航误差，导致前倾变侧倾、双脚错位等问题。

机器人 T-pose 设置：
- 根节点四元数：旋转至与人体朝向一致（hc_mocap 为 -90° Z 轴）
- `left_shoulder_roll_joint = +π/2`，`right_shoulder_roll_joint = -π/2`（手臂水平）
- 其余关节为 0

使用 `scripts/compute_ik_offsets.py` 可自动从 BVH 骨架几何推算人体朝向并计算偏移量：

```bash
python scripts/compute_ik_offsets.py                          # 仅打印偏移量
python scripts/compute_ik_offsets.py --write                  # 打印并写入 config
python scripts/compute_ik_offsets.py --bvh path/to/tpose.bvh  # 指定 T-pose 文件
```

**FootMod 朝向来源**：hc_mocap 的 `LeftFootMod`/`RightFootMod` 使用 `LeftToeBase`/`RightToeBase` 的朝向（非 `hc_Foot_L`/`hc_Foot_R`）。若更改此来源，需同步重算踝关节偏移量。

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