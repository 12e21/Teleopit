# AGENTS.md

## Project Overview

Teleopit is a lightweight, extensible, self-contained humanoid robot whole-body teleoperation framework. It integrates GMR (General Motion Retargeting) and supports train_mimic-exported ONNX RL policy inference.

Language: Python 3.10+
Package: `teleopit` (installed via `pip install -e .`)
Config: Hydra/OmegaConf YAML files in `teleopit/configs/`

## Architecture

```
InputProvider (BVH file / UDP realtime / Pico4 VR) → Retargeter (GMR) → ObservationBuilder (mjlab 160D sim / 154D real) → Controller (ONNX RL) → Robot (MuJoCo + PD / Unitree SDK)
```

Module-internal isolation: all modules run in-process, communicate via `InProcessBus` (zero-copy). Core interfaces defined as `typing.Protocol` in `teleopit/interfaces.py`.

## Directory Structure

```
teleopit/                 # Core package
├── interfaces.py         # Protocol definitions: Robot, Controller, InputProvider, Retargeter, etc.
├── pipeline.py           # TeleopPipeline — thin sim runtime facade
├── runtime/              # Shared runtime assembly: config/path resolution, factories, CLI helpers
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
│   └── observation.py    # MjlabObservationBuilder — 160D (sim) / 154D (real) policy obs aligned with train_mimic
├── inputs/
│   ├── bvh_provider.py       # BVHInputProvider — offline BVH file; exposes fps, bone_names, bone_parents
│   ├── pico4_provider.py     # Pico4InputProvider — xrobotoolkit_sdk realtime body tracking input
│   ├── rot_utils.py          # Quaternion helpers for input-space transforms
│   └── udp_bvh_provider.py   # UDPBVHInputProvider — realtime UDP BVH; daemon receiver thread
├── retargeting/
│   ├── core.py           # RetargetingModule + extract_mimic_obs()
│   └── gmr/              # Self-contained GMR (assets, IK solver, 17+ robot configs)
│       └── assets/unitree_g1/
│           ├── g1_mocap_29dof.xml    # Kinematic retarget only (has ctrlrange bug)
│           ├── g1_sim2sim_29dof.xml  # Legacy sim2sim (old PD gains)
│           └── g1_mjlab.xml          # Default sim2sim — matches mjlab training XML
├── robots/
│   └── mujoco_robot.py   # MuJoCoRobot — MuJoCo sim wrapper
├── sim/
│   └── loop.py           # SimulationLoop — PD control at 1000Hz, policy at 50Hz
└── recording/            # HDF5Recorder
scripts/
├── run_sim.py            # Offline BVH sim2sim pipeline
├── run_sim2real.py       # G1 sim2real control; supports Pico4 via --config-name pico4_sim2real
├── send_bvh_udp.py       # UDP BVH test sender (--bvh, --loop, --fps, --downsample)
├── render_sim.py         # Render single BVH → 3 videos (bvh skeleton, retarget, sim2sim), supports --format flag
├── compute_ik_offsets.py # Compute IK quaternion offsets for new BVH formats (see IK Offset Calibration)
└── setup_pico4.sh        # Pico4 environment setup helper
train_mimic/scripts/data/ # Dataset system scripts: ingest / validate / build / review
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
- Must use `g1_mjlab.xml` for sim2sim (not `g1_mocap_29dof.xml` which clamps torques to ±1 Nm)

### Multi-Viewer Support
`SimulationLoop` supports three simultaneous viewer windows controlled by the `viewers` config:

```bash
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=sim2sim
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[bvh,retarget,sim2sim]'
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=none
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
- `viewers` is the only supported viewer config key; legacy `viewer` alias is removed
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
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx viewers=sim2sim

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

### Pico4 Realtime Input

支持通过 `xrobotoolkit_sdk` 接收 Pico4 实时全身动捕数据，入口脚本：

```bash
python scripts/run_sim.py --config-name pico4_sim controller.policy_path=policy.onnx
python scripts/run_sim2real.py --config-name pico4_sim2real controller.policy_path=policy.onnx
```

**`Pico4InputProvider` 关键设计**：
- 输出骨骼名使用 `xrobot_to_g1.json` 对应的人体命名（如 `Pelvis`、`Left_Hip`、`Right_Shoulder`）
- provider 对 SDK 位姿做一层输入空间变换，以匹配当前 `xrobot` retarget 配置
- 不在代码注释中把该变换固定解释为某个公开坐标系映射；若 SDK 版本、设备固件或上游约定变化，必须用实际 retarget/sim2sim 结果重新验证
- 依赖专有 `xrobotoolkit_sdk`，需用户手动安装；`pyproject.toml` 中不自动拉取该包

**`SimulationLoop` 变更**：
- 新增 `realtime` config flag：即使无 viewer 也进行 wall-clock 限速
- `num_steps=0` 表示无限循环（`max_steps = 2**63`）
- 添加 `KeyboardInterrupt` 处理，Ctrl+C 优雅退出

### 帧率对齐
- BVH 输入帧率（如 hc_mocap 30fps）可能与 policy 频率（50Hz）不同
- `SimulationLoop` 按时间对齐：`bvh_idx = int(policy_time × input_fps)`，多个 policy step 复用同一 BVH 帧
- `bvh_provider.py` 的 `fps` 属性返回降采样后的实际帧率（hc_mocap 60→30fps）

### Inference Observation
- Inference policy observation is **mjlab-aligned**, with two modes controlled by `has_state_estimation`:
  - **160D** (`has_state_estimation=true`): `command(58) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6) + base_lin_vel(3) + base_ang_vel(3) + joint_pos_rel(29) + joint_vel(29) + last_action(29)`.
  - **154D** (current default, `has_state_estimation=false`): `command(58) + motion_anchor_ori_b(6) + base_ang_vel(3) + joint_pos_rel(29) + joint_vel(29) + last_action(29)`. Omits `motion_anchor_pos_b` and `base_lin_vel` (unavailable without state estimation).
- `has_state_estimation` is **not** set in `robot/g1.yaml` (shared config). Current inference defaults are `false` in both `pipeline.py` and `sim2real/controller.py`. Users may override it to `true` for MuJoCo/sim2sim 160D inference, but sim2real must stay `false` and only supports 154D ONNX.
- At startup, obs_builder dimension is validated against the ONNX policy input dimension; mismatches raise `ValueError` immediately.
- Legacy TWIST2 1402D policy path is deprecated and rejected at runtime.

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

### Runtime Validation Policy
- **Fail fast for logical mismatches**: if pipeline components are semantically incompatible (e.g., observation dimension/definition does not match policy expectation), code must raise a clear error immediately.
- **禁止静默修补**：遇到不符合逻辑的数据或配置，必须报错；不要通过自动裁剪、补零、默认替换等方式“强行跑通”。
- Error messages must point to the mismatched components and give a direct fix path (what to align/override).

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
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/lafan1/dance1_subject2.bvh --policy policy.onnx

# Single BVH — hc_mocap format
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/motion_corrected_v2.bvh --format hc_mocap --policy policy.onnx

```

## Known Issues

1. **lafan1-resolved retarget broken**: Different BVH skeleton format (75 joints vs 22). Needs adapter layer to map joints. User deferred fix.
2. **g1_mocap_29dof.xml ctrlrange**: Has `ctrlrange="-1 1"` that clamps all torques. Only use for kinematic retarget visualization, never for sim2sim.

## Training Package (train_mimic/)

Training code migrated from TWIST2, integrated as a separate package with one-way dependency (train → inference).

### Directory Structure

```
train_mimic/               # Training package (pip install -e '.[train]')
├── __init__.py               # Version 0.1.0, TRAIN_MIMIC_ROOT_DIR
├── app.py                    # Shared app helpers for train/play/benchmark
├── data/
│   ├── dataset_builder.py    # YAML spec dataset pipeline
│   ├── dataset_lib.py        # NPZ merge / inspect / hash utilities
│   └── motion_fk.py          # FK consistency checks
├── scripts/                  # Training, export, evaluation, data prep
│   ├── train.py              # mjlab + rsl_rl PPO training entry point
│   ├── save_onnx.py          # ONNX export
│   ├── play.py               # Checkpoint playback
│   ├── benchmark.py          # Policy evaluation with tracking errors
│   └── convert_pkl_to_npz.py # PKL → NPZ clip conversion and merge (body pose labels rebuilt via MuJoCo FK)
├── tasks/                    # Task registration + env/runner cfg
│   └── tracking/config/      # Single official tracking task + internal profiles
├── configs/
│   └── datasets/             # YAML dataset specs
├── assets/g1/                # G1 assets
└── scripts/data/             # Dataset system scripts
    ├── build_dataset.py          # Official one-shot dataset build from YAML spec
    ├── build_twist2_full.sh      # Wrapper around build_dataset.py
    └── check_motion_npz_fk.py    # Validate NPZ body pose labels against MuJoCo FK
```

### Environment Setup & Training

See [docs/training.md](docs/training.md) for full setup and training instructions.
See [docs/dataset.md](docs/dataset.md) for manifest/validate/build data workflow.
See [docs/assets.md](docs/assets.md) for USD/URDF asset management.
See [docs/troubleshooting.md](docs/troubleshooting.md) for known issues.

Quick reference:
 Conda env: `teleopit` (Python 3.10)
 Ingest data: `python train_mimic/scripts/data/ingest_motion.py --input data/hc_mocap_bvh --source hc_mocap_v1 --bvh_format hc_mocap --manifest data/motion/manifests/v1.csv --npz_root .`
 Build dataset (recommended): `python train_mimic/scripts/data/build_dataset.py --spec train_mimic/configs/datasets/twist2_full.yaml`
 Build dataset (wrapper): `bash train_mimic/scripts/data/build_twist2_full.sh`
 Train: `python train_mimic/scripts/train.py --task Tracking-Flat-G1-NoStateEst --motion_file data/datasets/builds/twist2_full/train.npz --num_envs 4096 --max_iterations 30000`
 Multi-GPU Train: `python train_mimic/scripts/train.py --task Tracking-Flat-G1-NoStateEst --motion_file data/datasets/builds/twist2_full/train.npz --gpu_ids 0 1 2 3 --num_envs 1024 --max_iterations 30000` (`--num_envs` is per-GPU)
 Export: `python train_mimic/scripts/save_onnx.py --checkpoint <path> --output policy.onnx`
 Eval: `python train_mimic/scripts/benchmark.py --task Tracking-Flat-G1-NoStateEst --checkpoint <path> --motion_file data/datasets/builds/twist2_full/val.npz --num_envs 1`

### Key Technical Details

 **Environment API**: mjlab `ManagerBasedRlEnv` + standard rsl_rl runner
 **Config system**: Python class-based env/runner cfg from task registry
 **Training task surface**: official task is `Tracking-Flat-G1-NoStateEst`; deprecated alias `Tracking-Flat-G1-v2-NoStateEst` is accepted by CLI and mapped to the official task
 **Multi-GPU training**: supported on a single node via `train_mimic/scripts/train.py --gpu_ids ...`; script relaunches itself with distributed workers, and `--num_envs` means per-GPU environments
 **Checkpoint format**: `logs/rsl_rl/{experiment}/{run}/model_{iter}.pt`
 **Network**: Standard MLP actor/critic (`[512,256,128]`, ELU)
 **Dataset system**: YAML spec -> cached NPZ clips -> `train.npz`/`val.npz`; legacy manifest/review scripts are removed
 **Motion label consistency**: `convert_pkl_to_npz.py` must generate `body_pos_w/body_quat_w/body_ang_vel_w` from MuJoCo FK; use `train_mimic/scripts/data/check_motion_npz_fk.py` to validate clips before large training runs
 **One-way dependency**: train_mimic imports from teleopit, never the reverse
 **Motion file**: single NPZ required by `MotionLoader`; build script outputs `train.npz` / `val.npz`
 **Evaluation**: Use `benchmark.py` to compute 8 tracking errors on trained policies
 **Tracking errors**: joint_dof, joint_vel, root_translation, root_rotation, root_vel, root_ang_vel, keybody_pos, feet_slip
 **Wandb logging**: Episode metrics (rewards + errors) logged via `self.extras["episode"]` in `_reset_idx`
