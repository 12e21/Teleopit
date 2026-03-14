# Teleopit

Teleopit 是一个轻量、可扩展、自包含的人形机器人全身遥操作框架。

它把 `BVH / UDP 动捕输入 → GMR 运动重定向 → mjlab 观测构建 → ONNX RL 策略推理 → MuJoCo / Unitree G1 执行` 串成一条可复用的工程链路。

## What It Is

- **推理主路径明确**：当前在线/离线推理统一走 mjlab 观测（sim2sim 160D / 真机 154D）与 `train_mimic` 导出的 ONNX 策略。
- **运行方式完整**：支持离线 sim2sim、UDP 实时 online sim2sim、Pico4 实时输入、G1 sim2real，以及训练数据准备与策略训练。
- **模块边界清晰**：核心组件通过 `Protocol` 和 `InProcessBus` 解耦，便于替换输入源、机器人后端和控制器。

## Architecture

```text
InputProvider (BVH file / UDP realtime / Pico4)
    -> Retargeter (GMR)
    -> ObservationBuilder (mjlab 160D sim / 154D real)
    -> Controller (ONNX RL policy)
    -> Robot (MuJoCo sim or Unitree G1)
```

核心装配入口已经收敛到 `teleopit/runtime/`；`teleopit/pipeline.py` 是 sim runtime façade，`teleopit/sim2real/controller.py` 保留真机状态机，离线主循环在 `teleopit/sim/loop.py`。

## Install

项目要求 Python `3.10+`。

```bash
# 核心推理依赖（离线 sim / online sim）
pip install -e .

# 推理 + 训练
pip install -e '.[train]'

# 推理 + sim2real
pip install -e '.[sim2real]'

# 全部能力
pip install -e '.[train,sim2real]'
```

如果要运行测试：

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

## Quick Start

### 离线 sim2sim

```bash
python scripts/run_sim.py \
  controller.policy_path=policy.onnx \
  input.bvh_file=data/lafan1/dance1_subject2.bvh
```

### UDP 实时 online sim2sim

```bash
# Terminal 1
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx

# Terminal 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

### Pico4 实时 sim2sim

```bash
python scripts/run_sim.py --config-name pico4_sim controller.policy_path=policy.onnx
```

### Pico4 实时 sim2real

```bash
python scripts/run_sim2real.py --config-name pico4_sim2real controller.policy_path=policy.onnx
```

### G1 sim2real

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx
```

### 训练入口

```bash
python train_mimic/scripts/train.py \
  --task Tracking-Flat-G1-NoStateEst \
  --motion_file data/datasets/twist2_full/train.npz
```

## Choose Your Path

- **我想一键重建推荐训练数据集**：运行 `python train_mimic/scripts/data/build_dataset.py --spec train_mimic/configs/datasets/twist2_full.yaml`
- **我想先把 BVH / PKL / 现有 NPZ 批量转成标准 clips**：运行 `python train_mimic/scripts/data/ingest_motion.py --type <bvh|pkl|npz> --input <path> --output <dataset/clips/source>`
- **我想看新数据系统说明**：看 [`docs/dataset.md`](docs/dataset.md)

- **我想先把一个 BVH 跑起来**：看 [`docs/getting-started.md`](docs/getting-started.md)
- **我想了解离线 / 在线推理、viewer、录制**：看 [`docs/inference.md`](docs/inference.md)
- **我想接 Pico4 全身动捕**：看 README 里的 Pico4 运行入口，并确认本机已手动安装 `xrobotoolkit_sdk`
- **我想理解 Hydra 配置该怎么改**：看 [`docs/configuration.md`](docs/configuration.md)
- **我想构建训练数据集**：看 [`docs/dataset.md`](docs/dataset.md)
- **我想训练或导出 ONNX policy**：看 [`docs/training.md`](docs/training.md)
- **我想检查 motion NPZ 标签是否和 FK 一致**：运行 `python train_mimic/scripts/data/check_motion_npz_fk.py --npz <clip.npz>`
- **我想控制真机 G1**：看 [`docs/sim2real.md`](docs/sim2real.md)
- **我遇到训练侧问题**：看 [`docs/training_troubleshooting.md`](docs/training_troubleshooting.md)

## Docs

- [`docs/getting-started.md`](docs/getting-started.md)：按任务导航的最短上手路径
- [`docs/architecture.md`](docs/architecture.md)：系统边界、层次与运行时装配
- [`docs/inference.md`](docs/inference.md)：离线 sim、online sim、录制、viewer、渲染
- [`docs/configuration.md`](docs/configuration.md)：配置组合方式、关键字段、常见 override
- [`docs/dataset.md`](docs/dataset.md)：YAML spec 数据集构建主线
- [`docs/training.md`](docs/training.md)：官方训练、评估、导出 ONNX 主线
- [`docs/sim2real.md`](docs/sim2real.md)：Unitree G1 实机部署与状态机
- [`docs/training_troubleshooting.md`](docs/training_troubleshooting.md)：训练侧常见问题排查

## Current Constraints

- **只支持 mjlab policy（160D 或 154D）**：运行时会拒绝旧 TWIST2 1402D ONNX。当前推理路径默认走 154D（`has_state_estimation=false`）；160D ONNX 只适用于 MuJoCo/sim2sim，并且需要显式传入 `robot.has_state_estimation=true`。sim2real 只支持 154D。ONNX 维度与该配置不匹配时启动即报错。
- **必须提供有效 ONNX 路径**：`controller.policy_path` 不能为空，且应来自 `train_mimic` 导出。
- **离线 BVH 必须显式提供输入文件**：`teleopit/configs/input/bvh.yaml` 不再内置机器相关默认路径；请总是传 `input.bvh_file=...`。
- **viewer 只接受 `viewers`**：旧 `viewer=true/false` 兼容键已移除。
- **Pico4 输入依赖厂商 SDK**：`Pico4InputProvider` 依赖 `xrobotoolkit_sdk`，该依赖不会随 `pip install -e .` 自动安装。
- **Pico4 坐标约定以运行结果为准**：`teleopit/inputs/pico4_provider.py` 会对 SDK 返回的位姿做一层输入空间变换，以匹配现有 `xrobot` retarget 配置。这里不再把该变换硬编码描述为某个公开坐标系转换；若后续替换 SDK 或更新设备固件，应以实际 sim2sim/retarget 标定结果重新验证。
- **sim2sim 应使用正确 XML**：G1 仿真默认使用 `g1_mjlab.xml`（从 unitree_rl_mjlab 训练 XML 复制，含 7-capsule 碰撞足和 affine PD actuator）。

## Repo Map

```text
teleopit/           推理与控制核心包
scripts/            运行、渲染、调试入口脚本
train_mimic/        训练、评估、ONNX 导出、数据集构建
docs/               面向任务的说明文档
tests/              单元测试与集成测试
```

如果你是第一次接触这个项目，建议先读 [`docs/getting-started.md`](docs/getting-started.md)，再按目标场景进入对应专项文档。
