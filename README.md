# Teleopit

Teleopit 是一个轻量、可扩展、自包含的人形机器人全身遥操作框架。

它把 `BVH / UDP 动捕输入 → GMR 运动重定向 → mjlab 160D 观测构建 → ONNX RL 策略推理 → MuJoCo / Unitree G1 执行` 串成一条可复用的工程链路。

## What It Is

- **推理主路径明确**：当前在线/离线推理统一走 `mjlab 160D` 观测与 `train_mimic` 导出的 ONNX 策略。
- **运行方式完整**：支持离线 sim2sim、UDP 实时 online sim2sim、G1 sim2real，以及训练数据准备与策略训练。
- **模块边界清晰**：核心组件通过 `Protocol` 和 `InProcessBus` 解耦，便于替换输入源、机器人后端和控制器。

## Architecture

```text
InputProvider (BVH file / UDP realtime)
    -> Retargeter (GMR)
    -> ObservationBuilder (mjlab 160D)
    -> Controller (ONNX RL policy)
    -> Robot (MuJoCo sim or Unitree G1)
```

核心装配入口在 `teleopit/pipeline.py`，离线主循环在 `teleopit/sim/loop.py`，实物控制状态机在 `teleopit/sim2real/controller.py`。

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
python scripts/run_online_sim.py controller.policy_path=policy.onnx

# Terminal 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

### G1 sim2real

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx
```

### 训练入口

```bash
python train_mimic/scripts/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file data/datasets/builds/twist2_full/train.npz
```

## Choose Your Path

- **我想一键重建推荐训练数据集**：运行 `bash scripts/data/build_twist2_full.sh`
- **我想看新数据系统说明**：看 [`docs/dataset.md`](docs/dataset.md)

- **我想先把一个 BVH 跑起来**：看 [`docs/getting-started.md`](docs/getting-started.md)
- **我想了解离线 / 在线推理、viewer、录制**：看 [`docs/inference.md`](docs/inference.md)
- **我想理解 Hydra 配置该怎么改**：看 [`docs/configuration.md`](docs/configuration.md)
- **我想做数据清洗、manifest、build**：看 [`docs/dataset.md`](docs/dataset.md)
- **我想训练或导出 ONNX policy**：看 [`docs/training.md`](docs/training.md)
- **我想检查 motion NPZ 标签是否和 FK 一致**：运行 `python scripts/data/check_motion_npz_fk.py --npz <clip.npz>`
- **我想控制真机 G1**：看 [`docs/sim2real.md`](docs/sim2real.md)
- **我遇到训练侧问题**：看 [`docs/training_troubleshooting.md`](docs/training_troubleshooting.md)

## Docs

- [`docs/getting-started.md`](docs/getting-started.md)：按任务导航的最短上手路径
- [`docs/inference.md`](docs/inference.md)：离线 sim、online sim、录制、viewer、渲染
- [`docs/configuration.md`](docs/configuration.md)：配置组合方式、关键字段、常见 override
- [`docs/dataset.md`](docs/dataset.md)：manifest / validate / build 数据流程
- [`docs/training.md`](docs/training.md)：训练、评估、导出 ONNX
- [`docs/sim2real.md`](docs/sim2real.md)：Unitree G1 实机部署与状态机
- [`docs/training_troubleshooting.md`](docs/training_troubleshooting.md)：训练侧常见问题排查

## Current Constraints

- **只支持 mjlab 160D policy**：运行时会拒绝旧 TWIST2 1402D ONNX。
- **必须提供有效 ONNX 路径**：`controller.policy_path` 不能为空，且应来自 `train_mimic` 导出。
- **建议显式指定 `input.bvh_file`**：当前 `teleopit/configs/input/bvh.yaml` 中的默认路径是机器相关示例，不应依赖。
- **sim2sim 应使用正确 XML**：G1 仿真应使用 `g1_sim2sim_29dof.xml`，避免旧 mocap XML 的 actuator limit 问题。

## Repo Map

```text
teleopit/           推理与控制核心包
scripts/            运行、渲染、数据处理入口脚本
train_mimic/        训练、评估、ONNX 导出
docs/               面向任务的说明文档
tests/              单元测试与集成测试
```

如果你是第一次接触这个项目，建议先读 [`docs/getting-started.md`](docs/getting-started.md)，再按目标场景进入对应专项文档。
