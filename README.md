# Teleopit

Teleopit 是一个轻量、可扩展、自包含的人形机器人全身遥操框架。它借鉴了 TWIST2 的分层架构设计，集成了 GMR (General Motion Retargeting) 运动重定向算法，通过抽象接口实现了对多种机器人、输入设备和控制器的灵活支持。

## 项目概述

Teleopit 旨在为人形机器人遥操提供一个标准化的流程。其设计灵感源自 TWIST2 项目，但在可扩展性和自包含性上做了深度优化：
- **集成 GMR**: 内置了成熟的运动重定向模块，支持 17 种以上的机器人型号。
- **抽象接口**: 基于 Python `Protocol` 定义了机器人、输入设备、控制器等核心接口，方便开发者扩展。
- **低耦合**: 模块间通过轻量级的消息总线通信，既保证了同进程内的高性能（零拷贝），也预留了跨进程拆分的灵活性。

## 架构

Teleopit 采用模块化架构，核心数据流如下所示：

```text
[ InputProvider ] (例如: BVH 文件, VR 设备)
       |
       v (人体关节位置与姿态)
 [ Retargeter ] (集成 GMR: 人体姿态 -> 机器人关节位置)
       |
       v (机器人目标 qpos)
[ ObservationBuilder ] (复刻 TWIST2 1402D 观测构建)
       |
       v (平坦化的观测向量)
 [ Controller ] (ONNX 格式 of RL Policy)
       |
       v (目标关节位置/动作)
   [ Robot ] (MuJoCo 仿真 + PD 控制循环)
```

项目采用了模块内隔离的设计，默认在同一进程内运行，通过 `InProcessBus` 实现消息的发布与订阅。

## 特性

- **GMR 运动重定向**: 独立集成的 GMR 模块，包含完整资产，支持 unitree_g1, h1 等多种机器人。支持 lafan1 和 hc_mocap 等多种 BVH 格式。
- **TWIST2 兼容观测**: 精确复刻 1402D (127×11 + 35) 观测构建逻辑，直接支持 TWIST2 预训练模型。
- **ONNX RL 策略推理**: 采用 `onnxruntime` 进行推理，不依赖完整的 PyTorch 环境即可运行。
- **MuJoCo 仿真与 PD 控制**: 集成 MuJoCo 物理引擎，支持 1000Hz PD 控制与 50Hz Policy 推理频率。
- **HDF5 数据记录**: 支持高效的数据录制，采用分块存储与 gzip 压缩。
- **Hydra 配置系统**: 全面的配置管理，通过 YAML 文件轻松切换机器人和控制器参数。
- **零拷贝消息总线**: 进程内通信采用对象引用传递，确保高性能数据传输。
- **可扩展接口**: 核心组件均遵循 `Protocol` 接口协议，实现新功能仅需实现对应接口。

## 安装

Teleopit 要求 Python 3.10 或更高版本。

```bash
# 进入项目根目录
cd Teleopit

# 以开发模式安装
pip install -e .
```

或者直接将项目路径添加到 `PYTHONPATH` 中。

## 快速开始

### 运行仿真

可以使用内置的脚本快速启动基于 BVH 文件的 G1 机器人仿真：

```bash
# 运行默认配置 (G1 机器人 + BVH 输入 + RL 策略)
python scripts/run_sim.py
```

### 编程方式使用

通过 `TeleopPipeline` 可以快速组装并运行整个管线：

```python
from teleopit.pipeline import TeleopPipeline
from omegaconf import OmegaConf

# 加载配置
cfg = OmegaConf.load("teleopit/configs/default.yaml")

# 初始化管线
pipeline = TeleopPipeline(cfg)

# 运行 1000 步并记录数据
result = pipeline.run(num_steps=1000, record=True)
print(f"Session recorded to: {result['record_path']}")
```

## 配置说明

Teleopit 使用 Hydra 管理配置，结构如下：
- `default.yaml`: 顶层配置文件，组合了机器人、控制器和输入源。
- `robot/g1.yaml`: 定义机器人参数（XML 路径、PD 增益、动作维度、默认姿态等）。
- `controller/rl_policy.yaml`: 定义控制器参数（ONNX 模型路径、动作缩放等）。
- `input/bvh.yaml`: 定义输入源参数（BVH 文件路径、重定向格式等）。

关键配置字段示例：
- `num_actions`: 关节动作维度（G1 为 29）。
- `policy_hz`: 策略运行频率（默认 50Hz）。
- `pd_hz`: PD 控制频率（默认 1000Hz）。

## 项目结构

```text
Teleopit/
├── scripts/             # 入口脚本
│   └── run_sim.py       # 仿真运行示例
├── teleopit/            # 核心源码
│   ├── bus/             # 消息总线 (InProcessBus)
│   ├── configs/         # Hydra YAML 配置文件
│   ├── controllers/     # 控制器实现 (RLPolicy, ObservationBuilder)
│   ├── inputs/          # 输入设备支持 (BVHProvider, VRInputStub)
│   ├── interfaces.py    # 核心抽象接口定义 (Protocol)
│   ├── pipeline.py      # 管线封装 (TeleopPipeline)
│   ├── recording/       # 数据记录模块 (HDF5Recorder)
│   ├── retargeting/     # 运动重定向 (GMR 集成)
│   ├── robots/          # 机器人物理抽象 (MuJoCoRobot)
│   └── sim/             # 仿真控制循环 (SimulationLoop)
└── tests/               # 单元测试与集成测试
```

## 扩展指南

要添加新功能，只需实现 `teleopit/interfaces.py` 中定义的相应 `Protocol`：

1. **新机器人**: 实现 `Robot` 接口，并在 `teleopit/configs/robot/` 下创建新的 YAML 配置。
2. **新输入设备**: 实现 `InputProvider` 接口，返回包含人体姿态数据的字典。
3. **新控制器**: 实现 `Controller` 接口，接收观测并返回关节动作。

## 测试

使用 `pytest` 运行所有测试套件：

```bash
pytest tests/ -v
```

项目包含 60+ 个单元测试，涵盖了从数据总线到运动重定向的核心功能，并包含一个端到端的仿真测试。

## 依赖项

核心依赖项包括：
- `mujoco`: 物理仿真
- `onnxruntime`: 策略推理
- `hydra-core`: 配置管理
- `mink`, `qpsolvers`: 逆运动学求解 (GMR 依赖)
- `h5py`: 数据记录
- `numpy`, `scipy`, `torch`: 数值计算与张量处理
- `rich`: 终端日志增强


## 训练

Teleopit 集成了 TWIST2 的训练代码，作为独立的 `teleopit_train` 包，基于 Isaac Lab 进行 GPU 并行仿真训练。

快速开始：

```bash
# 1. 环境搭建（详见 docs/training.md）
conda activate teleopit_isaaclab
# 2. 训练 teacher 策略
python teleopit_train/scripts/train.py \
    --task Isaac-G1-Mimic-v0 \
    --num_envs 4096 \
    --max_iterations 30000 \
    --headless

# 3. 导出 ONNX 模型
python teleopit_train/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_mimic/<timestamp>/model_30000.pt \
    --output policy.onnx
# 4. 推理
python scripts/run_sim.py --policy policy.onnx
```

详细文档：
- [训练指南](docs/training.md) — 环境搭建、训练流程、指标解读
- [资产管理](docs/assets.md) — USD/URDF 转换与验证
- [常见问题](docs/troubleshooting.md) — PhysX 挂起、Isaac Sim 警告等