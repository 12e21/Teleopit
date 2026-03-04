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
- **Sim2Real 实物部署**: 通过 Unitree SDK2 控制实物 G1 机器人，支持手柄遥控和动捕遥操作双模式，含状态机安全管理和急停功能。
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

使用 `run_sim.py` 启动基于 BVH 文件的 G1 机器人 sim2sim 仿真。通过 Hydra override 配置参数：

```bash
# 运行默认配置 (lafan1 格式 BVH + G1 机器人 + RL 策略)
python scripts/run_sim.py

# 指定 BVH 文件
python scripts/run_sim.py input.bvh_file=data/lafan1/dance1_subject2.bvh

# 使用 hc_mocap 格式（自动推导 human_format=bvh_hc_mocap）
python scripts/run_sim.py input.bvh_file=data/hc_mocap/walk.bvh input.bvh_format=hc_mocap

# 同时开启 BVH 骨架、运动学重定向、物理仿真三个可视化窗口
python scripts/run_sim.py input.bvh_file=data/hc_mocap/walk.bvh input.bvh_format=hc_mocap viewers=all

# 只开 retarget + sim2sim（不显示 BVH 原始骨架）
python scripts/run_sim.py 'viewers=[retarget,sim2sim]'

# 关闭所有可视化窗口
python scripts/run_sim.py viewers=none

# 运行更多步数并录制 HDF5
python scripts/run_sim.py num_steps=5000 record=true
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input.bvh_file` | (见 bvh.yaml) | BVH 文件路径 |
| `input.bvh_format` | `lafan1` | BVH 格式：`lafan1` 或 `hc_mocap` |
| `num_steps` | `1000` | 仿真步数 |
| `viewers` | `"sim2sim"` | 可视化窗口，支持逗号分隔：`bvh`, `retarget`, `sim2sim`；特殊值：`all`（全开）、`none`（无窗口） |
| `record` | `false` | 是否录制 HDF5 数据 |
| `policy_hz` | `50.0` | 策略推理频率 |
| `pd_hz` | `1000.0` | PD 控制频率 |

> **多 Viewer 模式**：`viewers` 支持三种窗口类型——`bvh`（BVH 原始骨架可视化，matplotlib 3D 散点+连线，与 `render_sim.py` 一致）、`retarget`（运动学重定向结果，MuJoCo 设 qpos + 脚底 Z 修正）、`sim2sim`（MuJoCo 物理仿真结果）。所有 viewer 均在独立子进程中运行，关闭所有窗口时仿真自动结束。向后兼容旧的 `+viewer=true/false` 写法。

> **帧率对齐**：RL policy 以 `policy_hz`（默认 50Hz）运行，而 BVH 输入通常为 30fps（hc_mocap 降采样后）。`SimulationLoop` 会自动按时间对齐，多个 policy step 复用同一 BVH 帧，确保动作以原始速度播放。`num_steps` 指 policy step 数，对应仿真时长 = `num_steps / policy_hz` 秒。

### 实时 Online Sim2Sim（UDP 输入）

除了离线 BVH 文件回放，Teleopit 还支持通过 UDP 接收实时动捕数据进行 online sim2sim。每个 UDP 包对应一行 BVH motion data（与 `data/hc_mocap/wander.bvh` 格式一致，159 个 float），接收频率约 30Hz。

**启动 online 仿真：**

```bash
# 终端 1: 启动 online sim（默认监听 UDP 端口 1118，等待数据到达后开始仿真）
python scripts/run_online_sim.py

# 终端 2: 发送测试数据（从 BVH 文件读取并循环发送）
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

**常用参数：**

```bash
# 全部 viewer（BVH 骨架 + 运动学重定向 + 物理仿真）
python scripts/run_online_sim.py viewers=all

# 无窗口模式
python scripts/run_online_sim.py viewers=none

# 自定义 UDP 端口
python scripts/run_online_sim.py input.udp_port=1119

# 指定步数（默认 num_steps=0 表示无限循环，Ctrl+C 退出）
python scripts/run_online_sim.py num_steps=5000
```

**send_bvh_udp.py 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bvh` | (必选) | BVH 文件路径 |
| `--host` | `127.0.0.1` | 目标 IP |
| `--port` | `1118` | 目标 UDP 端口 |
| `--fps` | `0` | 发送帧率（0 = 使用 BVH 原始 Frame Time） |
| `--loop` | `false` | 循环发送 |
| `--downsample` | `1` | 每 N 帧发送一帧 |

> **工作原理**：`UDPBVHInputProvider` 从一个参考 BVH 文件（`reference_bvh`）解析骨骼层级结构（骨骼名、父子关系、偏移量、Euler 旋转顺序），然后在后台线程中通过 `recvfrom` 接收 UDP 数据包，逐帧完成 Euler→四元数→FK→坐标变换，并以线程安全的方式存储最新帧。`get_frame()` 首次调用时会阻塞等待第一帧到达（超时默认 30 秒），后续调用立即返回最新帧。退出方式：Ctrl+C / 关闭 viewer 窗口 / UDP 超时。

### 离线渲染视频

使用 `render_sim.py` 对 BVH 文件进行离线渲染，生成三个对比视频：BVH 原始骨架、GMR 运动学重定向、RL 策略物理仿真（sim2sim）。适用于无显示器的服务器环境（通过 EGL 离屏渲染）。

```bash
# 渲染单个 BVH 文件（输出到 outputs/ 目录）
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/hc_mocap/wander.bvh --format hc_mocap

# 限制渲染时长（秒）
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/hc_mocap/wander.bvh --format hc_mocap --max_seconds 10

# lafan1 格式（默认）
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/lafan1/dance1_subject2.bvh

# 自定义分辨率
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/hc_mocap/wander.bvh --format hc_mocap --width 1280 --height 720

# 批量渲染 data/hc_mocap 下所有动作文件
for f in data/hc_mocap/motion-*.bvh; do
    MUJOCO_GL=egl python scripts/render_sim.py --bvh "$f" --format hc_mocap
done
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bvh` | (必选) | BVH 文件路径 |
| `--format` | `lafan1` | BVH 格式：`lafan1` 或 `hc_mocap` |
| `--max_seconds` | `0` | 最大渲染时长（秒），0 = 渲染完整 BVH |
| `--width` | `640` | 视频宽度（像素） |
| `--height` | `360` | 视频高度（像素） |

输出文件保存在 `outputs/` 目录下，命名规则：`{bvh文件名}_bvh.mp4`、`{bvh文件名}_retarget.mp4`、`{bvh文件名}_sim2sim.mp4`。

### 离线渲染 PKL Retargeting 视频

使用 `render_pkl_sim2sim.py` 对 TWIST2 retargeting 输出的 `.pkl` 文件进行离线渲染，生成两个视频：运动学重定向（直接设 qpos）和 RL 策略物理仿真（sim2sim）。

```bash
# 渲染单个 pkl 文件（输出到 outputs/pkl_sim2sim/{文件名}/ 目录）
MUJOCO_GL=egl python scripts/render_pkl_sim2sim.py --pkl data/twist2_retarget_pkl/OMOMO_g1_GMR/sub1_clothesstand_000.pkl

# 限制渲染时长
MUJOCO_GL=egl python scripts/render_pkl_sim2sim.py --pkl data/twist2_retarget_pkl/OMOMO_g1_GMR/sub1_clothesstand_000.pkl --max_seconds 10

# 自定义输出目录和分辨率
MUJOCO_GL=egl python scripts/render_pkl_sim2sim.py --pkl data/twist2_retarget_pkl/v1_v2_v3_g1/0807_yanjie_walk_001.pkl --output_dir outputs/custom --width 1280 --height 720

# 批量渲染
for f in data/twist2_retarget_pkl/v1_v2_v3_g1/*.pkl; do
    MUJOCO_GL=egl python scripts/render_pkl_sim2sim.py --pkl "$f"
done
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pkl` | (必选) | `.pkl` 运动文件路径 |
| `--output_dir` | `outputs/pkl_sim2sim/{stem}/` | 输出目录 |
| `--max_seconds` | `0` | 最大渲染时长（秒），0 = 渲染完整动作 |
| `--width` | `640` | 视频宽度（像素） |
| `--height` | `360` | 视频高度（像素） |

PKL 文件格式：包含 `root_pos` (N×3)、`root_rot` (N×4, xyzw)、`dof_pos` (N×29)、`fps` 等字段，由 TWIST2 retargeting pipeline 生成。输出文件为 `retarget.mp4` 和 `sim2sim.mp4`。

### Sim2Real 实物部署

Teleopit 支持通过 Unitree SDK2 直接控制实物 G1 机器人，提供手柄遥控和动捕遥操作两种模式。

> **重要**：脚本启动前需要用遥控器手动激活走跑运控（`ai_sport`）。启动时序：机器人开机 → 遥控器进入预备模式 → 遥控器进入走跑运控 → 启动脚本。

**前置准备：**

```bash
# 添加 Unitree SDK2 Python 为 git submodule
git submodule add https://github.com/unitreerobotics/unitree_sdk2_python.git third_party/unitree_sdk2_python
```

**启动控制器：**

```bash
# 连接实物 G1 后启动（默认 eth0）
python scripts/run_sim2real.py

# 指定网络接口
python scripts/run_sim2real.py real_robot.network_interface=enp3s0
```

**操作流程：**
1. 按遥控器 **Start** 键 → 进入手柄模式（左摇杆行走，右摇杆转向）
2. 按 **Y** 键 → 切换到动捕模式（需 UDP 信号校验通过）
3. 按 **X** 键 → 退出动捕模式进入阻尼（用遥控器重新激活走跑运控后按 Start 恢复）
4. **L1 + R1** → 急停（任意状态）

详细文档请参见 [Sim2Real 部署指南](docs/sim2real.md)。

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
- `default.yaml`: 离线仿真顶层配置文件，组合了机器人、控制器和输入源。
- `online.yaml`: 实时 online sim2sim 顶层配置（`realtime: true`, `num_steps: 0`）。
- `sim2real.yaml`: Sim2Real 实物部署配置（手柄/动捕双模式，SDK 参数）。
- `robot/g1.yaml`: 定义机器人参数（XML 路径、PD 增益、动作维度、默认姿态等）。
- `controller/rl_policy.yaml`: 定义控制器参数（ONNX 模型路径、动作缩放等）。
- `input/bvh.yaml`: 定义离线 BVH 输入源参数（BVH 文件路径、重定向格式等）。
- `input/udp_bvh.yaml`: 定义 UDP 实时 BVH 输入源参数（参考 BVH、UDP 端口、超时等）。

关键配置字段示例：
- `num_actions`: 关节动作维度（G1 为 29）。
- `policy_hz`: 策略运行频率（默认 50Hz）。
- `pd_hz`: PD 控制频率（默认 1000Hz）。

## 项目结构

```text
Teleopit/
├── scripts/                 # 入口脚本
│   ├── run_sim.py           # 离线 BVH 仿真
│   ├── run_online_sim.py    # 实时 UDP 在线仿真
│   ├── run_sim2real.py      # Sim2Real 实物部署（手柄/动捕双模式）
│   ├── render_sim.py       # 离线渲染 BVH/重定向/sim2sim 对比视频
│   ├── render_pkl_sim2sim.py # 离线渲染 PKL retargeting/sim2sim 视频
│   └── send_bvh_udp.py     # UDP BVH 测试发送工具
├── teleopit/                # 核心源码
│   ├── bus/                 # 消息总线 (InProcessBus)
│   ├── configs/             # Hydra YAML 配置文件
│   ├── controllers/         # 控制器实现 (RLPolicy, ObservationBuilder)
│   ├── inputs/              # 输入设备支持 (BVHProvider, UDPBVHProvider, VRInputStub)
│   ├── interfaces.py        # 核心抽象接口定义 (Protocol)
│   ├── pipeline.py          # 管线封装 (TeleopPipeline)
│   ├── recording/           # 数据记录模块 (HDF5Recorder)
│   ├── retargeting/         # 运动重定向 (GMR 集成)
│   ├── robots/              # 机器人物理抽象 (MuJoCoRobot)
│   ├── sim/                 # 仿真控制循环 (SimulationLoop)
│   └── sim2real/            # 实物部署 (UnitreeG1Robot, 遥控器, 状态机)
├── third_party/             # 第三方依赖 (git submodule)
│   └── unitree_sdk2_python/ # Unitree SDK2 Python
└── tests/                   # 单元测试与集成测试
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

Teleopit 集成了 TWIST2 的训练代码，作为独立的 `train_mimic` 包，基于 Isaac Lab 进行 GPU 并行仿真训练。

训练需要 GMR retarget 后的运动数据（pkl 格式），默认使用 `data/twist2_retarget_pkl/OMOMO_g1_GMR`。
`--motion_file` 支持目录（加载所有 pkl）、单个 `.pkl` 文件或 `.yaml` 清单文件。

快速开始：

```bash
# 1. 环境搭建（详见 docs/training.md）
conda activate teleopit_isaaclab
# 2. 训练 teacher 策略（默认使用 OMOMO 数据集）
python train_mimic/scripts/train.py \
    --task Isaac-G1-Mimic-v0 \
    --num_envs 4096 \
    --max_iterations 30000 \
    --headless

# 3. 导出 ONNX 模型
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_mimic/{run_name}/model_30000.pt \
    --output policy.onnx
# 4. 推理
python scripts/run_sim.py controller.policy_path=policy.onnx
```

详细文档：
- [Sim2Real 部署指南](docs/sim2real.md) — 实物 G1 控制、手柄/动捕双模式、安全操作
- [训练指南](docs/training.md) — 环境搭建、训练流程、指标解读
- [资产管理](docs/assets.md) — USD/URDF 转换与验证
- [常见问题](docs/troubleshooting.md) — PhysX 挂起、Isaac Sim 警告等
