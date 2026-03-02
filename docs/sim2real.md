# Sim2Real 部署指南

本文档介绍如何使用 Teleopit 通过 Unitree SDK2 控制实物 G1 机器人，支持手柄遥控和动捕遥操作两种模式。

## 概述

Sim2Real 模块复用了 Teleopit 现有的动捕输入、运动重定向、观测构建和 RL 策略推理管线，将底层从 MuJoCo 仿真替换为 Unitree SDK2 DDS 通信，直接控制实物 G1 机器人。

### 两种操作模式

| 模式 | 数据流 | 适用场景 |
|------|--------|----------|
| **手柄模式** | 遥控器摇杆 → 速度指令 → LocoClient → G1 机载步态控制器 | 基础行走测试、场地移动 |
| **动捕模式** | UDP BVH → 重定向 → RL 策略 → SDK 底层关节控制 | 全身运动模仿遥操作 |

## 前置要求

### 硬件

- Unitree G1 机器人（29-DOF 配置）
- Unitree 无线遥控器
- 以太网连接（机器人 ↔ 控制 PC）
- （动捕模式）HC mocap 动捕系统，能输出 BVH 格式 UDP 数据

### 软件

- Python 3.10+
- Teleopit 核心依赖（`pip install -e .`）
- Unitree SDK2 Python（作为 git submodule）

### 安装 SDK

```bash
cd Teleopit

# 添加 SDK 为 git submodule
git submodule add https://github.com/unitreerobotics/unitree_sdk2_python.git third_party/unitree_sdk2_python

# 如果已有 submodule，初始化并更新
git submodule update --init --recursive
```

SDK 不需要单独安装，`run_sim2real.py` 入口脚本会自动将其添加到 `sys.path`。

## 快速开始

### 1. 手柄模式基础测试

```bash
# 启动控制器（默认网络接口 eth0）
python scripts/run_sim2real.py

# 指定网络接口
python scripts/run_sim2real.py real_robot.network_interface=enp3s0
```

启动后按遥控器按键操作（详见下方按键映射）。

### 2. 动捕模式测试

需要两个终端：

```bash
# 终端 1: 启动控制器
python scripts/run_sim2real.py real_robot.network_interface=eth0

# 终端 2: 发送测试动捕数据（从 BVH 文件循环发送）
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop --port 1118
```

在终端 1 中，先用遥控器进入手柄模式（Start → A），然后按 Y 切换到动捕模式。

### 3. 自定义参数

```bash
# 调整速度限制
python scripts/run_sim2real.py gamepad.max_vx=0.3 gamepad.max_vyaw=0.3

# 调整控制频率
python scripts/run_sim2real.py policy_hz=30

# 自定义 UDP 端口
python scripts/run_sim2real.py input.udp_port=1119

# 调整 PD 增益（高级用户）
python scripts/run_sim2real.py 'real_robot.kp_real=[100,100,...]'
```

## 状态机

```text
                    ┌──────────────────────────────────┐
                    │     遥控器 Start                  │
   [DAMPING] ──────►[PREPARATION] ──────►[STANDING]    │
     ▲   ▲          (Start)        遥控器A  (就绪)      │
     │   │                           │                  │
     │   │ 急停(L1+R1)              默认进入            │
     │   │ (任意状态)                │                   │
     │   │                ┌─────────▼──────────┐       │
     │   ├────────────────┤   GAMEPAD (手柄)    │       │
     │   │                │  左摇杆: 前进/横移  │       │
     │   │                │  右摇杆: 转向       │       │
     │   │                └──────┬───────▲──────┘       │
     │   │                遥控器Y│       │遥控器X        │
     │   │          (需UDP校验)  │       │              │
     │   │                ┌──────▼───────┴──────┐       │
     │   └────────────────┤   MOCAP (动捕)      │       │
     │                    │  UDP BVH → 策略推理   │       │
     │                    │  → SDK底层关节控制   │       │
     │                    └─────────────────────┘       │
     │                                                  │
     └──────────────────────────────────────────────────┘
```

### 模式说明

| 模式 | 说明 |
|------|------|
| **DAMPING** | 安全阻尼模式（所有电机 kp=0, kd=8）。初始状态，急停后进入此模式 |
| **PREPARATION** | Start 启动运动控制器。悬吊状态下使用，等待机器人站稳 |
| **STANDING** | 就绪状态，自动进入手柄模式 |
| **GAMEPAD** | 手柄遥控。使用遥控器摇杆控制行走和转向 |
| **MOCAP** | 动捕遥操作。接收 UDP BVH 数据，通过 RL 策略控制关节 |

### 遥控器按键映射

| 按键 | 功能 |
|------|------|
| **Start** | DAMPING → PREPARATION（启动运动控制器） |
| **A** | PREPARATION → STANDING → GAMEPAD |
| **Y** | GAMEPAD → MOCAP（需 UDP 信号校验通过） |
| **X** | MOCAP → GAMEPAD（2s 平滑过渡） |
| **L1 + R1** | 急停：任意模式 → DAMPING |
| 左摇杆 Y | 前进/后退（手柄模式） |
| 左摇杆 X | 左右横移（手柄模式） |
| 右摇杆 X | 左右转向（手柄模式） |

## 配置说明

主配置文件：`teleopit/configs/sim2real.yaml`

### 实物机器人配置 (`real_robot`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `network_interface` | `eth0` | 连接机器人的网络接口 |
| `joint_map` | `[0..28]` | Policy 关节 → SDK 电机索引映射（顺序 1:1） |
| `kp_real` | 见下方 | 位置增益，直接传给 SDK（无需缩放） |
| `kd_real` | 见下方 | 速度增益，直接传给 SDK |
| `kd_damping` | `8.0` | 阻尼模式下的 kd 值 |

### PD 增益（来自 TWIST2 deploy_real）

```
kp: [100,100,100,150,40,40, 100,100,100,150,40,40, 150,150,150,
     40,40,40,40,20,20,20, 40,40,40,40,20,20,20]
kd: [2,2,2,4,2,2, 2,2,2,4,2,2, 4,4,4,
     5,5,5,5,1,1,1, 5,5,5,5,1,1,1]
```

注意：手腕末端 kp=20/kd=1（仿真中 kp=4/kd=0.2），实物需要更大增益。

### 手柄模式配置 (`gamepad`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_vx` | `0.5` | 前进速度上限 (m/s) |
| `max_vy` | `0.3` | 横向速度上限 (m/s) |
| `max_vyaw` | `0.5` | 转向速度上限 (rad/s) |

### 动捕切换安全校验 (`mocap_switch`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `check_frames` | `10` | 切换前需连续有效帧数 |
| `max_position_value` | `5.0` | 骨骼位置合理性阈值 (m) |

## 技术细节

### 与 Sim2Sim 的关键区别

| 方面 | Sim2Sim (MuJoCo) | Sim2Real (SDK) |
|------|-------------------|----------------|
| 机器人状态 | `mujoco.MjData` | SDK `LowState_` |
| 动作执行 | PD 内循环 × decimation | 直接位置指令（SDK 电机 PD） |
| PD 增益 | 仿真值（手腕 kp=4） | 实物值（手腕 kp=20） |
| 控制模式 | MuJoCo torque | SDK FOC mode |

### 关节映射

Policy joint i → SDK motor i，29-DOF 顺序 1:1 对应：

```
左腿 [0-5]:  hip_roll, hip_pitch, hip_yaw, knee, ankle_roll, ankle_pitch
右腿 [6-11]: (同上)
腰部 [12-14]: roll, pitch, yaw
左臂 [15-21]: shoulder_roll, shoulder_pitch, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
右臂 [22-28]: (同上)
```

### CRC32 校验

纯 Python SDK 需手动计算 CRC（TWIST2 pybind 版本自动处理）：

```python
from unitree_sdk2py.utils.crc import CRC
crc = CRC()
cmd.crc = crc.Crc(cmd)  # 每条 lowcmd 发送前调用
```

### motion_switcher

手柄 ↔ 动捕模式切换需要在 LocoClient（高层）和底层控制之间切换：

- **切到底层**（动捕模式前）：停止机载运动服务
- **切回高层**（手柄模式前）：恢复机载运动服务

### 模式切换平滑过渡

**手柄 → 动捕**：
1. `LocoClient.Move(0, 0, 0)` 停止行走，等待 0.5s
2. `motion_switcher` 停止机载运动服务
3. 读当前关节位置作为初始 `_last_retarget_qpos`

**动捕 → 手柄**：
1. 线性插值 2s：当前关节 → 默认站立姿态 (`default_angles`)
2. `motion_switcher` 恢复机载运动服务

## 安全注意事项

1. **首次测试时务必悬吊机器人**，确认各关节运动正常后再落地
2. **始终保持遥控器在手**，随时准备急停（LB + RB）
3. **动捕模式切换前**会自动校验 UDP 信号（连续 10 帧无异常），校验失败不会切换
4. **任何异常**（UDP 断连、策略输出异常等）均会自动进入阻尼模式
5. **Ctrl+C** 退出程序时会自动发送阻尼指令

## 项目结构

```text
teleopit/sim2real/
├── __init__.py          # 包导出
├── controller.py        # 主控制器（状态机 + 控制循环）
├── remote.py            # 遥控器协议解析（40字节，带边沿检测）
└── unitree_g1.py        # SDK 底层接口（DDS、CRC、motion_switcher）

scripts/
└── run_sim2real.py      # Hydra 入口脚本

teleopit/configs/
└── sim2real.yaml        # 顶层配置文件

third_party/
└── unitree_sdk2_python/  # Unitree SDK2 Python（git submodule）
```

## 故障排除

### No LowState received within 3s

机器人未连接或网络接口配置错误。检查：
- 以太网连接是否正常
- `network_interface` 参数是否正确（`ip link show` 查看）
- 机器人是否开机

### LocoClient 超时

机载运动服务未启动。尝试重启机器人或增加 `loco.timeout` 值。

### UDP 校验失败无法切换到动捕

检查：
- 动捕系统是否正在发送数据
- UDP 端口是否匹配（默认 1118）
- 发送的 BVH 数据格式是否与 `reference_bvh` 骨架一致
- 骨骼位置是否在合理范围内（< 5m）

### motion_switcher 切换失败

SDK 版本可能不兼容。查看日志中的具体错误信息，确认 SDK 版本支持 `G1LocoClient` 或 `SportClient`。
