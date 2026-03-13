# Sim2Real 部署指南

本文档介绍如何使用 Teleopit 通过 Unitree SDK2 控制实物 G1 机器人。当前真机控制统一使用 `standing/mocap` 双模式状态机，不再使用旧版 `gamepad/LocoClient` 路径。

> 入口导航：如果你还在熟悉离线/在线推理主路径，先看 [`docs/inference.md`](inference.md)；如果你只是第一次接触项目，先看 [`docs/getting-started.md`](getting-started.md)。

## 概述

Sim2Real 模块复用了 Teleopit 现有的输入、运动重定向、观测构建和 RL 策略推理管线，将执行后端从 MuJoCo 仿真替换为 Unitree SDK2 DDS 通信，直接控制实物 G1。

当前支持两类 mocap 输入：

| 输入源 | 入口脚本 | 说明 |
|------|--------|------|
| **UDP BVH (`hc_mocap`)** | `python scripts/run_sim2real.py ...` | 默认真机输入路径，接收实时 BVH UDP 数据 |
| **Pico4** | `python scripts/run_pico4_sim2real.py ...` | 通过 `xrobotoolkit_sdk` 接收 Pico4 实时全身动捕 |

> **观测维度**：真机只支持 **154D** 观测（`has_state_estimation=false`），因为实物 G1 无法提供 `base_pos` 和 `base_lin_vel`。对应的 ONNX policy 应使用 154D `*-NoStateEst` 任务导出：
>
> ```bash
> python train_mimic/scripts/train.py \
>   --task Tracking-Flat-G1-v0-NoStateEst ...
> ```
>
> 若传入 160D ONNX，或显式设置 `robot.has_state_estimation=true`，启动时会立即报错。

## 控制模式

### 两种运行模式

| 模式 | 数据流 | 适用场景 |
|------|--------|----------|
| **STANDING** | 默认站姿参考 → RL policy → SDK 底层关节控制 | 起步、恢复、等待进入动捕 |
| **MOCAP** | UDP BVH / Pico4 → 重定向 → RL policy → SDK 底层关节控制 | 全身运动遥操作 |

### 状态机

```text
                     ┌──────────────────────────────────────┐
                     │          L1+R1 急停 (任意状态)         │
                     ▼                                      │
  [IDLE] ──Start──▶ [STANDING] ──Y──▶ [MOCAP] ──X──▶ [STANDING]
    ▲                                                   │
    └──────────────────────Start─────────────────────────┘
                             [DAMPING]
```

### 遥控器按键映射

| 按键 | 功能 |
|------|------|
| **Start** | `IDLE/DAMPING -> STANDING` |
| **Y** | `STANDING -> MOCAP`（需输入信号校验通过） |
| **X** | `MOCAP -> STANDING` |
| **L1 + R1** | 急停：任意模式 -> `DAMPING` |

### 模式说明

| 模式 | SDK 层 | 说明 |
|------|--------|------|
| **IDLE** | 未进入 debug | 脚本空闲等待，尚未接管机器人 |
| **STANDING** | debug 模式 | RL policy 保持默认站姿参考，机器人稳定站立 |
| **MOCAP** | debug 模式 | RL policy 跟踪实时运动参考 |
| **DAMPING** | 已退出 debug | 急停/恢复状态，按 Start 可重新进入 STANDING |

## 前置要求

### 硬件

- Unitree G1 机器人（29-DOF 配置）
- Unitree 无线遥控器
- 以太网连接（机器人 ↔ 控制 PC）
- 以下二选一：
  - `hc_mocap` 动捕系统，能输出 BVH 格式 UDP 数据
  - Pico4 头显与配套上游服务

### 软件

- Python 3.10+
- Teleopit 核心依赖
- Unitree SDK2 Python（git submodule）
- 若使用 Pico4：`xrobotoolkit_sdk` 及其上游 PC Service

### 安装 Unitree SDK2

```bash
cd Teleopit

git submodule update --init --recursive
```

`run_sim2real.py` 和 `run_pico4_sim2real.py` 会自动把 `third_party/unitree_sdk2_python` 加到 `sys.path`。

### 安装 Pico4 SDK（仅 Pico4 路径）

```bash
pip install -e '.[pico4]'
bash scripts/setup_pico4.sh
```

说明：
- `.[pico4]` 会连带安装 `sim2real` 依赖
- `xrobotoolkit_sdk` 不会从 PyPI 自动安装，仍需通过厂商提供的流程安装
- `scripts/setup_pico4.sh` 依赖 Ubuntu 22.04 和厂商 PC Service 安装包

## 快速开始

### 0. 启动时序

当前 sim2real 路径不要求先手动激活 `ai_sport`。推荐流程：

1. 机器人开机
2. 确认遥控器可用
3. 在控制 PC 上启动 sim2real 控制脚本
4. 按 Start 进入 `STANDING`
5. 确认机器人稳定站立后，再按 Y 切入 `MOCAP`

### 1. UDP BVH (`hc_mocap`) 真机遥操作

需要两个终端：

```bash
# 终端 1：启动真机控制器
python scripts/run_sim2real.py \
  controller.policy_path=policy.onnx \
  real_robot.network_interface=eth0

# 终端 2：发送测试动捕数据
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop --port 1118
```

进入流程：
1. 在终端 1 启动控制器
2. 按 Start，进入 `STANDING`
3. 确认 UDP 数据已开始发送
4. 按 Y，切换到 `MOCAP`

### 2. Pico4 真机遥操作

```bash
python scripts/run_pico4_sim2real.py \
  controller.policy_path=policy.onnx \
  real_robot.network_interface=eth0
```

进入流程：
1. 确认 Pico4 上游服务已启动，`xrobotoolkit_sdk` 可正常读取 body pose
2. 启动脚本
3. 按 Start，进入 `STANDING`
4. 确认 Pico4 body tracking 可用
5. 按 Y，切换到 `MOCAP`

### 3. 常用参数

```bash
# 调整控制频率
python scripts/run_sim2real.py controller.policy_path=policy.onnx policy_hz=30

# 自定义 UDP 端口
python scripts/run_sim2real.py controller.policy_path=policy.onnx input.udp_port=1119

# 调整 Pico4 首帧等待超时
python scripts/run_pico4_sim2real.py controller.policy_path=policy.onnx input.pico4_timeout=30

# 调整 PD 增益（高级用户）
python scripts/run_sim2real.py controller.policy_path=policy.onnx 'real_robot.kp_real=[100,100,...]'
```

## 配置说明

### 顶层配置

- UDP BVH 真机路径：`teleopit/configs/sim2real.yaml`
- Pico4 真机路径：`teleopit/configs/pico4_sim2real.yaml`

### 输入配置

#### UDP BVH (`teleopit/configs/input/udp_bvh.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `provider` | `udp_bvh` | 输入类型 |
| `reference_bvh` | `data/hc_mocap_bvh/ref_with_toe.bvh` | 参考骨架 |
| `bvh_format` | `hc_mocap` | 单帧 BVH 解析格式 |
| `udp_port` | `1118` | 接收端口 |
| `udp_timeout` | `30.0` | 首帧等待超时 |

#### Pico4 (`teleopit/configs/input/pico4.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `provider` | `pico4` | 输入类型 |
| `human_format` | `xrobot` | 对应 `xrobot_to_g1.json` |
| `pico4_timeout` | `60.0` | 首帧等待超时 |

### 实物机器人配置 (`real_robot`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `network_interface` | `eth0` | 连接机器人的网络接口 |
| `joint_map` | `[0..28]` | Policy 关节 → SDK 电机索引映射 |
| `kp_real` | 见下方 | 位置增益，直接传给 SDK |
| `kd_real` | 见下方 | 速度增益，直接传给 SDK |
| `kd_damping` | `8.0` | 阻尼模式下的 kd 值 |
| `control_mode` | `PR` | 脚踝控制模式 |
| `msg_type` | `HG` | G1 使用的消息协议 |

### 动捕切换安全校验 (`mocap_switch`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `check_frames` | `10` | 切换前需连续有效帧数 |
| `max_position_value` | `5.0` | 骨骼位置合理性阈值 (m) |

## 技术细节

### 与 Sim2Sim 的关键区别

| 方面 | Sim2Sim (MuJoCo) | Sim2Real (SDK) |
|------|-------------------|----------------|
| 观测维度 | 160D 或 154D | 154D |
| 机器人状态 | `mujoco.MjData` | SDK `LowState_` |
| 动作执行 | 仿真内循环 | SDK 电机位置目标 |
| 控制状态机 | 纯软件循环 | `STANDING/MOCAP/DAMPING` |

### 调试模式切换

当前真机控制依赖 `motion_switcher` 进入/退出 debug mode：

- 进入 `STANDING`：`enter_debug_mode()`，然后 `lock_all_joints()`
- 进入 `MOCAP`：保持 debug mode，不再重复切模式
- 进入 `DAMPING`：发送 damping LowCmd，然后 `exit_debug_mode()`
- `Ctrl+C` 正常退出：`shutdown()` 会发送 damping，并尝试 `exit_debug_mode()`

### 关节映射

Policy joint i → SDK motor i，29-DOF 顺序 1:1：

```text
左腿 [0-5]:  hip_roll, hip_pitch, hip_yaw, knee, ankle_roll, ankle_pitch
右腿 [6-11]: (同上)
腰部 [12-14]: roll, pitch, yaw
左臂 [15-21]: shoulder_roll, shoulder_pitch, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
右臂 [22-28]: (同上)
```

## 安全注意事项

1. 首次测试时务必悬吊机器人，确认关节运动正常后再落地。
2. 始终保持遥控器在手，随时准备急停（L1 + R1）。
3. 从 `STANDING` 切到 `MOCAP` 前，脚本会校验输入信号；校验失败不会切换。
4. 任意输入异常（UDP 超时、Pico4 tracking 丢失等）都会触发进入 `DAMPING`。
5. `Ctrl+C` 退出程序时会自动尝试退出 debug mode，但仍建议在现场确认机器人恢复到安全状态。

## 项目结构

```text
teleopit/sim2real/
├── __init__.py          # 包导出
├── controller.py        # 主控制器（standing/mocap 状态机 + 控制循环）
├── remote.py            # 遥控器协议解析（40字节，带边沿检测）
└── unitree_g1.py        # SDK 底层接口（DDS、CRC、motion_switcher）

scripts/
├── run_sim2real.py          # UDP BVH 真机入口
└── run_pico4_sim2real.py    # Pico4 真机入口

teleopit/configs/
├── sim2real.yaml            # UDP BVH 顶层配置
├── pico4_sim2real.yaml      # Pico4 顶层配置
└── input/
    ├── udp_bvh.yaml
    └── pico4.yaml
```

## 故障排除

### No LowState received within 3s

机器人未连接或网络接口配置错误。检查：
- 以太网连接是否正常
- `network_interface` 参数是否正确（`ip link show` 查看）
- 机器人是否开机

### 按 Start 无法进入 STANDING

通常表示 `enter_debug_mode()` 失败。检查：
- `motion_switcher` 是否可用
- SDK 版本是否兼容
- 机器人当前是否处于可切换状态

### 无法切换到 MOCAP

对 UDP BVH：
- 动捕系统是否正在发送数据
- UDP 端口是否匹配（默认 1118）
- 发送 BVH 是否与 `reference_bvh` 骨架一致

对 Pico4：
- `xrobotoolkit_sdk` 是否可正常导入
- Pico4 body tracking 是否已经开始推送
- PC Service 是否正常运行

### 退出后机器人仍不安全

脚本会在 `DAMPING` 或 `shutdown()` 中尝试退出 debug mode，但现场仍应确认：
- 机器人已进入阻尼/安全状态
- 发布线程已停止
- 必要时手动用遥控器恢复或重新上电
