# 训练指南

本文档介绍如何使用 Teleopit 训练 G1 人形机器人的全身运动模仿策略（TWIST2 架构）。

## 环境搭建

### 前置要求

- Python 3.11+
- NVIDIA GPU（CUDA 支持）
- Isaac Sim 5.1.0 + Isaac Lab v2.3.2

### 安装步骤

1. 创建 conda 环境：

```bash
conda create -n teleopit_isaaclab python=3.11
conda activate teleopit_isaaclab
```

2. 安装 Isaac Sim 5.1.0：

```bash
pip install isaacsim==5.1.0
```

3. 从源码安装 Isaac Lab v2.3.2：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
pip install -e .
```

4. 安装 Teleopit 及训练依赖：

```bash
cd Teleopit
pip install -e '.[train]'
```

5. 验证安装：

```bash
python -c "import isaacsim; import isaaclab; import torch; print(f'Isaac Lab OK, CUDA: {torch.cuda.is_available()}')"
```

## 训练流程

### Phase 1: Teacher 策略训练（PPO）

使用 Isaac Lab 的 GPU 并行仿真环境训练 teacher 策略：

```bash
cd Teleopit
python teleopit_train/scripts/train.py \
    --task Isaac-G1-Mimic-v0 \
    --num_envs 4096 \
    --max_iterations 30000 \
    --headless \
    --wandb_project teleopit_isaaclab
```

参数说明：
- `--task Isaac-G1-Mimic-v0`：G1 运动模仿任务
- `--num_envs 4096`：并行环境数量（GPU 显存允许的情况下越多越好）
- `--max_iterations 30000`：训练迭代次数
- `--headless`：无头模式（无 GUI）
- `--wandb_project`：可选，启用 wandb 日志记录
- `--seed 42`：可选，固定随机种子

Checkpoint 保存在 `logs/rsl_rl/g1_mimic/<timestamp>/` 目录下。

### Phase 2: 导出 ONNX 模型

训练完成后，将 PyTorch checkpoint 导出为 ONNX 格式：

```bash
python teleopit_train/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_mimic/<timestamp>/model_30000.pt \
    --output policy.onnx
```

### Phase 3: 推理部署

使用导出的 ONNX 模型进行遥操推理（MuJoCo 仿真）：

```bash
python scripts/run_sim.py --policy policy.onnx
```

## 如何判断训练是否有效

### 关键指标

1. **Mean episode length**（最重要）：应该持续上升，表示机器人站得越来越久
2. **Mean reward (total)**：在 mimic 任务中，早期可能下降（机器人活得越久，累积 tracking error 越多），后期应回升
3. **Surrogate loss**：PPO 的策略梯度 loss，应为负值且绝对值逐渐增大
4. **Mean action noise std**：探索噪声，训练过程中应逐渐减小

### 典型训练曲线

```
迭代 0-100:     episode_length 上升，reward 可能下降（正常）
迭代 100-5000:  episode_length 继续上升，reward 开始回升
迭代 5000-30000: 两者趋于收敛
```

### 使用 wandb 监控

如果启用了 `--wandb_project`，可以在 wandb 面板中查看：
- 各 reward 分量的变化趋势
- episode length 分布
- 策略网络的梯度和权重统计

## 训练配置

### 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| sim_dt | 0.002s | 物理仿真步长 |
| decimation | 10 | 每 10 个仿真步执行一次策略 |
| policy_dt | 0.02s (50Hz) | 策略执行频率 |
| num_envs | 4096 | 推荐并行环境数 |

### 网络架构

- ActorCriticFuture：Conv1D 历史编码器 + MLP 未来编码器
- Actor 网络：[512, 512, 256, 128]
- 观测维度：1402D（127×11 历史 + 35 mimic 特征）

### PhysX 配置

当前已优化的 PhysX 参数（在 `teleopit_train/envs/g1_mimic_env.py` 中）：

| 参数 | 值 | 说明 |
|------|-----|------|
| max_depenetration_velocity | 10.0 | 防止穿透时的弹射速度上限 |
| solver_position_iteration_count | 8 | 位置求解器迭代次数 |
| solver_velocity_iteration_count | 4 | 速度求解器迭代次数 |
| ImplicitActuator stiffness | 0.0 | 关闭 Isaac Lab 内置 PD（使用自定义 PD） |
| ImplicitActuator damping | 0.0 | 同上 |

## USD 资产管理

G1 机器人的 USD 资产位于 `teleopit_train/assets/g1/usd/g1_29dof.usd`。

如需重新生成 USD（例如修改了 URDF），使用官方 UrdfConverter：

```bash
conda activate teleopit_isaaclab
OMNI_KIT_ACCEPT_EULA=YES python teleopit_train/scripts/convert_urdf_isaaclab.py --headless
```

详细的资产管理说明参见 [assets.md](assets.md)。

## Phase 2/3: Student 蒸馏（DAgger）

Student 策略训练（DAgger 蒸馏 + 未来运动编码器）计划在后续迭代中实现。
