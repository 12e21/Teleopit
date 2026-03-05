# 训练指南

本文档介绍如何使用 Teleopit 训练 G1 人形机器人的全身运动追踪策略（mjlab + rsl_rl PPO）。

## 环境搭建

### 前置要求

- Python 3.10
  - **注意**：`cyclonedds==0.10.2`（sim2real 依赖）在 Python 3.10 有预编译 wheel，3.11 需要手动编译 CycloneDDS C 库
- NVIDIA GPU（CUDA 支持）

### 依赖版本说明

训练框架使用 **rsl_rl_lib 5.x**（当前验证版本：`5.0.1`）。rsl_rl 5.x 对观测配置的 API 做了调整：

- `obs_groups` 中的 actor 观测 key 从 `"policy"` 改为 `"actor"`
- `train_mimic/scripts/train.py` 的 `_to_rsl_rl5_cfg()` 已处理此兼容性转换，旧版 mjlab 配置（使用 `"policy"` key）可正常工作

### 安装步骤

训练和推理共用同一个 conda 环境，通过 optional dependencies 按需安装：

```bash
# 创建环境（推荐 Python 3.10，sim2real 兼容性最好）
conda create -n teleopit python=3.10
conda activate teleopit

cd Teleopit

# 仅推理（核心依赖：mujoco, onnxruntime, torch 等）
pip install -e .

# 推理 + 训练（额外安装 rsl_rl, mjlab, wandb 等）
pip install -e '.[train]'

# 推理 + 训练 + sim2real（全部安装）
pip install -e '.[train,sim2real]'
```

验证安装：

```bash
# 验证推理侧
python -c "from teleopit.pipeline import TeleopPipeline; print('inference OK')"

# 验证训练侧
python -c "import train_mimic.tasks; print('training OK')"
```

## 运动数据准备

### PKL → NPZ 转换

训练使用 NPZ 格式的运动数据。mjlab 的 `MotionLoader` 要求**单个 NPZ 文件**，需要把多个片段合并。

```bash
# 转换单个文件
python train_mimic/scripts/convert_pkl_to_npz.py \
    --input data/twist2_retarget_pkl/OMOMO_g1_GMR/sub10_clothesstand_000.pkl \
    --output data/twist2_retarget_npz/OMOMO_g1_GMR/sub10_clothesstand_000.npz

# 批量转换 + 合并为单文件（推荐）
python train_mimic/scripts/convert_pkl_to_npz.py \
    --input data/twist2_retarget_pkl/OMOMO_g1_GMR \
    --output data/twist2_retarget_npz/OMOMO_g1_GMR \
    --merge
# 产出: data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz

# 已有 NPZ 目录，只做合并
python train_mimic/scripts/convert_pkl_to_npz.py \
    --input data/twist2_retarget_npz/OMOMO_g1_GMR \
    --output data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
    --merge
```

> **注意**：`--merge` 沿时间轴拼接所有片段，训练时随机采样起始帧，片段间的不连续不影响训练效果。

### NPZ 数据格式

每个 NPZ 文件包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `fps` | scalar | 帧率 |
| `joint_pos` | (T, 29) | 关节位置 |
| `joint_vel` | (T, 29) | 关节速度（有限差分） |
| `body_pos_w` | (T, nb, 3) | 世界坐标系 body 位置 |
| `body_quat_w` | (T, nb, 4) | 世界坐标系 body 朝向（wxyz） |
| `body_lin_vel_w` | (T, nb, 3) | 线速度 |
| `body_ang_vel_w` | (T, nb, 3) | 角速度 |
| `body_names` | (nb,) | body 名称列表 |

### 可用数据集

运动数据位于 `data/twist2_retarget_pkl/` 下（PKL 格式，需转换）：

| 目录 | 数量 | 说明 |
|------|------|------|
| `OMOMO_g1_GMR` | 5882 | OMOMO 数据集（默认） |
| `AMASS_g1_GMR8` | 13218 | AMASS 数据集 |
| `twist1_to_twist2` | 12788 | TWIST1→TWIST2 转换 |
| `v1_v2_v3_g1` | 73 | 真实动捕数据 |

## 训练流程

### 快速验证

```bash
# 默认使用 tensorboard 日志（不需要 wandb）
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 64 --max_iterations 100 \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz
```

### 完整训练

```bash
# Tensorboard 日志（默认）
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 4096 --max_iterations 30000 \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz

# 启用 wandb 日志（需要 wandb 账号）
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 4096 --max_iterations 30000 \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
    --wandb_project teleopit
```

查看 Tensorboard：
```bash
tensorboard --logdir logs/rsl_rl/g1_tracking
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `Tracking-Flat-G1-v0` | 任务名 |
| `--num_envs` | cfg 默认值 | 并行环境数 |
| `--max_iterations` | cfg 默认值 | 训练迭代数 |
| `--motion_file` | cfg 默认值 | NPZ 运动数据路径（必须是单文件） |
| `--seed` | `42` | 随机种子 |
| `--wandb_project` | 不设置 | 设置后启用 wandb，否则用 tensorboard |
| `--experiment_name` | `g1_tracking` | 实验名（影响日志目录） |
| `--resume` | 不设置 | 从指定 checkpoint 恢复训练 |
| `--device` | `cuda:0` | 训练设备 |

```bash
# 使用 AMASS 数据集
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --motion_file data/twist2_retarget_npz/AMASS_g1_GMR8/merged.npz \
    --num_envs 4096 --max_iterations 30000

# 从 checkpoint 恢复
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --resume logs/rsl_rl/g1_tracking/2026-.../model_10000.pt \
    --max_iterations 30000
```

Checkpoint 保存在 `logs/rsl_rl/g1_tracking/{run_name}/` 目录下。

### 导出 ONNX 模型

训练完成后，将 PyTorch checkpoint 导出为 ONNX 格式：

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_tracking/{run_name}/model_30000.pt \
    --output policy.onnx
```

ONNX 模型参数：
- 输入：`observations`
- 输出：`actions`（29D）
- 内嵌 empirical normalization（训练时的运行均值/方差）

### 策略回放（Viewer）

mjlab 提供两种 viewer：

| Viewer | 命令 | 要求 |
|--------|------|------|
| **native**（默认）| 无额外参数 | 需要本地显示器（X11/Wayland） |
| **viser** | `--viewer viser` | 无需显示器，浏览器访问 `localhost:8012` |

```bash
# Native window（需要显示器）
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_tracking/{run_name}/model_30000.pt \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz

# 浏览器 viewer（SSH 时推荐，加 --viewer viser 后访问 localhost:8012）
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_tracking/{run_name}/model_30000.pt \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
    --viewer viser

# 录制视频（保存到 checkpoint 目录的 videos/play/）
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_tracking/{run_name}/model_30000.pt \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
    --video
```

### 推理部署

使用导出的 ONNX 模型进行遥操推理（MuJoCo 仿真）：

```bash
python scripts/run_sim.py controller.policy_path=policy.onnx robot.obs_builder=mjlab
```

## 训练配置

### 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| sim_dt | 0.005s | 物理仿真步长 |
| decimation | 4 | 每 4 个仿真步执行一次策略 |
| policy_dt | 0.02s (50Hz) | 策略执行频率 |
| num_envs | 4096 | 推荐并行环境数 |

### 网络架构

- 标准 rsl_rl `ActorCritic` MLP
- Actor 网络：[512, 256, 128]，ELU 激活
- 观测维度：~189D（anchor-relative body poses + 关节状态）
- 动作维度：29D（关节位置目标）
- Empirical normalization（替代自定义 Normalizer）

### 观测结构

观测定义来自 mjlab 内置的 `TrackingEnvCfg`（[源码](https://github.com/mujocolab/mjlab)）。

**Policy 观测**（actor 输入，带噪声）：

| 观测项 | 说明 |
|--------|------|
| `generated_commands` | MotionCommand 输出的命令向量 |
| `motion_anchor_pos_b` | 运动目标位置（机器人坐标系，3D） |
| `motion_anchor_ori_b` | 运动目标朝向（6D 旋转表示） |
| `base_lin_vel` | 根节点线速度（3D） |
| `base_ang_vel` | 根节点角速度（3D） |
| `joint_pos_rel` | 关节位置（相对默认值，29D） |
| `joint_vel_rel` | 关节速度（29D） |
| `last_action` | 上一步动作（29D） |

**Critic 观测**（value function 输入，无噪声，额外含）：

| 观测项 | 说明 |
|--------|------|
| `body_pos` | 关键 body 位置（anchor 坐标系） |
| `body_ori` | 关键 body 朝向（anchor 坐标系） |

### 奖励函数

奖励定义来自 mjlab 内置的 `TrackingEnvCfg`，DeepMimic 指数核模式：`R = exp(-error² / std²)`

| 奖励 | 权重 | std | 说明 |
|------|------|-----|------|
| `motion_global_root_pos` | 0.5 | 0.3 | 根位置追踪 |
| `motion_global_root_ori` | 0.5 | 0.4 | 根朝向追踪 |
| `motion_body_pos` | 1.0 | 0.3 | 关键 body 位置追踪 |
| `motion_body_ori` | 1.0 | 0.4 | 关键 body 朝向追踪 |
| `motion_body_lin_vel` | 1.0 | 1.0 | 线速度追踪 |
| `motion_body_ang_vel` | 1.0 | 3.14 | 角速度追踪 |
| `action_rate_l2` | -0.1 | - | 动作平滑正则 |
| `joint_limit` | -10.0 | - | 关节限位惩罚 |
| `self_collisions` | -10.0 | - | 自碰撞惩罚 |

### PPO 超参数

| 参数 | 值 |
|------|-----|
| learning_rate | 1e-3 |
| gamma | 0.99 |
| lam | 0.95 |
| entropy_coef | 0.005 |
| clip_param | 0.2 |
| num_epochs | 5 |
| num_mini_batches | 4 |
| desired_kl | 0.01 |
| num_steps_per_env | 24 |
| empirical_normalization | True |

## 如何判断训练是否有效

### 关键指标

1. **Mean episode length**（最重要）：应该持续上升，表示机器人站得越来越久
2. **Mean reward (total)**：早期可能下降（机器人活得越久，累积 tracking error 越多），后期应回升
3. **Body position tracking error**：应逐渐下降

### 典型训练曲线

```
迭代 0-100:     episode_length 上升，reward 可能下降（正常）
迭代 100-5000:  episode_length 继续上升，reward 开始回升
迭代 5000-30000: 两者趋于收敛
```

## 训练日志解读

下面按 `train.py` 控制台日志字段分组说明“含义、经验正常值、趋势”。
注意：不同数据集/动作难度会导致绝对值不同，优先看趋势。

### 1) 训练吞吐与耗时

| 字段 | 含义 | 正常值（经验） | 期望趋势 |
|------|------|------------------|----------|
| `Learning iteration` | 当前迭代/总迭代 | 单调递增 | 按计划推进到目标迭代 |
| `Total steps` | 累积环境步数 | 单调递增 | 持续增长 |
| `Steps per second` | 训练吞吐（env step/s） | 与机器相关；稳定比绝对值更重要 | 波动小、长期稳定 |
| `Collection time` | 采样耗时 | 通常明显大于学习耗时 | 稳定 |
| `Learning time` | 反向传播耗时 | 通常小于采样耗时 | 稳定 |
| `Iteration time` | 单轮总耗时 | `Collection + Learning` 附近 | 稳定 |
| `ETA` | 剩余训练时间估计 | 参考值 | 随迭代推进逐步下降 |

### 2) PPO 优化信号

| 字段 | 含义 | 正常值（经验） | 期望趋势 |
|------|------|------------------|----------|
| `Mean value loss` | value function 拟合误差 | 常见在 `0.1 ~ 5`，偶发抖动正常 | 中后期下降并趋稳 |
| `Mean surrogate loss` | PPO 策略目标（带符号） | 小幅负值常见（如 `-0.03 ~ 0`） | 绝对值逐步减小、趋稳 |
| `Mean entropy loss` | 熵正则项（日志名含 loss） | 通常为正且随 std 变化 | 前期较高，后期缓降 |
| `Mean action std` | 动作分布标准差（探索强度） | 常见 `0.3 ~ 1.0` | 训练推进时缓慢下降 |

异常信号：
- `value loss` 长时间爆高（如持续 >10）且不回落，常见于学习率过高或奖励尺度异常。
- `action std` 很快塌到极小值（接近 0），常见于过早收敛、探索不足。

### 3) Episode 总体质量

| 字段 | 含义 | 正常值（经验） | 期望趋势 |
|------|------|------------------|----------|
| `Mean reward` | 每回合总回报 | 可能为负（惩罚项存在） | 中后期上升或至少不持续恶化 |
| `Mean episode length` | 平均存活步数 | 初期常很低（1~3） | 持续上升（最关键） |

`Mean reward` 在早期可能下降，但如果 `Mean episode length` 稳步上升，通常仍是正向训练。

### 4) `Episode_Reward/*` 子项

判读规则：
- `motion_*`（追踪奖励）为正，越高越好。
- `action_rate_l2/joint_limit/self_collisions`（惩罚）为负，越接近 0 越好。

建议关注组合趋势：
- `motion_body_pos` 上升 + `Metrics/motion/error_body_pos` 下降，说明姿态跟踪在变好。
- `self_collisions` 长期偏负且不改善，通常要查碰撞体或动作过激。

### 5) `Metrics/motion/*` 误差项

这些是“越小越好”的直接 tracking error（单位随字段而定，位置通常是米，旋转通常是弧度）。

| 字段 | 含义 | 经验目标（中后期） | 期望趋势 |
|------|------|---------------------|----------|
| `error_anchor_pos` | 根位置误差 | 尽量 < `0.25`（终止阈值附近） | 下降 |
| `error_anchor_rot` | 根朝向误差 | 常见目标 < `0.6` rad | 下降 |
| `error_body_pos` | 关键 body 位置误差 | 常见目标 < `0.2` m | 下降 |
| `error_body_rot` | 关键 body 朝向误差 | 常见目标 < `1.0` rad | 下降 |
| `error_joint_pos` | 关节角误差 | 与动作集相关，重点看下降趋势 | 下降 |
| `error_joint_vel` | 关节速度误差 | 与动作激烈程度强相关 | 下降 |
| `error_*_lin_vel` | 线速度误差 | 动作快时会偏大 | 下降或稳定 |
| `error_*_ang_vel` | 角速度误差 | 动作快时会偏大 | 下降或稳定 |

采样相关：
- `sampling_entropy`：采样分布熵，过低可能表示采样过于集中。
- `sampling_top1_prob`：最高概率 bin 的占比，过高可能表示覆盖不足。
- `sampling_top1_bin`：当前最常采样的 bin（用于观察采样偏置）。

### 6) `Episode_Termination/*` 终止原因

判读规则：
- `time_out`：越高越好（说明更多回合是“活到时限”结束）。
- 其他失败原因（`anchor_pos`, `anchor_ori`, `ee_body_pos`）：越低越好。

注意：不同 logger 归一化方式下，这些值不一定严格是 `[0,1]` 概率（可能是每回合平均触发次数或加权统计），所以以相对变化趋势判断更稳妥。

### 7) 对示例日志的快速解读

对于你给的这条 `iteration 967` 示例：
- `Mean episode length = 13.72`：已明显高于起步期，训练在推进。
- `error_anchor_pos = 0.246`：接近 `0.25` 终止边界，根位置稳定性仍是主要瓶颈。
- `Episode_Termination/ee_body_pos = 16.7083`：末端 body 位置误差触发较多，手脚关键点跟踪还不稳。
- `action_rate_l2/joint_limit/self_collisions` 接近 0：正则惩罚总体可控，不是当前主矛盾。

结论：优先继续压低 `anchor_pos` 与 `ee_body_pos` 相关误差，而不是先调正则项。

## 评估

使用 benchmark 脚本评估训练好的策略：

```bash
python train_mimic/scripts/benchmark.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/rsl_rl/g1_tracking/{run_name}/model_30000.pt \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
    --num_envs 1 \
    --num_eval_steps 2000 \
    --warmup_steps 100
```

可选：录制评估视频（用于直观观察动作质量）：

```bash
MUJOCO_GL=egl python train_mimic/scripts/benchmark.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/rsl_rl/g1_tracking/{run_name}/model_30000.pt \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
    --num_envs 1 \
    --video \
    --video_length 600 \
    --video_folder benchmark_results/videos
```

### 跟踪误差

benchmark 脚本会输出：
- `total_error(anchor_pos+anchor_rot+body_pos)`：主对比指标（越小越好）
- `error_anchor_pos`：根位置误差（越小越好）
- `error_anchor_rot`：根朝向误差（越小越好）
- `error_body_pos`：关键 body 位置误差（越小越好）
- `error_anchor_lin_vel / error_body_rot / error_joint_pos / error_joint_vel` 等分布统计（`mean/std/p50/p95/min/max`）
- `mean_step_reward`：每步平均奖励（越高越好）
- `done_rate`, `timeout_rate`, `completed_episodes`, `mean_episode_length`（稳定性指标）

结果保存到：
- `benchmark_results/{task}-{checkpoint}.txt`（人类可读摘要）
- `benchmark_results/{task}-{checkpoint}.json`（完整指标，便于后处理）

## Troubleshooting

常见问题请参考 [训练问题排查文档](training_troubleshooting.md)。

---

## 与旧系统的对比

| 特性 | 旧系统 (Isaac Lab) | 新系统 (mjlab) |
|------|---------------------|----------------|
| 仿真引擎 | PhysX / USD | MuJoCo Warp |
| 环境 API | DirectRLEnv + 自定义 runner | ManagerBasedRlEnvCfg + 标准 rsl_rl |
| 网络架构 | Conv1d motion encoder + MLP | 标准 MLP |
| 观测维度 | 10098D (含 9120D motion history) | ~189D |
| 运动数据 | PKL (自定义 MotionLib) | NPZ (mjlab MotionCommand) |
| Normalization | 自定义 Normalizer | rsl_rl empirical normalization |
| ONNX 导出 | HardwareStudentFutureNN wrapper | 标准 MLP + 内嵌 normalization |
