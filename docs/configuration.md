# 配置说明

Teleopit 使用 Hydra 组合配置。大多数运行入口都会从一个顶层 YAML 开始，再通过命令行 override 修改局部字段。

## 顶层配置

- `teleopit/configs/default.yaml`：离线 sim2sim
- `teleopit/configs/online.yaml`：UDP 实时 online sim2sim
- `teleopit/configs/sim2real.yaml`：Unitree G1 真机控制

这三个文件都会组合若干子配置：

- `teleopit/configs/robot/g1.yaml`
- `teleopit/configs/controller/rl_policy.yaml`
- `teleopit/configs/input/bvh.yaml`
- `teleopit/configs/input/udp_bvh.yaml`

## 最常改的字段

### 顶层字段

- `policy_hz`：策略推理频率
- `pd_hz`：PD 控制频率
- `viewers`：viewer 集合，支持 `bvh`、`retarget`、`sim2sim`、`all`、`none`
- `realtime`：是否按 wall clock 限速
- `num_steps`：运行步数；online 模式常用 `0` 表示无限循环

### `robot`

- `num_actions`：关节动作维度，G1 当前为 `29`
- `xml_path`：MuJoCo XML 路径
- `kps` / `kds`：PD 增益
- `default_angles`：默认站姿
- `torque_limits`：关节扭矩限幅
- `obs_builder`：当前应为 `mjlab`

### `controller`

- `policy_path`：ONNX policy 路径，必填
- `device`：`cpu` / `auto` / `cuda...`
- `action_scale`：动作缩放
- `clip_range`：动作裁剪范围
- `default_dof_pos`：目标关节角偏移基准

### `input`

离线 BVH：

- `bvh_file`
- `bvh_format`
- `human_format`

UDP 实时输入：

- `provider=udp_bvh`
- `reference_bvh`
- `udp_host`
- `udp_port`
- `udp_timeout`
- `bvh_format`

## 最关键的一条：`default_dof_pos`

RL policy 输出的 action 不是绝对关节角，而是相对默认站姿的偏移。目标关节角计算逻辑是：

```text
target_dof_pos = clip(action, low, high) * action_scale + default_dof_pos
```

因此：

- `default_dof_pos` 必须与机器人 `default_angles` 对齐
- 缺失这条偏移时，机器人很容易因为目标姿态错误而站不住

`TeleopPipeline` 会在初始化时自动把 `robot.default_angles` 传给控制器的 `default_dof_pos`。但理解这条链路仍然很重要，尤其是在你写自定义入口或单测时。

## 推荐 override 示例

### 离线 sim2sim

```bash
python scripts/run_sim.py \
  controller.policy_path=policy.onnx \
  input.bvh_file=data/lafan1/dance1_subject2.bvh \
  policy_hz=50 pd_hz=1000
```

### 改 viewer

```bash
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=none
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
```

### 改 UDP 端口

```bash
python scripts/run_online_sim.py controller.policy_path=policy.onnx input.udp_port=1119
```

### 改真机网络接口

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx real_robot.network_interface=enp3s0
```

## 设计原则：不做静默修补

Teleopit 当前配置与运行时逻辑遵循 fail-fast 原则：

- policy 维度不对，直接报错
- 观测定义不匹配，直接报错
- 必需路径缺失，直接报错
- 不会自动 pad/trim 观测，也不会用默认值“强行跑通”

这意味着你在修改配置时，应优先追查“哪两个组件定义不一致”。

## 常见问题

### 为什么我明明设置了 `policy_path` 还是起不来？

- 先确认文件存在
- 再确认它不是旧的 1402D / TWIST2 ONNX
- 再确认输入维度是 `160`

### 为什么不建议依赖 `input/bvh.yaml` 的默认 `bvh_file`？

因为它现在是示例性质的机器本地路径。最稳妥的做法始终是命令行显式指定：

```bash
python scripts/run_sim.py controller.policy_path=policy.onnx input.bvh_file=...
```

## 继续阅读

- 上手导航：[`docs/getting-started.md`](getting-started.md)
- 推理与运行：[`docs/inference.md`](inference.md)
- 数据集流程：[`docs/dataset.md`](dataset.md)
- 训练流程：[`docs/training.md`](training.md)
- 真机部署：[`docs/sim2real.md`](sim2real.md)
