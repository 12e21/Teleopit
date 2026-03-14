# 推理与运行指南

这篇文档覆盖 Teleopit 当前的推理主路径：**离线 sim2sim、UDP 实时 online sim2sim、viewer、录制和渲染**。

当前运行时主路径是：`mjlab observation + train_mimic 导出的 ONNX policy`。默认使用 154D 观测（`has_state_estimation=false`）；若 ONNX 是 160D，则仅能用于 MuJoCo/sim2sim，并需显式设置 `robot.has_state_estimation=true`。sim2real 只支持 154D。

## 运行前确认

- 已安装核心依赖：`pip install -e .`
- 已准备 `controller.policy_path=...`
- 如果跑离线 BVH，建议显式设置 `input.bvh_file=...`

## 离线 sim2sim

最小命令：

```bash
python scripts/run_sim.py \
  controller.policy_path=policy.onnx \
  input.bvh_file=data/lafan1/dance1_subject2.bvh
```

使用 `hc_mocap`：

```bash
python scripts/run_sim.py \
  controller.policy_path=policy.onnx \
  input.bvh_file=data/hc_mocap/walk.bvh \
  input.bvh_format=hc_mocap
```

常见 override：

```bash
# 关闭窗口
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=none

# 开三个 viewer
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=all

# 只开 retarget + sim2sim
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'

# 运行更多步数并录制
python scripts/run_sim.py controller.policy_path=policy.onnx +num_steps=5000 +record=true
```

## UDP 实时 online sim2sim

启动在线仿真：

```bash
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx
```

发送测试数据：

```bash
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

常见 override：

```bash
# 指定端口
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx input.udp_port=1119

# 无窗口模式
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx viewers=none

# 显式设定有限步数
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx num_steps=300
```

## Viewer 模式

`viewers` 支持以下值：

- `sim2sim`：MuJoCo 物理仿真结果
- `retarget`：重定向后的运动学结果
- `bvh`：原始 BVH 骨架
- `all`：全部打开
- `none`：全部关闭

注意事项：

- 多 viewer 运行在独立子进程中，不共享 GLFW context。
- `'viewers=[retarget,sim2sim]'` 这种 Hydra list override 需要 shell 引号。
- 旧 `viewer=true/false` alias 已移除；统一使用 `viewers=...`。
- 当所有活动 viewer 关闭时，仿真会自动结束。

## 录制

开启录制：

```bash
python scripts/run_sim.py \
  controller.policy_path=policy.onnx \
  input.bvh_file=data/lafan1/dance1_subject2.bvh \
  +record=true \
  recording.output_path=outputs/session.h5
```

录制文件会保存：

- `joint_pos`
- `joint_vel`
- `mimic_obs`
- `action`
- `target_dof_pos`
- `torque`
- `timestamp`

## 渲染

离线渲染单个 BVH：

```bash
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/lafan1/dance1_subject2.bvh --policy policy.onnx
```

`hc_mocap` 示例：

```bash
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/hc_mocap/wander.bvh --format hc_mocap --policy policy.onnx
```

## 当前行为约束

- **只接受 mjlab ONNX（160D 或 154D）**：旧的 1402D / TWIST2 policy 会在运行时直接报错。当前默认是 154D（`has_state_estimation=false`）；若需要跑 160D policy，请只在 MuJoCo/sim2sim 路径显式传入 `robot.has_state_estimation=true`。sim2real 不支持 160D。
- **观测不自动 pad/trim**：观测定义与 ONNX 输入维度不一致时会 fail fast（启动时即校验）。
- **离线输入文件必须显式指定**：不要依赖 `teleopit/configs/input/bvh.yaml`；请总是传 `input.bvh_file=...`。

## 继续阅读

- 配置说明：[`docs/configuration.md`](configuration.md)
- 数据准备：[`docs/dataset.md`](dataset.md)
- 训练与导出：[`docs/training.md`](training.md)
- 真机部署：[`docs/sim2real.md`](sim2real.md)
