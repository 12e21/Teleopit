# 快速上手

这篇文档只回答一个问题：**你现在想做什么？**

如果你需要完整背景，请先看仓库首页 [`README.md`](../README.md)；如果你已经知道目标场景，直接按下面的路径走。

## 先决条件

- Python `3.10+`
- 已在仓库根目录执行 `pip install -e .`
- 已准备好一个由 `train_mimic` 导出的 ONNX policy

如果你要训练或上真机，请分别安装额外依赖：

```bash
pip install -e '.[train]'
pip install -e '.[sim2real]'
```

## 选择你的目标

### 1. 我想离线跑一个 BVH

这是最推荐的新手入口。

```bash
python scripts/run_sim.py \
  controller.policy_path=policy.onnx \
  input.bvh_file=data/lafan1/dance1_subject2.bvh
```

如果你使用 `hc_mocap`：

```bash
python scripts/run_sim.py \
  controller.policy_path=policy.onnx \
  input.bvh_file=data/hc_mocap/walk.bvh \
  input.bvh_format=hc_mocap
```

继续阅读：[`docs/inference.md`](inference.md)

### 2. 我想接入 UDP 实时动捕

```bash
# Terminal 1
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx

# Terminal 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

继续阅读：[`docs/inference.md`](inference.md)

### 3. 我想理解配置怎么改

Teleopit 使用 Hydra 组合配置，最常改的是：

- `policy_hz`
- `pd_hz`
- `viewers`
- `input.bvh_file`
- `input.bvh_format`
- `controller.policy_path`

继续阅读：[`docs/configuration.md`](configuration.md)

### 4. 我想准备训练数据

推荐从统一 ingestion 开始：

```bash
python train_mimic/scripts/data/ingest_motion.py \
  --input data/hc_mocap_bvh \
  --source hc_mocap_v1 \
  --bvh_format hc_mocap \
  --manifest data/motion/manifests/v1.csv \
  --npz_root .
```

继续阅读：[`docs/dataset.md`](dataset.md)

### 5. 我想训练 / 导出 ONNX policy

```bash
python train_mimic/scripts/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file data/datasets/builds/twist2_full/train.npz
```

继续阅读：[`docs/training.md`](training.md)

### 6. 我想上真机 G1

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx
```

继续阅读：[`docs/sim2real.md`](sim2real.md)

## 常见误区

- **没有 policy 就直接运行**：`controller.policy_path` 是必填项。
- **把旧的 1402D / TWIST2 ONNX 拿来跑**：当前主路径只支持 mjlab policy（sim2sim 160D / 真机 154D）。
- **依赖默认 BVH 路径**：请显式设置 `input.bvh_file`，不要假设示例路径在本机存在。
- **把训练、推理、sim2real 混成一个安装步骤**：按需要安装 extras，排错会更简单。

## 下一步建议

- 第一次上手：先完成“离线跑一个 BVH”
- 想调 viewer / 录制 / realtime：读 [`docs/inference.md`](inference.md)
- 想系统理解配置：读 [`docs/configuration.md`](configuration.md)
- 想进入训练闭环：读 [`docs/dataset.md`](dataset.md) 和 [`docs/training.md`](training.md)
