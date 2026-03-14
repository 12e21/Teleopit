# 训练指南

本文档只描述当前 `train_mimic` 的官方训练主线。

> 入口导航：数据准备看 [`docs/dataset.md`](dataset.md)，项目整体边界看 [`docs/architecture.md`](architecture.md)。

## 当前主线

- 官方训练任务只有一个：`Tracking-Flat-G1-NoStateEst`
- 该任务对应 **154D no-state-estimation** actor 观测，也是当前 sim2real 唯一支持的训练路径
- 旧 task `Tracking-Flat-G1-v2-NoStateEst` 仅保留为兼容别名；`v0`、`v1`、state-estimation 任务不再是正式支持接口
- adaptive sampling 相关实现仍保留在代码里作为内部参考，但不再通过公开 task 暴露

## 环境安装

```bash
conda create -n teleopit python=3.10
conda activate teleopit

cd Teleopit
pip install -e '.[train]'
```

快速校验：

```bash
python -c "import train_mimic.tasks; print('training OK')"
```

## 数据集准备

当前唯一推荐的数据构建入口是 YAML spec 驱动的 dataset pipeline：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml
```

常用变体：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --force

python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --jobs 4

bash train_mimic/scripts/data/build_twist2_full.sh
```

构建产物：

- `data/datasets/builds/twist2_full/train.npz`
- `data/datasets/builds/twist2_full/val.npz`
- `data/datasets/builds/twist2_full/build_info.json`

若要单独检查某个 clip 的 FK 标签一致性：

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/datasets/cache/twist2_full/npz_clips/<source>/<clip>.npz
```

## 训练

快速 smoke test：

```bash
python train_mimic/scripts/train.py \
    --task Tracking-Flat-G1-NoStateEst \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets/builds/twist2_full/train.npz
```

完整训练：

```bash
python train_mimic/scripts/train.py \
    --task Tracking-Flat-G1-NoStateEst \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets/builds/twist2_full/train.npz
```

单机多卡：

```bash
python train_mimic/scripts/train.py \
    --task Tracking-Flat-G1-NoStateEst \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets/builds/twist2_full/train.npz
```

说明：

- `--num_envs` 在多卡模式下表示每张卡的环境数
- 默认 logger 是 tensorboard；传 `--wandb_project <name>` 才会启用 wandb
- `--motion_file` 必须指向单个 merged NPZ
- 如果传入旧别名 `Tracking-Flat-G1-v2-NoStateEst`，脚本会给出弃用提示并自动映射到新 task

## 导出与评估

导出 ONNX：

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
    --output policy.onnx
```

播放 checkpoint：

```bash
python train_mimic/scripts/play.py \
    --task Tracking-Flat-G1-NoStateEst \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/builds/twist2_full/val.npz
```

benchmark：

```bash
python train_mimic/scripts/benchmark.py \
    --task Tracking-Flat-G1-NoStateEst \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/builds/twist2_full/val.npz \
    --num_envs 1
```

录视频：

```bash
python train_mimic/scripts/benchmark.py \
    --task Tracking-Flat-G1-NoStateEst \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/builds/twist2_full/val.npz \
    --num_envs 1 \
    --video \
    --video_length 600
```

## 训练侧边界

当前 `train_mimic` 按 4 层组织：

```text
scripts
    -> app helpers
    -> task registry / env builder / runner cfg
    -> mjlab + rsl_rl

scripts/data
    -> dataset_builder
    -> dataset_lib / motion_fk / convert_pkl_to_npz
```

对应目录：

```text
train_mimic/
├── app.py                    # train/play/benchmark 共享应用层
├── data/
│   ├── dataset_builder.py    # YAML spec 数据集主线
│   ├── dataset_lib.py        # NPZ merge / inspect / hash 等通用工具
│   └── motion_fk.py          # FK consistency 检查
├── scripts/
│   ├── train.py
│   ├── play.py
│   ├── benchmark.py
│   ├── save_onnx.py
│   └── data/build_dataset.py
└── tasks/tracking/config/
    ├── registry.py           # 官方 task 注册
    ├── env.py                # 官方 env builder
    ├── rl.py                 # 官方 runner cfg
    └── profiles.py           # 内部 profile / sampling 参考实现
```

## 迁移说明

- `build_dataset_v2.py` 已并入 `build_dataset.py`
- manifest/review/export/migrate 那套 legacy 数据脚本已移除
- `Tracking-Flat-G1-v2-NoStateEst` 仍可用，但只作为兼容别名
- `Tracking-Flat-G1-v0*`、`Tracking-Flat-G1-v1*` 已退出正式支持面
