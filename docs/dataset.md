# 数据集构建

当前训练侧只保留一条正式数据主线：

`typed source YAML -> 标准 NPZ clips -> train.npz / val.npz`

用户只需要理解一个命令：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml
```

## 输出目录

每个 dataset 的全部正式产物都落在同一个目录：

```text
data/datasets/<dataset>/
├── clips/
│   └── <source>/...
├── train.npz
├── val.npz
├── manifest_resolved.csv
└── build_info.json
```

这意味着导入、导出、备份一个 dataset 时，直接处理整个 `data/datasets/<dataset>/` 目录即可。

## YAML spec

`train_mimic/configs/datasets/twist2_full.yaml` 示例：

```yaml
name: twist2_full
target_fps: 30
val_percent: 5
hash_salt: ""
sources:
  - name: OMOMO_g1_GMR
    type: pkl
    input: data/twist2_retarget_pkl/OMOMO_g1_GMR
  - name: lafan1_v1
    type: bvh
    input: data/lafan1_bvh
    bvh_format: lafan1
```

字段说明：

- `name`: dataset 名称，对应输出目录 `data/datasets/<name>/`
- `target_fps`: merge 前统一重采样到的目标帧率
- `val_percent`: 基于 `clip_id` hash 的验证集比例
- `hash_salt`: 可选 split salt
- `sources[].name`: source 名称，也会成为 `clips/<source>/` 子目录名
- `sources[].type`: `bvh` / `pkl` / `npz`
- `sources[].input`: 原始输入文件或目录。对于 `type: npz`，目录应直接指向 clip 目录，不要指向已有 dataset 根目录
- `sources[].weight`: 可选源级别采样权重，默认 `1.0`
- `sources[].bvh_format`: 仅 `bvh` source 必填，当前支持 `lafan1` / `hc_mocap` / `nokov`
- `sources[].robot_name`: 仅 `bvh` source 使用，默认 `unitree_g1`；当前数据集 BVH 转换只支持 `unitree_g1`
- `sources[].max_frames`: 仅 `bvh` source 使用，默认 `0` 表示全长

## 转换规则

统一转换到标准 NPZ clip：

- `bvh -> retarget pkl -> npz clip`
- `pkl -> npz clip`
- `npz -> validate + copy/reuse`

最终 build 阶段只消费标准 NPZ clip，不再区分原始输入类型。

## 常用命令

强制重建整个 dataset 目录：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --force
```

按文件多进程并行转换：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --jobs 8
```

输出到自定义 datasets 根目录：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --output_root /tmp/my_datasets
```

跳过 sampled FK check：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --skip_fk_check
```

打印 build report：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --json
```

## 单独批量转换为 NPZ clips

如果你只想先把某一批原始数据转成标准 NPZ clip，而不做 train/val merge，可以用：

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type bvh \
    --input data/lafan1_bvh \
    --output data/datasets/lafan1_v1/clips/lafan1_v1 \
    --source lafan1_v1 \
    --bvh_format lafan1 \
    --jobs 8
```

或者：

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type pkl \
    --input data/twist2_retarget_pkl/OMOMO_g1_GMR \
    --output data/datasets/omomo_only/clips/omomo \
    --source omomo \
    --jobs 8
```

## 单独检查 clip

检查某个 NPZ 的 FK 一致性：

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/datasets/twist2_full/clips/<source>/<clip>.npz
```

推荐判据：

- `pos_max < 1e-3 m`
- `quat_mean < 0.05 rad`
- `quat_p95 < 0.10 rad`

## 和训练主线的连接

构建完成后，训练和评估直接消费：

```bash
python train_mimic/scripts/train.py \
    --task Tracking-Flat-G1-NoStateEst \
    --motion_file data/datasets/twist2_full/train.npz

python train_mimic/scripts/benchmark.py \
    --task Tracking-Flat-G1-NoStateEst \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/twist2_full/val.npz \
    --num_envs 1
```
