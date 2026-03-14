# 数据集构建

当前训练侧只保留一条正式数据主线：`YAML spec -> NPZ clip cache -> train.npz / val.npz`。

> 入口导航：训练主线看 [`docs/training.md`](training.md)，系统边界看 [`docs/architecture.md`](architecture.md)。

## 快速开始

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml
```

或使用仓库内置 wrapper：

```bash
bash train_mimic/scripts/data/build_twist2_full.sh
```

## 输入与输出

输入：

- `train_mimic/configs/datasets/*.yaml`
- 每个 source 对应的 PKL 目录

输出：

- `data/datasets/cache/<dataset>/npz_clips/<source>/...`
- `data/datasets/builds/<dataset>/train.npz`
- `data/datasets/builds/<dataset>/val.npz`
- `data/datasets/builds/<dataset>/manifest_resolved.csv`
- `data/datasets/builds/<dataset>/build_info.json`

## YAML spec

`train_mimic/configs/datasets/twist2_full.yaml` 示例：

```yaml
name: twist2_full
target_fps: 30
val_percent: 5
hash_salt: ""
sources:
  - name: OMOMO_g1_GMR
    input: data/twist2_retarget_pkl/OMOMO_g1_GMR
  - name: AMASS_g1_GMR8
    input: data/twist2_retarget_pkl/AMASS_g1_GMR8
```

字段说明：

- `name`: dataset 名称，同时决定 cache/build 目录名
- `target_fps`: merge 前统一重采样到的目标帧率
- `val_percent`: 基于 `clip_id` hash 的验证集比例
- `hash_salt`: 可选 split salt
- `sources[].input`: PKL 输入目录，相对仓库根目录解析
- `sources[].weight`: 可选源级别采样权重，默认 `1.0`

## 常用命令

强制重建：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --force
```

并行转换：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --jobs 4
```

跳过 sampled FK check：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --skip_fk_check
```

打印最终 build report：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --json
```

## 单独转换或检查 clip

如果你只想先把一个 source 目录转换成 NPZ clip：

```bash
python train_mimic/scripts/convert_pkl_to_npz.py \
    --input data/twist2_retarget_pkl/OMOMO_g1_GMR \
    --output data/datasets/cache/twist2_full/npz_clips/OMOMO_g1_GMR
```

检查某个 NPZ 的 FK 一致性：

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/datasets/cache/twist2_full/npz_clips/OMOMO_g1_GMR/<clip>.npz
```

推荐判据：

- `pos_max < 1e-3 m`
- `quat_mean < 0.05 rad`
- `quat_p95 < 0.10 rad`

## 和训练主线的连接

构建完成后，训练和评估都直接消费单文件 NPZ：

```bash
python train_mimic/scripts/train.py \
    --task Tracking-Flat-G1-NoStateEst \
    --motion_file data/datasets/builds/twist2_full/train.npz

python train_mimic/scripts/benchmark.py \
    --task Tracking-Flat-G1-NoStateEst \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/builds/twist2_full/val.npz \
    --num_envs 1
```

## 已移除的旧路径

以下内容已不再是正式支持接口：

- `build_dataset_v2.py` 旧脚本名
- manifest/review/export/build-from-review 这套人工 review 数据流
- legacy manifest CSV 驱动构建脚本与迁移脚本

如果你之前使用旧名字，直接迁移到 `train_mimic/scripts/data/build_dataset.py`。
