# 训练数据系统（v2：Spec + Cache + Build）

当前推荐的数据集主路径只保留一个入口：**按 YAML spec 一键构建 train/val**。

## 主入口

```bash
python scripts/data/build_dataset_v2.py   --spec train_mimic/configs/datasets/twist2_full.yaml
```

或直接使用 twist2_full 包装脚本：

```bash
bash scripts/data/build_twist2_full.sh
```

默认输出：

```text
data/datasets/
├── cache/
│   └── twist2_full/
│       └── npz_clips/<source>/...
└── builds/
    └── twist2_full/
        ├── train.npz
        ├── val.npz
        ├── manifest_resolved.csv
        └── build_info.json
```

## Spec 文件

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
- `name`：dataset 名称，同时决定 cache/build 目录名；
- `target_fps`：最终 `train.npz / val.npz` 的统一 fps；
- `val_percent`：按 `clip_id` 哈希切分到 `val` 的比例；
- `hash_salt`：可选切分盐值；
- `sources`：输入 PKL 目录列表。

## 日常使用

完整重建：

```bash
bash scripts/data/build_twist2_full.sh --force
```

复用已有 cache，只重做 merge：

```bash
bash scripts/data/build_twist2_full.sh
```

加速 source 级转换：

```bash
bash scripts/data/build_twist2_full.sh --jobs 2
```

跳过 sampled FK 检查：

```bash
bash scripts/data/build_twist2_full.sh --skip-fk-check
```

## 训练接入

训练：

```bash
python train_mimic/scripts/train.py   --task Tracking-Flat-G1-v0   --motion_file data/datasets/builds/twist2_full/train.npz   --num_envs 4096 --max_iterations 30000
```

评估：

```bash
python train_mimic/scripts/benchmark.py   --task Tracking-Flat-G1-v0   --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt   --motion_file data/datasets/builds/twist2_full/val.npz   --num_envs 1
```

## 校验

构建时默认会对每个 source 抽样若干 clip 做 FK 一致性检查。
如果需要手工检查单个 clip：

```bash
python scripts/data/check_motion_npz_fk.py   --npz data/datasets/cache/twist2_full/npz_clips/<source>/<clip>.npz
```

## Legacy

以下脚本继续保留，用于高级/历史场景，但不再是推荐主路径：
- `scripts/ingest_motion.py`
- `scripts/data/validate_dataset.py`
- `scripts/data/build_dataset.py`
- `scripts/data/migrate_legacy_dataset.py`
