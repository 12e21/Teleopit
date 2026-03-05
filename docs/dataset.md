# 训练数据系统（Manifest + Validate + Build）

本文档定义 train_mimic 的最小数据系统：在不改训练主入口（仍用 `--motion_file`）的前提下，实现可复现、可扩展的数据管理。

## 目录约定

建议目录（按需逐步迁移）：

```text
data/motion/
├── raw/<source>/...               # 原始数据（PKL/BVH）
├── npz_clips/<source>/...         # 按 clip 切分的标准 NPZ
├── manifests/<version>.csv        # 数据清单（唯一事实源）
└── builds/<version>/              # 构建产物
    ├── merged_train.npz
    ├── merged_val.npz
    ├── manifest_resolved.csv
    ├── validation_report.json
    └── build_info.json
```

> 当前仓库 `data/` 在 `.gitignore` 中，数据与构建产物默认不入库。

## Manifest 格式

CSV 必须包含以下字段：

| 字段 | 含义 |
|------|------|
| `clip_id` | 样本唯一 ID |
| `source` | 数据来源（如 `OMOMO_g1_GMR`） |
| `file_rel` | 相对 `--npz_root` 的 NPZ 路径 |
| `num_frames` | 帧数 |
| `fps` | 帧率 |
| `split` | `train` / `val` / 空（空时走哈希切分） |
| `weight` | 采样权重（v1 仅记录，不进入训练采样器） |
| `enabled` | `1/0` 或 `true/false` |
| `quality_tag` | 质量标签（如 `legacy`, `reviewed`） |

## 四个入口脚本

位于根 `scripts/` 与 `scripts/data/`：

1. `ingest_motion.py`（统一 ingestion：BVH/PKL/NPZ -> NPZ clips + manifest）
2. `migrate_legacy_dataset.py`
3. `validate_dataset.py`
4. `build_dataset.py`

核心逻辑位于 `train_mimic/data/dataset_lib.py`，根目录 CLI 仅做参数解析与调用。

### 0) 统一 ingestion（推荐）

```bash
# BVH 来源（自动做 BVH -> PKL -> NPZ，并自动追加 manifest）
python scripts/ingest_motion.py \
  --input data/hc_mocap_bvh \
  --source hc_mocap_v1 \
  --bvh_format hc_mocap \
  --manifest data/motion/manifests/v1.csv \
  --npz_root .

# PKL 来源
python scripts/ingest_motion.py \
  --input data/twist2_retarget_pkl/v1_v2_v3_g1 \
  --source mocap_v1 \
  --manifest data/motion/manifests/v1.csv \
  --npz_root .
```

manifest 自动维护规则：
- 新 `clip_id`：自动追加；
- 现有 `clip_id`：默认报错（fail-fast）；
- 显式 `--allow_update`：允许覆盖更新该行；
- `num_frames/fps/file_rel` 由脚本从输出 NPZ 自动填充，不需要手工填写。

### 1) 迁移旧 NPZ 目录到 manifest

```bash
python scripts/data/migrate_legacy_dataset.py \
  --input_npz_dir data/twist2_retarget_npz/OMOMO_g1_GMR \
  --output_manifest data/motion/manifests/v1.csv \
  --npz_root .
```

### 2) 校验 manifest 与 NPZ

```bash
python scripts/data/validate_dataset.py \
  --manifest data/motion/manifests/v1.csv \
  --npz_root .
```

### 3) 生成训练/验证 merged 产物

```bash
python scripts/data/build_dataset.py \
  --manifest data/motion/manifests/v1.csv \
  --dataset_version v1 \
  --npz_root . \
  --build_root data/motion/builds

# 若数据源存在混合 fps，可显式指定统一 fps（例如 30）
python scripts/data/build_dataset.py \
  --manifest data/motion/manifests/v1.csv \
  --dataset_version v1_30hz \
  --npz_root . \
  --build_root data/motion/builds \
  --target_fps 30
```

`--target_fps` 对齐方式：
- 仅在你显式设置该参数时启用；默认仍是严格同 fps 合并（不自动修补）；
- 按时间轴线性重采样：`new_t = round(old_t * target_fps / old_fps)`；
- 对 `joint_pos/joint_vel/body_*` 全部沿时间维插值；
- `body_quat_w` 插值后会重新单位化，避免非单位四元数。

默认切分策略：
- 若 `split` 非空，使用 manifest 指定值；
- 若 `split` 为空，使用 `md5(clip_id) % 100 < 5` 进入 `val`（默认 5%）；
- 可通过 `--val_percent` 与 `--hash_salt` 调整。

## 训练/评估接入

训练：

```bash
python train_mimic/scripts/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file data/motion/builds/v1/merged_train.npz \
  --num_envs 4096 --max_iterations 30000
```

评估：

```bash
python train_mimic/scripts/benchmark.py \
  --task Tracking-Flat-G1-v0 \
  --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
  --motion_file data/motion/builds/v1/merged_val.npz \
  --num_envs 1
```

## Scale Up 流程（推荐）

1. 新数据落到 `raw/<source>/`
2. 跑 `ingest_motion.py`（统一转换并自动追加/更新 manifest）
3. 跑 `validate_dataset.py`
4. 跑 `build_dataset.py` 产出新版本（如 `v1.1`, `v2`）
5. 训练与评估固定引用该版本 build 产物

版本原则：
- 构建版本不可变；修改数据即升版本；
- 实验记录必须包含 `dataset_version`。
