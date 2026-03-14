# 训练数据系统（v2：Spec + Cache + Build）

当前推荐的数据集主路径只保留一个入口：**按 YAML spec 一键构建 train/val**。

## 主入口

```bash
python train_mimic/scripts/data/build_dataset_v2.py   --spec train_mimic/configs/datasets/twist2_full.yaml
```

或直接使用 twist2_full 包装脚本：

```bash
bash train_mimic/scripts/data/build_twist2_full.sh
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
bash train_mimic/scripts/data/build_twist2_full.sh --force
```

复用已有 cache，只重做 merge：

```bash
bash train_mimic/scripts/data/build_twist2_full.sh
```

加速 source 级转换：

```bash
bash train_mimic/scripts/data/build_twist2_full.sh --jobs 2
```

跳过 sampled FK 检查：

```bash
bash train_mimic/scripts/data/build_twist2_full.sh --skip-fk-check
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
python train_mimic/scripts/data/check_motion_npz_fk.py   --npz data/datasets/cache/twist2_full/npz_clips/<source>/<clip>.npz
```

## 数据清洗（人工 Review）

对已构建的数据集进行逐 clip 人工审阅，标注 keep/drop/skip，最终生成清洗后的训练数据。

### 目录结构

```text
data/datasets/review/<dataset>/
├── review_state.csv        # 审阅状态（每次 Save 即时落盘）
├── metrics_cache.json      # 异常指标缓存（自动生成）
├── filtered_manifest.csv   # 导出的保留清单
└── review_summary.json     # 导出统计摘要

data/datasets/builds/<dataset>_cleaned/
├── train.npz
├── val.npz
├── manifest_resolved.csv
└── build_info.json
```

### 步骤 1：初始化审阅清单

从已有 `manifest_resolved.csv` 生成 `review_state.csv`：

```bash
python train_mimic/scripts/data/init_review_manifest.py \
  --dataset twist2_full \
  --manifest data/datasets/builds/twist2_full/manifest_resolved.csv
```

- 已存在时拒绝覆盖，加 `--force` 强制重建。
- 会保留原始 clip 的 `weight`。

### 步骤 2：Web Viewer 审阅

启动浏览器端审阅工具（默认 http://localhost:8012）：

```bash
python train_mimic/scripts/data/review_dataset.py --dataset twist2_full
```

常用选项：

| 参数 | 说明 |
|------|------|
| `--skip_metrics` | 跳过异常指标预计算，加速启动 |
| `--sort suspicion_desc` | 按异常程度排序，优先审可疑 clip |
| `--sort source` | 按数据源排序 |
| `--port 8013` | 自定义端口 |

GUI 功能：
- **Playback**：播放 / 暂停 / 拖动帧 / 调速 / 重播
- **Annotation**：选择 Decision（Keep / Drop / Skip）、Difficulty、备注，点 Save & Next 保存并跳下一个未审 clip
- **Navigation**：上一个 / 下一个 / 跳转到指定编号 / Next Unreviewed
- **Stats**：实时显示审阅进度、已保留时长、按 source 统计

支持随时 Ctrl+C 退出，下次启动自动从未审 clip 继续。

### 步骤 3：导出保留清单

```bash
python train_mimic/scripts/data/export_reviewed_manifest.py \
  --review data/datasets/review/twist2_full/review_state.csv
```

- 不要求全部审完即可导出（只导出已标 keep 的 clip）
- 加 `--require_complete` 可要求全部审完后才允许导出
- 输出 `filtered_manifest.csv` + `review_summary.json`

### 步骤 4：重建清洗后数据集

```bash
python train_mimic/scripts/data/build_dataset_from_review.py \
  --filtered_manifest data/datasets/review/twist2_full/filtered_manifest.csv \
  --output_dir data/datasets/builds/twist2_full_cleaned \
  --target_fps 30
```

- `--target_fps 30`：默认值，将混合 FPS 的 clip 统一重采样到 30fps
- 缺失 NPZ 文件会报错，不会静默跳过
- 保留原始 source weight

训练时指向清洗后的数据集：

```bash
python train_mimic/scripts/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file data/datasets/builds/twist2_full_cleaned/train.npz \
  --num_envs 4096 --max_iterations 30000
```

### review_state.csv 字段说明

| 字段 | 说明 |
|------|------|
| `clip_id` | 唯一标识，格式 `{source}:{clip_name}` |
| `source` | 数据来源 |
| `file_rel` | NPZ 相对路径 |
| `resolved_split` | train / val |
| `num_frames` | 帧数 |
| `fps` | 原始帧率 |
| `duration_s` | 时长（秒） |
| `weight` | 采样权重（来自原始 manifest） |
| `decision` | keep / drop / skip / 空（未审） |
| `difficulty` | easy / medium / hard / bad_data / 空 |
| `issue_tags` | 逗号分隔的问题标签 |
| `note` | 自由备注 |
| `reviewed_at` | 审阅时间戳（UTC ISO） |

## Legacy

以下脚本继续保留，用于高级/历史场景，但不再是推荐主路径：
- `train_mimic/scripts/data/ingest_motion.py`
- `train_mimic/scripts/data/validate_dataset.py`
- `train_mimic/scripts/data/build_dataset.py`
- `train_mimic/scripts/data/migrate_legacy_dataset.py`
