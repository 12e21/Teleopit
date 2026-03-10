# 数据清洗与人工标注方案

## 目标

实现一套面向 `twist2_full` 风格数据集的人工 review 流程：

1. 逐个 clip 播放参考动作，使用 web viewer 进行人工检查。
2. 对每个 clip 标注：
   - `keep` / `drop`
   - 难度等级
   - 备注
3. 持久化 review 进度，支持中断后续跑。
4. 实时显示：
   - 总进度
   - 已保留 clip 数
   - 已保留总时长
   - 按 source / split 的统计
5. 基于 review 结果导出“清洗后数据清单”，并重新 build 训练用 `train.npz / val.npz`。

## 范围

本期只做人审工作流，不在 review 阶段自动修改原始 PKL/NPZ。

- 原始数据保持只读。
- review 结果单独存放。
- 清洗后的数据通过“过滤清单 + rebuild”生成。
- 自动规则只做提示，不做静默裁剪或自动丢弃。

## 设计原则

- 人工决策优先：工具提供播放、统计、异常提示，但最终保留/舍弃由人工决定。
- 可恢复：任意时刻退出后，下次可继续未审 clip。
- 可追溯：每条标注都能回溯到具体 `clip_id`、来源文件、时间戳和备注。
- 不污染原数据：review 产物与 build 产物分目录存放。
- 兼容现有 build v2：尽量复用 `manifest_resolved.csv`、`build_dataset_v2.py`、现有 NPZ cache。

## 输入与输出

### 输入

- `data/datasets/builds/<dataset>/manifest_resolved.csv`
- `data/datasets/cache/<dataset>/npz_clips/<source>/*.npz`

### 新增输出

- `data/datasets/review/<dataset>/review_state.csv`
- `data/datasets/review/<dataset>/review_summary.json`
- `data/datasets/review/<dataset>/filtered_manifest.csv`
- `data/datasets/builds/<dataset>_cleaned/`

## 数据模型

新增 review 记录格式，建议 CSV，便于人工查看和脚本处理。

字段建议：

- `clip_id`
- `source`
- `file_rel`
- `resolved_split`
- `num_frames`
- `fps`
- `duration_s`
- `decision`：`keep` / `drop` / `skip`
- `difficulty`：`easy` / `medium` / `hard` / `bad_data`
- `issue_tags`：逗号分隔，如 `first_frame_jump,root_height_spike`
- `note`
- `reviewed_at`

说明：

- `decision` 为空表示未审。
- `skip` 用于暂不判断但保留 review 进度。
- `difficulty` 独立于 `decision`，便于后续做 curriculum 或子集训练。

## 实现阶段

### 阶段 1：准备 review 基础数据

目标：把 `manifest_resolved.csv` 转成 review 工作清单，并补充统计字段。

工作项：

- 新增脚本：`scripts/data/init_review_manifest.py`
- 读取 `manifest_resolved.csv`
- 为每个 clip 计算 `duration_s = num_frames / fps`
- 生成初始 `review_state.csv`
- 如果 review 文件已存在，拒绝覆盖，除非显式 `--force`

验收标准：

- 能从现有 `twist2_full` build 直接生成 review 状态文件
- review 文件行数与 manifest 一致

### 阶段 2：实现单 clip web viewer review 工具

目标：通过 web viewer 播放单个 clip，并在浏览器中完成人工标注。

建议方案：

- 新增脚本：`scripts/data/review_dataset.py`
- 优先复用现有 `viser` 能力，而不是新造前端框架
- viewer 只展示参考动作，不依赖 policy checkpoint

核心功能：

- 按顺序加载待审 clip
- 显示当前 clip 基本信息：
  - `clip_id`
  - `source`
  - `split`
  - `frames`
  - `fps`
  - `duration_s`
- 播放控制：
  - 播放 / 暂停
  - 重播
  - 上一个 / 下一个
  - 调速
  - 跳到首帧 / 尾帧
- 标注操作：
  - `keep`
  - `drop`
  - `skip`
  - 选择难度
  - 输入备注
  - 保存后自动进入下一个未审 clip

实现建议：

- 不走 `play.py + checkpoint` 这条链路
- 单独实现“参考动作可视化器”，直接从 NPZ 读 `body_pos_w / body_quat_w`
- 这样可以避免把“策略表现差”和“数据本身坏”混在一起

验收标准：

- 启动命令后可在浏览器打开 viewer
- 能完成逐 clip 播放与标注
- 每次保存后，`review_state.csv` 立即落盘

### 阶段 3：加入异常提示与排序能力

目标：降低人工 review 成本，但不替代人工判断。

建议增加的只读提示项：

- 首帧跳变幅度
- 根高度范围
- 根平移速度最大值
- body 线速度最大值
- body 角速度最大值
- clip 总时长

建议实现：

- 新增 `review_metrics` 预计算模块
- 首次运行 `review_dataset.py` 时缓存到 JSON 或 CSV
- UI 中显示 warning badge，但不自动 `drop`

建议排序方式：

- `unreviewed_first`
- `source`
- `duration_desc`
- `suspicion_desc`

验收标准：

- 能快速把明显异常 clip 排到前面
- 不会因为启发式判断静默过滤数据

### 阶段 4：加入进度与保留时长统计

目标：让人工 review 时实时知道已经筛出多少有效数据。

界面和终端都应显示：

- `reviewed / total`
- `keep / drop / skip`
- `review progress %`
- `kept duration`
- `kept train duration`
- `kept val duration`
- `kept duration by source`

统计口径：

- 时长统一使用 `num_frames / fps`
- 只统计 `decision == keep`
- `skip` 不计入保留时长

验收标准：

- 每次保存标注后统计即时更新
- 中断重启后统计结果一致

### 阶段 5：导出清洗结果

目标：根据 review 结果导出新的过滤清单。

新增脚本：

- `scripts/data/export_reviewed_manifest.py`

功能：

- 读取 `review_state.csv`
- 只保留 `decision == keep` 的 clip
- 输出 `filtered_manifest.csv`
- 输出汇总 `review_summary.json`

可选导出模式：

- `--min_difficulty medium` 之类的后处理暂不做第一版
- 第一版只做 keep/drop 过滤

验收标准：

- `filtered_manifest.csv` 只包含保留 clip
- 汇总里能看到保留数量、总时长、按 source 统计

### 阶段 6：基于 review 结果重建 cleaned 数据集

目标：从已保留的 clip 重新生成训练数据。

推荐做法：

- 不回退到旧版 manifest build
- 在 `build_dataset_v2.py` 路径上新增“按过滤清单 build”的能力

两种实现方案：

方案 A，推荐：

- 为 `build_dataset_v2.py` 增加 `--review_filter <filtered_manifest.csv>`
- build 时只 merge 被保留的 cache NPZ
- 输出到 `data/datasets/builds/<dataset>_cleaned/`

方案 B：

- 新增独立脚本 `scripts/data/build_dataset_from_review.py`
- 输入 `filtered_manifest.csv`
- 复用现有 merge / inspect / FK check 工具

建议优先选方案 B：

- 对现有 `build_dataset_v2.py` 侵入更小
- review 流程与原始 build 流程解耦
- 更适合快速迭代

验收标准：

- 能从 review 结果生成新的 `train.npz / val.npz`
- `manifest_resolved.csv` 只包含保留样本
- 统计时长与 review summary 一致

### 阶段 7：验证与文档

目标：保证工具可长期使用。

测试项：

- review state 初始化测试
- review 记录读写测试
- 过滤导出测试
- cleaned build 测试
- 统计结果测试

文档更新：

- `README.md`
- `docs/dataset.md`
- `AGENTS.md`

文档需要包含：

- review 初始化命令
- web viewer 启动命令
- 标注字段说明
- cleaned dataset rebuild 命令

## 建议新增文件

- `scripts/data/init_review_manifest.py`
- `scripts/data/review_dataset.py`
- `scripts/data/export_reviewed_manifest.py`
- `scripts/data/build_dataset_from_review.py`
- `train_mimic/data/review_lib.py`
- `train_mimic/data/review_metrics.py`

## 命令草案

初始化 review 清单：

```bash
python scripts/data/init_review_manifest.py \
  --dataset twist2_full \
  --manifest data/datasets/builds/twist2_full/manifest_resolved.csv
```

启动 web review：

```bash
python scripts/data/review_dataset.py \
  --dataset twist2_full \
  --review data/datasets/review/twist2_full/review_state.csv
```

导出保留清单：

```bash
python scripts/data/export_reviewed_manifest.py \
  --review data/datasets/review/twist2_full/review_state.csv \
  --output data/datasets/review/twist2_full/filtered_manifest.csv
```

重建清洗后的数据集：

```bash
python scripts/data/build_dataset_from_review.py \
  --dataset twist2_full \
  --review data/datasets/review/twist2_full/review_state.csv \
  --output_dataset twist2_full_cleaned
```

## 第一版明确不做

- 自动裁掉坏首帧并直接改写原始 PKL/NPZ
- 多用户协同 review
- 复杂前端页面或数据库
- 自动按 difficulty 重采样训练
- 在 viewer 中直接编辑骨架或修数据

## 风险与注意事项

- 若沿用 policy rollout viewer，人工会混淆“数据坏”和“策略差”，因此必须优先做参考动作 viewer。
- review 状态文件需要频繁落盘，避免长时间标注后丢进度。
- 若后续发现坏数据主要集中于“首帧污染”，也不要在第一版里自动修正；先保留人工标注闭环。
- build cleaned dataset 时必须严格报错，不允许悄悄跳过 review 文件里引用但缺失的 NPZ。

## 里程碑

### M1

- 能初始化 review 清单
- 能在 web viewer 中逐 clip 播放
- 能保存 `keep/drop/skip + note`

### M2

- 能显示异常提示
- 能显示进度和已保留总时长
- 能断点续审

### M3

- 能导出 `filtered_manifest.csv`
- 能 rebuild `twist2_full_cleaned`
- 文档和测试补齐

## 成功标准

满足以下条件即视为完成：

1. 你可以在浏览器里逐个审阅 clip。
2. 审阅过程中可随时看到已完成比例和已保留时长。
3. 审阅结果可稳定保存并恢复。
4. 审阅完成后可以一键生成 cleaned 数据集并用于重新训练。
