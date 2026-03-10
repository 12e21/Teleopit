# train_mimic 修复与对齐计划

相关项目路径：`/inspire/qb-dev/project/robot-action/czxs253130598/home_qbdev/project/teleop_projects/TWIST2`

本文整合两类内容：

1. 当前 `train_mimic` 训练链路已经确认的问题；
2. 如果输入数据来自 `TWIST2` PKL，想保留 `TWIST2` 原始训练语义，还需要补齐的关键环节。

---

## 一、问题总览

当前问题可以分成两层：

### A. 训练链路本身的语义错误

这些问题即使不考虑 `TWIST2`，也会直接影响训练正确性：

- merged 数据丢失 clip 边界；
- 采样基于整条时间轴而不是 clip；
- episode 可能跨 clip，导致 reference 突变；
- 训练未使用 NPZ `fps`；
- reference 播放速度与物理时间不一致；
- pose / velocity 标签时间尺度不一致；
- adaptive sampling 统计建立在错误时间轴上；
- 训练与推理链路时间语义不一致。

### B. 与 TWIST2 原始训练语义的偏差

这些问题不一定会让代码立即报错，但会让“同一份 PKL 数据”训练出的任务目标发生变化：

- 缺少按-motion采样语义；
- 缺少 motion weight；
- 缺少 motion-level curriculum；
- 缺少局部根增量标签；
- key-body 与 reward 语义可能已偏离 TWIST2；
- 缺少单-motion loop / 连续时间插值的原始语义。

---

## 二、已确认的问题清单

### 1. 训练集按时间维直接拼接，丢失 clip 边界

**现状**

- 多个 clip 在 build 时沿 `axis=0` 直接拼接成单个 `train.npz` / `val.npz`。
- 输出文件不包含 `clip_id`、`clip_start`、`clip_end`、`source` 等边界信息。

**代码位置**

- `train_mimic/data/dataset_lib.py:330`
- `train_mimic/data/dataset_lib.py:395`
- `train_mimic/data/dataset_lib.py:400`

**影响**

- 运行时无法知道“当前 reference 属于哪个 clip”。
- 无法正确实现 clip 级采样与 clip 内连续性约束。

### 2. 训练采样基于整条 merged timeline，而不是 clip 粒度

**现状**

- `MotionLoader` 只读取 merged 后的大数组和 `time_step_total`。
- `MotionCommand` 的 `uniform/adaptive` 都是采样整条时间轴上的位置。

**代码位置**

- `train_mimic/tasks/tracking/mdp/commands.py:32`
- `train_mimic/tasks/tracking/mdp/commands.py:56`
- `train_mimic/tasks/tracking/mdp/commands.py:80`
- `train_mimic/tasks/tracking/mdp/commands.py:246`
- `train_mimic/tasks/tracking/mdp/commands.py:289`

**影响**

- 长 clip 自动拥有更多采样质量。
- 训练目标偏向“全局帧空间”而不是“motion 集合”。

### 3. episode 可以跨 clip，导致 reference 动作硬切换

**现状**

- 每步 `time_steps += 1` 顺序推进。
- 只有 reset 或走到整个 merged 文件末尾才重采样。
- clip 尾部后面可能直接读到下一个 clip 头部。

**代码位置**

- `train_mimic/tasks/tracking/mdp/commands.py:365`
- `train_mimic/tasks/tracking/mdp/commands.py:367`
- `/inspire/qb-dev/project/robot-action/czxs253130598/home_qbdev/local/miniconda3/envs/teleopit/lib/python3.10/site-packages/mjlab/managers/command_manager.py:77`
- `/inspire/qb-dev/project/robot-action/czxs253130598/home_qbdev/local/miniconda3/envs/teleopit/lib/python3.10/site-packages/mjlab/managers/command_manager.py:96`

**影响**

- reference 在 clip 边界突变。
- reward / termination 会被不连续监督污染。

### 4. 训练没有使用 NPZ 中的 `fps` 做时间对齐

**现状**

- `fps` 被写入并保留在 NPZ 中。
- 但 `MotionLoader` 不读取 `fps`。
- 训练只按 step 数推进 reference。

**代码位置**

- `train_mimic/scripts/convert_pkl_to_npz.py:149`
- `train_mimic/data/dataset_lib.py:396`
- `train_mimic/tasks/tracking/mdp/commands.py:36`
- `train_mimic/tasks/tracking/mdp/commands.py:366`

**影响**

- 训练参考时间轴和 motion 真实时间轴脱钩。

### 5. reference 播放速度与物理时间轴不一致

**现状**

- 当前环境 step 为 `0.005 * 4 = 0.02s`，即 `50Hz`。
- 若数据集是 `30fps`，现在等于每 `0.02s` 播放一帧 `30fps` motion。

**代码位置**

- `train_mimic/tasks/tracking/tracking_env_cfg.py:299`
- `train_mimic/tasks/tracking/tracking_env_cfg.py:308`
- `train_mimic/tasks/tracking/tracking_env_cfg.py:309`
- `train_mimic/tasks/tracking/mdp/commands.py:366`

**影响**

- `30fps` motion 被按 `50fps` 节奏消费，约等于 `1.67x` 加速。
- 一个物理 `10s` episode 会消耗约 `16.7s` 的 `30fps` motion。

### 6. pose / velocity 监督在时间尺度上不一致

**现状**

- `joint_vel/body_*_vel` 是按 `dt = 1/fps` 生成的。
- 但 pose/orientation 却按错误的 step-rate 在训练中被消费。

**代码位置**

- `train_mimic/scripts/convert_pkl_to_npz.py:110`
- `train_mimic/scripts/convert_pkl_to_npz.py:132`
- `train_mimic/scripts/convert_pkl_to_npz.py:144`
- `train_mimic/tasks/tracking/tracking_env_cfg.py:206`

**影响**

- pose reward 和 velocity reward 隐含不同时间尺度。
- 策略无法同时完美满足两者。

### 7. adaptive sampling 的失败统计建立在错误时间轴上

**现状**

- `adaptive` 用大时间轴的 bin 做失败统计。
- 失败位置不是 clip 内真实时间，也不是按 motion 组织。

**代码位置**

- `train_mimic/tasks/tracking/mdp/commands.py:89`
- `train_mimic/tasks/tracking/mdp/commands.py:247`
- `train_mimic/tasks/tracking/mdp/commands.py:258`
- `train_mimic/tasks/tracking/mdp/commands.py:395`

**影响**

- 采样偏置不再严格对应“哪些 motion 更难”。

### 8. 训练链路与推理链路的时间对齐语义不一致

**现状**

- `teleopit` 推理链路显式用 `policy_time * input_fps` 对齐参考帧。
- `train_mimic` 没有对应逻辑。

**代码位置**

- `teleopit/sim/loop.py:383`
- `teleopit/sim/loop.py:395`
- `train_mimic/tasks/tracking/mdp/commands.py:366`

**影响**

- 同一份数据在训练和推理中具有不同时间语义。

### 9. 当前 v0 / v1 配置已经暴露 sampling 问题

**现状**

- `v0` 默认 `adaptive`。
- `v1` 改为 `uniform`，文档中明确写了是为了避免 adaptive 在 general motion 上的坏行为。

**代码位置**

- `train_mimic/tasks/tracking/mdp/commands.py:486`
- `train_mimic/tasks/tracking/config/g1_v1/flat_env_cfg.py:98`
- `docs/training.md:548`

**影响**

- 当前项目已经在配置层面绕开部分问题，但还没从根源修复。

---

## 三、TWIST2 原始训练链路的关键语义

### 1. 按 motion 粒度加载与采样

- 每个 PKL 单独保留 `fps`、`dt`、`length`、`weight`。
- 训练时先采样 `motion_id`，再采样该 motion 内时间。

关键代码：

- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:58`
- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:274`
- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:302`
- `project/teleop_projects/TWIST2/legged_gym/legged_gym/envs/base/humanoid_mimic.py:126`

### 2. 按连续时间推进 reference

- reference time = `episode_length_buf * dt + motion_time_offset`
- `calc_motion_frame()` 做插值，不是整帧跳转。

关键代码：

- `project/teleop_projects/TWIST2/legged_gym/legged_gym/envs/base/humanoid_mimic.py:164`
- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:347`

### 3. 单 motion 内 loop 与 root 位移连续

- 超过 motion 长度时在当前 motion 内循环。
- 使用 `root_pos_delta` 保持循环平移连续。

关键代码：

- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:193`
- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:348`
- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:372`

### 4. 局部根增量标签参与 reward / eval

- `root_pos_delta_local`
- `root_rot_delta_local`

关键代码：

- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:198`
- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:204`
- `project/teleop_projects/TWIST2/legged_gym/legged_gym/envs/base/humanoid_mimic.py:713`
- `project/teleop_projects/TWIST2/legged_gym/legged_gym/envs/base/humanoid_mimic.py:1047`

### 5. motion weight 与 curriculum

- 支持 YAML 配置每个 motion 的 `weight`。
- 支持 motion-level difficulty / curriculum。

关键代码：

- `project/teleop_projects/TWIST2/legged_gym/motion_data_configs/twist2_dataset.yaml:1`
- `project/teleop_projects/TWIST2/pose/pose/utils/motion_lib_pkl.py:309`
- `project/teleop_projects/TWIST2/legged_gym/legged_gym/envs/base/humanoid_mimic.py:299`

---

## 四、对齐 TWIST2 的改造清单

## P0：必须先做

### P0-1 保留 clip / motion 元信息

目标：让训练运行时知道“当前跟踪的是哪个 motion”。

建议：

- 在 build 后保留 `clip_id`、`clip_start`、`clip_end`、`fps`、`duration`、`weight`。
- 或直接引入一个 runtime `MotionLib`，不再把训练主输入简化成单个扁平大 NPZ。

### P0-2 训练采样改成 `motion_id + motion_time`

目标：恢复先选 motion，再选 motion 内起点的语义。

建议：

- reset 时先按 motion 权重采样；
- 再在 motion 内采样初始时间；
- episode 内只在当前 motion 时间轴上推进。

### P0-3 按真实时间播放 reference

目标：修复 `fps` 失效问题。

建议：

- 引入 `motion_time_offset`；
- 用 `env.step_dt` 累加 reference time；
- 用时间插值获取 reference 状态。

### P0-4 禁止跨 clip 硬切换

目标：消灭 clip 边界 reference 突变。

建议：

- 到达 clip 末尾时 reset；
- 或只在同一 motion 内 loop；
- 严禁继续顺着 merged timeline 读到下一个 clip。

## P1：建议补齐

### P1-1 补 `root_pos_delta_local / root_rot_delta_local`

目标：恢复 TWIST2 的局部根增量监督语义。

建议：

- 在 build 时离线保存；
- 或在 runtime 连续 reference 上在线计算；
- 但必须与 `fps` 对齐。

### P1-2 恢复 motion weight 采样

目标：避免“长 clip 自动更常被抽到”的隐式偏置。

建议：

- 从 manifest/spec 中保留显式 `weight`；
- reset 时按 `weight` 采样 motion。

### P1-3 校验 key-body 与 reward 语义

目标：确认当前 tracking 目标是否仍然代表 TWIST2 原始任务。

建议：

- 对齐 key body 集合；
- 对齐 reward 的 body 监督语义；
- 若不对齐，文档中明确声明训练目标已变化。

### P1-4 评估是否要引入 motion-level curriculum

目标：让“难 motion / 易 motion”的采样调节建立在 motion 粒度，而不是大时间轴 bin 上。

建议：

- 优先使用 motion 完成率驱动 curriculum；
- adaptive fail-bin 仅作为补充，而不是主机制。

## P2：可选增强

### P2-1 motion decomposition

- 把超长 motion 切成固定长度子 motion；
- 避免超长 clip 对采样分布造成过强支配。

### P2-2 error-aware sampling

- 使用 key-body 最大误差增强 motion 采样；
- 建议在 P0/P1 语义对齐后再考虑引入。

### P2-3 motion-domain-randomization

- 当前不是最优先；
- 时间轴与监督语义未修正前，不建议先开。

---

## 五、建议实施顺序

### 阶段 1：先修正确性

1. 保留 motion / clip 元信息
2. 改成 `motion_id + motion_time` 采样
3. 用真实 `fps/dt` 做 reference 时间推进
4. 禁止跨 clip 硬切换

### 阶段 2：再修监督完整性

5. 新增局部根增量标签
6. 校验 key-body / reward 语义
7. 恢复 motion weight 采样

### 阶段 3：再修训练策略

8. 引入 motion-level curriculum
9. 视需要加入 decomposition
10. 视需要加入 error-aware sampling / motion DR

---

## 六、落地建议

如果近期目标是尽快让 `train_mimic` 用 `TWIST2` 数据稳定训练，建议最小可行路径如下：

- **第一步**：不要再把训练主输入仅视为 merged NPZ；
- **第二步**：先补 runtime motion 索引层；
- **第三步**：让 reference 播放从“按帧推进”改成“按时间推进”；
- **第四步**：到 clip 末尾时 reset 或同 motion loop；
- **第五步**：再决定是否补局部根增量与 curriculum。

一句话总结：

当前 `train_mimic` 的问题不只是一个 bug，而是“数据组织语义 + 采样语义 + 时间语义”同时偏离了 `TWIST2`。如果只修其中一处，训练目标仍然会和原始 PKL 训练任务不一致。
