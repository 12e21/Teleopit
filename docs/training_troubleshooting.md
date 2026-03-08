# 训练问题排查

本文档记录 Teleopit 训练过程中的常见问题及解决方案。

> 入口导航：训练流程看 [`docs/training.md`](training.md)，数据准备看 [`docs/dataset.md`](dataset.md)，项目总入口看 [`docs/getting-started.md`](getting-started.md)。

> 说明：本文中的 `data/twist2_retarget_*` 路径主要是历史排障样本与旧数据目录名示例，不代表当前推荐的数据管理入口；新的训练数据组织方式以 `data/motion/` 下的 manifest/build 产物为准。

---

## 问题 1：Mean episode length = 1.00（机器人第一步就终止）

### 现象

训练日志常见以下组合：
- `Mean episode length: 1.00`
- `Episode_Termination/anchor_pos` 接近并行环境总数
- `Metrics/motion/error_anchor_pos` > `0.5m`
- `Metrics/motion/error_body_rot` 很大（常接近 `π`）

### 根本原因

通常不是 PPO 超参数本身的问题，而是 **motion NPZ 的监督标签与 MuJoCo 中实际 FK 不一致**。
在历史数据链路里，这类不一致主要有三种：

1. **body 位置坐标系错误**：把根局部坐标直接当世界坐标叠加；
2. **body 顺序错误**：沿用 PKL 的 38-body 顺序，而不是 mjlab G1 的 30-body 顺序；
3. **body 朝向 / 角速度标签错误**：把所有 body 近似成 root 朝向，导致 articulated pose reward 自相矛盾。

当前版本的 `convert_pkl_to_npz.py` 已修复上述问题：
- `body_pos_w/body_quat_w/body_ang_vel_w` 由 MuJoCo FK 重建；
- body 顺序与名称对齐 mjlab G1 robot；
- 可用 `scripts/data/check_motion_npz_fk.py` 做一致性校验。

### 快速排查

先检查当前 clip 是否与 FK 一致：

```bash
python scripts/data/check_motion_npz_fk.py     --npz data/motion/npz_clips/<source>/<clip>.npz
```

建议判据：
- `pos_max < 1e-3 m`
- `quat_mean < 0.05 rad`
- `quat_p95 < 0.10 rad`

如果检查失败，优先重新生成数据：

```bash
python train_mimic/scripts/convert_pkl_to_npz.py     --input data/twist2_retarget_pkl/<source>     --output data/twist2_retarget_npz/<source>     --merge
```

然后做一个短训练 smoke test：

```bash
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0     --num_envs 64 --max_iterations 100     --motion_file data/twist2_retarget_npz/<source>/merged.npz
```

预期现象：
- `Mean episode length` 明显大于 `1`
- `Metrics/motion/error_anchor_pos` 开始下降
- `Episode_Termination/anchor_pos` 不再在起步阶段爆满

---

## 问题 2：训练过程中 episode length 不增长

### 现象

训练 1000+ 轮后 `Mean episode length` 仍然很低（< 3），没有上升趋势。

### 可能原因

1. **运动数据质量问题**：retargeting 质量差，目标姿态不可达
2. **奖励函数权重不合理**：tracking reward 权重过低，regularization 权重过高
3. **超参数问题**：学习率过大/过小，clip_param 不合适
4. **终止条件过严**：阈值设置过小，机器人稍有偏差就终止

### 排查步骤

1. **检查运动数据**：用 `play.py` 可视化参考运动，确认姿态合理
2. **查看奖励分布**：检查各项 reward 的数值范围，确认 tracking reward 占主导
3. **放宽终止条件**：临时增大 `bad_anchor_pos` 阈值（0.25m → 0.5m）测试
4. **对比 mjlab 官方示例**：用 mjlab 内置的 G1 tracking task 训练，确认环境配置正确

---

## 问题 3：训练速度慢

### 现象

训练速度 < 1000 steps/s（预期 1500-2000 steps/s on RTX 4090）。

### 可能原因

1. **num_envs 过小**：并行环境数不足，GPU 利用率低
2. **视频录制开启**：`--video` 会显著降低速度
3. **wandb 同步慢**：网络问题导致日志上传阻塞

### 解决方案

1. 增加 `--num_envs` 到 4096（需要 24GB 显存）
2. 训练时关闭 `--video`，只在评估时录制
3. 使用 tensorboard 替代 wandb（默认）

---

## 问题 4：日志中反复出现 `nefc overflow - please increase njmax ...`

### 现象

训练日志中频繁出现类似警告：

```text
nefc overflow - please increase njmax to 257
nefc overflow - please increase njmax to 264
```

### 根本原因

这不是数据文件或 Python 训练脚本本身报错，而是 **MuJoCo 约束缓冲区不足**。

- `nefc`：当前仿真步中激活的标量约束数量
- `njmax`：仿真允许分配的最大约束数量

当机器人跌倒、发生大量足底接触或自碰撞时，活跃约束数会瞬间升高；如果超过 `njmax`，MuJoCo 就会打印该警告。

在 Teleopit 的 `mjlab` 训练链路里，真正控制这个上限的是 **tracking 环境的全局仿真配置**，而不是单独的机器人 XML。`mjlab` 上游 tracking 默认配置中 `sim.njmax=250`，这正是警告中常见 `257/264` 一类数值的来源。

### 解决方案

在 `train_mimic/tasks/tracking/config/g1/flat_env_cfg.py` 中覆盖训练仿真参数：

```python
self.sim.njmax = 500
if self.sim.nconmax is not None and self.sim.nconmax < 150_000:
    self.sim.nconmax = 150_000
```

当前仓库已经包含这一修复。

### 为什么不要只改机器人 XML

仅修改 G1 XML 或 `MjSpec` 中的 `njmax` 对这个问题通常不够，因为训练时实际生效的是 `mjlab` 的 **simulation-level** `njmax`。如果仿真层仍保持 `250`，即使机器人模型层更大，也仍然会触发 overflow。

### 验证方法

启动训练后观察日志：

- 若 warning 消失或显著减少，说明修复生效；
- 若仍出现，而且提示值接近或超过 `500`，说明当前训练场景接触更多，可继续提高到 `800`。

建议的下一档参数：

```python
self.sim.njmax = 800
```

### 影响

`nefc overflow` 不一定会立刻导致训练崩溃，但它意味着部分接触/约束被截断，可能降低物理精度并影响训练稳定性。若日志中持续刷出该警告，建议优先修复，而不是忽略。

---

## 问题 5：benchmark 视频异常（只有 1 帧 / 保存失败 / EGL 报错）

### 现象 A：视频只有 1 帧

常见原因：
1. 命令里设置了 `--video_length 1`
2. `--num_eval_steps` 太小

`benchmark.py` 的实际录制长度是：

```text
actual_video_length = min(video_length, num_eval_steps)
```

#### 解决方案

确保 `num_eval_steps >= video_length`，例如：

```bash
python train_mimic/scripts/benchmark.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/rsl_rl/g1_tracking/{run_name}/model_30000.pt \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz \
    --num_envs 1 \
    --num_eval_steps 2000 \
    --warmup_steps 100 \
    --video \
    --video_length 600 \
    --video_folder benchmark_results/videos/model_30000_eval
```

### 现象 B：报错 `indices should be either on cpu or on the same device`

示例报错：

```text
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```

这是旧版 `benchmark.py` 的设备不一致问题（`done_mask` 在 GPU，长度 buffer 在 CPU）。

#### 解决方案

更新到最新代码后重试。修复版本已将 episode 长度 buffer 放到和 rollout 相同设备，并在收集日志时转回 CPU。

### 现象 C：警告 `Unable to save last video! Did you call close()?`

该警告通常表示评估过程中发生异常，`RecordVideo` 没有机会正常 `close()`，导致视频收尾失败。

#### 解决方案

1. 先修复导致中断的首个异常（通常是设备或渲染后端问题）
2. 使用最新 `benchmark.py`（已在主循环异常时保证 `env.close()`）
3. 重新运行评估

### 现象 D：EGL/OpenGL 后端报错

常见日志：
- `libEGL warning: failed to open /dev/dri/renderD128: Permission denied`
- `Video renderer initialization failed...`
- `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`
- `AttributeError: 'NoneType' object has no attribute 'glGetError'`

#### 根因

常见原因有三类：
1. 当前用户对 GPU render 节点无权限；
2. 机器上缺少可用 EGL/GL 后端库；
3. conda 环境只装了部分 EGL / GLVND 库，导致 `libEGL` 被 conda 覆盖，但 `libOpenGL` / `libGLX` 等基础库不完整。

#### 解决方案

1. 先补齐 conda 侧 OpenGL/EGL 依赖：
   ```bash
   conda install -c conda-forge libopengl libglx libegl libglvnd pyopengl
   ```
2. 在当前项目环境中，补齐这组依赖后，`benchmark.py --video` 往往可以直接工作；通常不再需要额外手动设置 `MUJOCO_GL`、`PYOPENGL_PLATFORM` 或 EGL vendor 环境变量。
3. 如果仍失败，再检查 `/dev/dri/renderD*` 权限与用户组（通常需加入 `video`/`render` 组并重新登录）
4. 若机器无可用 GPU EGL，可尝试 CPU 渲染后端：
   ```bash
   MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python train_mimic/scripts/benchmark.py ... --video
   ```

---

## 其他问题

如遇到其他问题，请在 [GitHub Issues](https://github.com/your-repo/issues) 提交，附上：
- 完整的训练命令
- 训练日志（前 100 轮）
- `convert_pkl_to_npz.py` 版本（git commit hash）
- 环境信息（`python --version`, `pip list | grep -E "torch|rsl|mjlab"`）
