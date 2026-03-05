# 训练问题排查

本文档记录 Teleopit 训练过程中的常见问题及解决方案。

---

## 问题 1：Mean episode length = 1.00（机器人第一步就终止）

### 现象

训练日志显示：
- `Mean episode length: 1.00`
- `Episode_Termination/anchor_pos` 接近并行环境总数（如 64 envs 中有 62 个触发）
- `Metrics/motion/error_anchor_pos` > 0.5m（远超 0.25m 终止阈值）
- `Metrics/motion/error_body_rot` ≈ 1.5-1.8 rad（接近 π）

### 根本原因

`convert_pkl_to_npz.py` 的两个转换错误导致 NPZ 运动数据与 MuJoCo 仿真坐标系不匹配。

---

### Bug 1：body 位置坐标系错误

#### 问题描述

PKL 文件中 `local_body_pos` 是 body 在**根节点局部坐标系**下的位置，不是世界坐标系下的偏移量。

旧版转换代码直接相加，忽略了根节点的旋转：

```python
# 错误（旧版）
body_pos_w = local_body_pos + root_pos[:, None, :]
```

正确做法是先将局部坐标旋转到世界坐标系，再加上根节点位置：

```python
# 正确（新版）
root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw)
root_rot_expanded = np.broadcast_to(root_rot_wxyz[:, None, :], (T, nb, 4)).reshape(T * nb, 4)
local_body_pos_flat = local_body_pos.reshape(T * nb, 3)
body_pos_w = root_pos[:, None, :] + quat_rotate(root_rot_expanded, local_body_pos_flat).reshape(T, nb, 3)
```

#### 影响

当机器人处于大角度旋转姿态时（如 yaw=120°），肢体位置误差显著：
- 脚踝：0.08-0.19m
- 手腕：0.46m

这些误差导致机器人在 reset 后立即触发 `bad_anchor_pos` 或 `bad_motion_body_pos` 终止条件。

#### 验证方法

使用 MuJoCo FK 验证转换正确性：

```python
import mujoco
import numpy as np

# 加载 NPZ
npz = np.load("data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz", allow_pickle=True)

# 加载 MuJoCo model
model = mujoco.MjModel.from_xml_path("teleopit/retargeting/gmr/assets/unitree_g1/g1_sim2sim_29dof.xml")
data = mujoco.MjData(model)

# 设置第一帧状态
data.qpos[:3] = npz["body_pos_w"][0, 0]  # root position
data.qpos[3:7] = npz["body_quat_w"][0, 0]  # root quaternion (wxyz)
data.qpos[7:] = npz["joint_pos"][0]  # joint positions

mujoco.mj_kinematics(model, data)

# 对比 NPZ 和 FK 结果
for body_name in ["pelvis", "left_ankle_roll_link", "torso_link", "left_wrist_yaw_link"]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    fk_pos = data.xpos[body_id]
    npz_idx = list(npz["body_names"]).index(body_name)
    npz_pos = npz["body_pos_w"][0, npz_idx]
    error = np.linalg.norm(fk_pos - npz_pos)
    print(f"{body_name}: FK={fk_pos}, NPZ={npz_pos}, error={error:.4f}m")
```

**预期结果**：所有 body 的误差应 < 0.001m。

---

### Bug 2：body 顺序与 mjlab G1 robot 不匹配

#### 问题描述

PKL 文件有 **38 个 body**，mjlab G1 robot 只有 **30 个 body**，且顺序不同。

**PKL 额外的 8 个 body**（mjlab G1 没有）：
- `left_toe_link`, `right_toe_link`（脚趾）
- `pelvis_contour_link`（骨盆轮廓）
- `head_link`, `head_mocap`（头部）
- `imu_in_torso`（IMU 传感器）
- `left_rubber_hand`, `right_rubber_hand`（手部）

**顺序错位示例**：
```
PKL[7] = left_toe_link（mjlab 没有）
PKL[8] = pelvis_contour_link（mjlab 没有）
PKL[9] = right_hip_pitch_link → 应该在 mjlab[7]
```

#### 为什么会出错

`mjlab` 的 `MotionLoader` 使用 **robot body 索引**直接访问 NPZ：

```python
# mjlab/tasks/tracking/mdp/commands.py
body_indexes = robot.find_bodies(cfg.body_names)  # 返回 [0, 2, 4, 6, 8, ...]
body_pos_w = motion.body_pos_w[time_steps, body_indexes]  # 直接用索引访问
```

如果 NPZ 按 PKL 顺序保存 38 个 body，当访问 `body_pos_w[:, 7]` 时：
- **期望**：`right_hip_pitch_link`（mjlab robot 的第 7 个 body）
- **实际**：`left_toe_link`（PKL 的第 7 个 body）

结果是读取到完全错误的 body 位置数据。

#### 修复方法

新版 `convert_pkl_to_npz.py` 会：
1. 从 PKL 的 38 个 body 中选出 mjlab G1 需要的 30 个
2. 按 mjlab G1 robot body 顺序重新排列
3. 保存时使用 mjlab G1 的 body 名称列表

```python
# 定义 mjlab G1 robot body 顺序（30 个）
_MJLAB_G1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]

# 创建索引映射并重新排序
pkl_to_mjlab_idx = [body_names.index(n) for n in _MJLAB_G1_BODY_NAMES]
body_pos_w = body_pos_w[:, pkl_to_mjlab_idx]  # (T, 38, 3) → (T, 30, 3)
body_quat_w = body_quat_w[:, pkl_to_mjlab_idx]
body_lin_vel_w = body_lin_vel_w[:, pkl_to_mjlab_idx]
body_ang_vel_w = body_ang_vel_w[:, pkl_to_mjlab_idx]

# 保存时使用 mjlab body 名称
np.savez(npz_path, ..., body_names=np.array(_MJLAB_G1_BODY_NAMES, dtype=str))
```

---

### 解决方案

#### 1. 检查是否使用旧版脚本生成的 NPZ

```bash
python -c "
import numpy as np
npz = np.load('data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz', allow_pickle=True)
print('Body count:', npz['body_pos_w'].shape[1])
print('Expected: 30 (mjlab G1 robot)')
if npz['body_pos_w'].shape[1] == 38:
    print('WARNING: NPZ has 38 bodies (PKL ordering), need to regenerate!')
"
```

#### 2. 重新生成 NPZ 数据

```bash
# 删除旧数据
rm -rf data/twist2_retarget_npz/OMOMO_g1_GMR

# 使用修复后的脚本重新转换
python train_mimic/scripts/convert_pkl_to_npz.py \
    --input data/twist2_retarget_pkl/OMOMO_g1_GMR \
    --output data/twist2_retarget_npz/OMOMO_g1_GMR \
    --merge
```

转换完成后验证：
```bash
python -c "
import numpy as np
npz = np.load('data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz', allow_pickle=True)
print('Body count:', npz['body_pos_w'].shape[1])
print('Body names:', list(npz['body_names']))
print('First body should be pelvis:', npz['body_names'][0])
print('Body 15 should be torso_link:', npz['body_names'][15])
"
```

#### 3. 验证修复效果

快速训练 100 轮：

```bash
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 64 --max_iterations 100 \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz
```

**预期指标**（第 99 轮）：
- `Mean episode length` > 5（理想情况 8-10）
- `Metrics/motion/error_anchor_pos` < 0.2m
- `Episode_Termination/anchor_pos` < 5（< 8% × 64 envs）
- `Metrics/motion/error_body_rot` < 1.0 rad

如果指标符合预期，说明修复成功，可以开始完整训练。

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

## 其他问题

如遇到其他问题，请在 [GitHub Issues](https://github.com/your-repo/issues) 提交，附上：
- 完整的训练命令
- 训练日志（前 100 轮）
- `convert_pkl_to_npz.py` 版本（git commit hash）
- 环境信息（`python --version`, `pip list | grep -E "torch|rsl|mjlab"`）
