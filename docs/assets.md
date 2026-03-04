# USD/URDF 资产管理

## 目录结构

```
train_mimic/assets/g1/
├── g1_29dof_rev_1_0.urdf              # Unitree 官方原始 URDF
├── g1_custom_collision_29dof.urdf     # 简化碰撞体的 URDF（训练用）
└── usd/
    ├── g1_29dof.usd                   # composition root（引用下方子文件）
    ├── config.yaml                    # UrdfConverter 生成的配置（相对路径）
    ├── configuration/                 # UrdfConverter 生成的 USD 分层
    │   ├── g1_29dof_base.usd          # 几何/mesh 层（~28MB）
    │   ├── g1_29dof_physics.usd       # 物理层（关节、刚体、碰撞）
    │   ├── g1_29dof_robot.usd         # 机器人配置层
    │   └── g1_29dof_sensor.usd        # 传感器层
    └── g1_29dof_custom_pxr.usd.bak    # 旧版 USD 备份（不要使用）
```

## URDF 来源

训练使用的 URDF 是 `g1_custom_collision_29dof.urdf`，基于 Unitree 官方 `g1_29dof_rev_1_0.urdf` 修改：
- 注释掉了原始 mesh 碰撞体
- 替换为简化的几何碰撞体（cylinder/sphere/box）
- 保留了 29 个旋转关节和所有 body 名称

## USD 转换

### 推荐方式：Isaac Lab UrdfConverter

```bash
conda activate teleopit_isaaclab
OMNI_KIT_ACCEPT_EULA=YES python train_mimic/scripts/convert_urdf_isaaclab.py --headless
```

转换脚本 `convert_urdf_isaaclab.py` 使用 Isaac Lab 官方 `UrdfConverter`，配置：

| 参数 | 值 | 原因 |
|------|-----|------|
| fix_base | False | 训练需要自由站立的机器人 |
| merge_fixed_joints | False | 保留 key body 名称（left_rubber_hand, head_mocap 等） |
| force_usd_conversion | True | 强制重新转换 |
| joint_drive stiffness | 0.0 | 训练代码使用自定义 PD 控制 |
| joint_drive damping | 0.0 | 同上 |
| joint_drive target_type | "none" | 不使用 Isaac Lab 内置驱动 |

❗ **重要**：UrdfConverter 会在 `config.yaml` 中写入绝对路径。转换完成后必须手动修改为相对路径，否则换机器后 USD 无法加载：

```yaml
# config.yaml — 转换后手动修改为：
asset_path: ../g1_custom_collision_29dof.urdf  # 相对于 usd/ 目录
usd_dir: .                                     # 当前目录
```

### 不要使用：自定义 pxr 转换器

`convert_urdf.py`（568 行）使用原始 `pxr` API 手动构建 USD，会创建闭合关节链，导致 PhysX GPU 求解器挂起。仅保留作为参考，不要用于生成训练用 USD。

## USD 结构验证

使用验证脚本检查 USD 结构：

```bash
python train_mimic/scripts/verify_usd_structure.py
```

验证项目：
- 29 个旋转关节（RevoluteJoint）
- 关节名称匹配预期的 29-DOF 集合
- 9 个 key body 存在：left_rubber_hand, right_rubber_hand, left_ankle_roll_link, right_ankle_roll_link, left_knee_link, right_knee_link, left_elbow_link, right_elbow_link, head_mocap
- 8 个 FixedJoint（用于连接末端执行器和传感器，非闭合环路）

## 关节规格

29 DOF 分布：

| 部位 | 关节数 | 力矩限制 (Nm) |
|------|--------|---------------|
| 髋关节 (pitch) | 2 | 88 |
| 髋关节 (roll) | 2 | 139 |
| 髋关节 (yaw) | 2 | 88 |
| 膝关节 | 2 | 139 |
| 踝关节 | 4 | 50 |
| 腰部 | 3 | 50-88 |
| 肩关节 | 6 | 25 |
| 肘关节 | 2 | 25 |
| 腕关节 | 6 | 5-25 |

## Unitree 官方参考资产

Unitree 官方 Isaac Lab 仿真项目的 USD 资产可从 HuggingFace 下载：

```bash
cd unitree_sim_isaaclab
bash fetch_assets.sh
```

注意：官方 wholebody_dex1 USD 包含 33 个关节（29 body + 4 hand），与本项目的 29 DOF 配置不兼容，不能直接使用。
