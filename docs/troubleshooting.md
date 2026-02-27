# 常见问题与解决方案

## PhysX sim.step() 挂起

### 症状

训练过程中 `sim.step()` 无限阻塞，表现为：
- 训练在某个迭代后卡住不动
- GPU 利用率降为 0
- 无错误输出，进程不退出

### 根因

旧版 USD 模型由自定义 `convert_urdf.py`（使用原始 `pxr` API）生成，会创建闭合关节链（closed articulation loops）。PhysX GPU 求解器在处理这种拓扑时会产生病态矩阵，导致求解无限循环。

### 解决方案

使用 Isaac Lab 官方 `UrdfConverter` 重新生成 USD：

```bash
conda activate teleopit_isaaclab
OMNI_KIT_ACCEPT_EULA=YES python teleopit_train/scripts/convert_urdf_isaaclab.py --headless
```

关键配置：
- `fix_base=False`（自由站立机器人）
- `merge_fixed_joints=False`（保留 `left_rubber_hand`、`head_mocap` 等 body 名称）
- `joint_drive` stiffness/damping = 0.0（使用自定义 PD 控制）

### 验证

```bash
# 运行 100 轮测试，确认无挂起
timeout 600 bash -c 'eval "$(conda shell.bash hook)" && conda activate teleopit_isaaclab && \
OMNI_KIT_ACCEPT_EULA=YES python teleopit_train/scripts/train.py \
  --task Isaac-G1-Mimic-v0 --num_envs 4 --max_iterations 100 --headless --seed 42 2>&1'
```

## Isaac Sim 启动警告

### libXt.so.6 / libGLU.so.1 缺失

```
libXt.so.6: cannot open shared object file
libGLU.so.1: cannot open shared object file
```

这些是显示库，在 `--headless` 模式下不影响功能，可以安全忽略。

### GLFW initialization failed

```
GLFW initialization failed
```

同上，headless 模式下正常。

### Unresolved reference prim path (visuals)

```
Unresolved reference prim path .../visuals/head_mocap
```

这是 USD 中视觉网格引用的警告，不影响物理仿真和训练。仅在需要渲染时才需要修复。

## wandb 相关

### wandb 未初始化

如果不需要 wandb 日志，不传 `--wandb_project` 参数即可。训练代码已处理了 wandb 未初始化的情况，不会崩溃。

### wandb.log() 报错

已在 `rsl_rl/runners/on_policy_runner_mimic.py` 中用 try-except 包裹，不会中断训练。

## conda 环境激活

在脚本中激活 conda 环境时，使用：

```bash
eval "$(conda shell.bash hook)" && conda activate teleopit_isaaclab
```

不要使用 `source activate`，在某些环境下会失败。
