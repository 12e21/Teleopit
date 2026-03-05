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
OMNI_KIT_ACCEPT_EULA=YES python train_mimic/scripts/convert_urdf_isaaclab.py --headless
```

关键配置：
- `fix_base=False`（自由站立机器人）
- `merge_fixed_joints=False`（保留 `left_rubber_hand`、`head_mocap` 等 body 名称）
- `joint_drive` stiffness/damping = 0.0（使用自定义 PD 控制）

### 验证

```bash
# 运行 100 轮测试，确认无挂起
timeout 600 bash -c 'eval "$(conda shell.bash hook)" && conda activate teleopit_isaaclab && \
OMNI_KIT_ACCEPT_EULA=YES python train_mimic/scripts/train.py \
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

## 训练进程被 killed（OOM）

### 症状

训练启动后在场景初始化阶段（Scene manager 创建后）进程被直接终止，终端显示：

```
[INFO]: Scene manager: <mjlab.scene.scene.Scene object at 0x...>
[1]    XXXXX killed     python train_mimic/scripts/train.py ...
```

无 Python 异常堆栈，进程退出码为非零。

### 根因

容器/cgroup 对内存有上限限制（如 100 GiB），而系统已用内存接近该上限。使用 `--num_envs 4096` 时，场景初始化需要大量内存，触发 OOM killer。

诊断命令：

```bash
# 查看 cgroup 内存上限
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
# 查看当前内存使用
free -h
```

若 cgroup 上限（如 107374182400 = 100 GiB）与当前已用内存之差不足几 GB，则可确认是此问题。

### 解决方案

**方案一：减小并行环境数**（推荐，无需额外权限）

```bash
# 先用小规模验证可以跑起来
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 64 --max_iterations 100 \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz

# 确认后再根据可用内存调整到合适规模（如 512 或 1024）
python train_mimic/scripts/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 1024 --max_iterations 30000 \
    --motion_file data/twist2_retarget_npz/OMOMO_g1_GMR/merged.npz
```

**方案二：释放系统 page cache**（需要 root 权限）

```bash
sync && echo 3 > /proc/sys/vm/drop_caches
```

执行后 buff/cache 会释放，再以 `--num_envs 4096` 重新训练。

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
