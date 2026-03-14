# 用 `/home/wubingqian/project/teleop_projects/unitree_rl_mjlab` 完全替换 Teleopit 的 G1 模型与参数基线

## Summary

目标是把 Teleopit 里与 G1 policy 执行相关的旧代码、旧参数、旧资产，全部切换到 `/home/wubingqian/project/teleop_projects/unitree_rl_mjlab` 作为唯一真源，并清理不再使用的 Teleopit/GMR G1 默认实现，不保留兼容层。

本次对齐范围包括：

- G1 MuJoCo 模型与资产引用
- 默认站姿、动作缩放、观测零点
- sim2sim 仿真 PD
- sim2real 真机 PD
- 所有入口脚本与配置的默认 G1 参数来源
- 文档中对 G1 默认模型/参数的说明

保留范围只限于 GMR 的人类动作 retarget/IK 逻辑；但其 G1 默认 XML/参数不再参与 policy、观测、sim2sim、sim2real。

## Key Changes

- 统一真源路径到 `unitree_rl_mjlab`：
  - 明确将 `/home/wubingqian/project/teleop_projects/unitree_rl_mjlab` 作为 Teleopit 默认 G1 基线来源。
  - 所有默认 G1 相关参数，均从该仓库读取或镜像同步，不再以 Teleopit 当前配置为主。
  - 在文档中写明当前默认对齐源为：
    - G1 模型：`/home/wubingqian/project/teleop_projects/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml`
    - 训练默认姿态与动作基线：`/home/wubingqian/project/teleop_projects/unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py`
    - 真机 mimic deploy 参数：`/home/wubingqian/project/teleop_projects/unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/dance1_subject2/params/deploy.yaml`

- 替换 G1 模型真源：
  - 将 Teleopit 所有默认 G1 XML 路径切换到 `unitree_rl_mjlab` 的 `g1.xml`。
  - 删除 Teleopit 中作为默认 policy/sim2sim/sim2real 机器人真源的 GMR G1 XML 引用和 fallback 逻辑。
  - 检查并修正所有依赖 body/joint 名称的代码，使其完全按 mjlab G1 模型解析。

- 替换 G1 参数真源：
  - 用 `unitree_rl_mjlab` 的 `HOME_KEYFRAME` 替换 Teleopit 当前 `default_angles`。
  - 用 `unitree_rl_mjlab` 的 `G1_ACTION_SCALE` / deploy `scale` 替换 Teleopit 当前 `action_scale`。
  - 用 `unitree_rl_mjlab` mimic deploy 的 `stiffness/damping` 替换：
    - Teleopit 仿真 `kps/kds`
    - Teleopit sim2real `kp_real/kd_real`
  - 用 mjlab 对齐值替换 `mujoco_default_qpos`。
  - 删除旧的 `KNEES_BENT` 风格默认数组及所有相关引用。

- 重构配置装配：
  - 将 G1 的默认参数集中到单一来源，供 `robot.g1`、sim2sim、sim2real、render、debug 统一读取。
  - 删除当前分散在多个入口中的 G1 特殊覆写和 fallback 注入逻辑。
  - 启动时做硬校验：默认 G1 policy 运行路径只能使用 `/home/wubingqian/project/teleop_projects/unitree_rl_mjlab` 对齐参数，发现不一致直接报错。

- 清理旧代码和旧资产引用：
  - 删除不再使用的 Teleopit G1 默认配置、回退路径、旧 XML 路径引用、旧数组常量。
  - 删除文档中关于旧 GMR G1 默认模型、旧默认姿态、旧真机 PD 的说明。
  - 保留 GMR 中仅用于 retarget/IK 的必要资源；如果某个 GMR G1 资产不再被任何 retarget/IK 路径使用，则一并移除。
  - 不保留 legacy 配置，不提供兼容开关。

- 对齐执行语义：
  - 确保 `joint_pos_rel` 的零点就是 mjlab `default_joint_pos`。
  - 确保 `action -> target_pos` 的 offset 就是 mjlab deploy `offset/default_joint_pos`。
  - 确保 observation builder 的 FK、anchor body、tracking body 命名与 mjlab 训练环境一致。
  - 确保 sim2real 切入低层控制后，standing pose、PD 和 policy 零点属于同一基线。

- 文档更新：
  - README、AGENTS、`docs/inference.md`、`docs/configuration.md`、`docs/sim2real.md` 改为明确说明：
    - Teleopit 默认 G1 模型和控制参数来自 `/home/wubingqian/project/teleop_projects/unitree_rl_mjlab`
    - GMR 仅负责人类动作重定向，不再定义默认 policy robot
  - 删除旧参数说明，避免用户继续沿用错误配置。

## Tests

- 模型与配置测试：
  - 验证 Teleopit 默认 G1 XML、默认姿态、动作缩放、PD 参数全部与 `/home/wubingqian/project/teleop_projects/unitree_rl_mjlab` 基线逐项一致。
  - 验证所有默认入口都不再引用旧 GMR G1 XML 或旧 `default_angles` 数组。

- 语义测试：
  - 零 action 时目标关节角应严格等于 mjlab `default_joint_pos`。
  - `joint_pos == default_joint_pos` 时 `joint_pos_rel` 必须全零。
  - 154D 观测顺序、维度和各切片长度与 mjlab deploy 定义一致。

- 入口回归：
  - `run_sim.py`、`run_sim2real.py`、`render_sim.py` 在默认配置下均能加载 mjlab G1 模型并运行；online/Pico4 路径通过 `--config-name` 切换。
  - sim2sim 首帧姿态为 `HOME_KEYFRAME`。
  - sim2real 初始化打印的默认姿态与 PD 为 mjlab 值。

- 清理验证：
  - 搜索仓库后不应再存在默认执行路径对旧 GMR G1 XML、旧 `KNEES_BENT` 默认姿态、旧真机硬 PD 的引用。
  - 保留的 GMR G1 资产必须都能对应到实际 retarget/IK 用途；无用途的旧资产应移除。

## Assumptions

- 允许直接删除旧 G1 默认实现与参数，不需要兼容旧 ONNX、旧脚本或旧配置行为。
- `unitree_rl_mjlab` 当前 G1 tracking/mimic 主线就是未来 Teleopit 唯一支持的默认基线。
- 若后续要支持别的 G1 policy 变体，应重新引入另一套完整参数包，而不是在当前默认路径里混用。
