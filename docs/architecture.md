# 架构与边界

Teleopit 当前按“边界先行”的方式组织：

```text
configs / scripts
    -> runtime
    -> interfaces + pipeline state machines
    -> adapters (inputs / retargeting / controller / robot / recording)

train_mimic/scripts
    -> train_mimic/app.py
    -> train_mimic/tasks
    -> mjlab / rsl_rl

train_mimic/scripts/data
    -> train_mimic/data/dataset_builder.py
    -> dataset_lib / motion_fk / convert_pkl_to_npz
```

## 核心边界

- `teleopit/interfaces.py`
  - 定义稳定协议：`InputProvider`、`Retargeter`、`Controller`、`Robot`、`ObservationBuilder`、`Recorder`。
- `teleopit/runtime/`
  - 负责配置解析、路径归一、组件装配、CLI 公共校验。
  - 这里是运行模式收敛层，不放算法实现。
- `teleopit/pipeline.py`
  - 只保留 offline / online sim 的轻量 runtime façade。
- `teleopit/sim2real/controller.py`
  - 只保留真机状态机与控制行为；mocap/provider/policy/obs 的装配由 `teleopit.runtime` 提供。
- `teleopit/inputs/`、`teleopit/retargeting/`、`teleopit/controllers/`、`teleopit/robots/`、`teleopit/recording/`
  - 作为具体 adapter 层，分别实现对应协议。
- `train_mimic/app.py`
  - 负责训练/播放/benchmark 共用的 task 规范化、校验和训练栈导入。
- `train_mimic/tasks/tracking/config/`
  - 只暴露单一官方训练 task；内部 profile 可保留参考实现，但不直接作为公共注册面。
- `train_mimic/data/dataset_builder.py`
  - 作为训练侧唯一正式数据集构建主线，不再与 legacy review/manifest 流程并存。

## 当前公共面

- 稳定运行模式：
  - offline sim2sim
  - UDP online sim2sim
  - Pico4 sim2sim
  - G1 sim2real
- 稳定配置入口：
  - `teleopit/configs/default.yaml`
  - `teleopit/configs/online.yaml`
  - `teleopit/configs/pico4_sim.yaml`
  - `teleopit/configs/sim2real.yaml`
  - `teleopit/configs/pico4_sim2real.yaml`
- 稳定训练入口：
  - `train_mimic/scripts/train.py`
  - `train_mimic/scripts/play.py`
  - `train_mimic/scripts/benchmark.py`
  - `train_mimic/scripts/save_onnx.py`
  - `train_mimic/scripts/data/build_dataset.py`

## 已收敛的约束

- `controller.policy_path` 必须显式提供且文件必须存在。
- 离线 BVH 运行必须显式提供 `input.bvh_file`。
- `viewers` 是唯一 viewer 配置入口；旧 `viewer` alias 已移除。
- 观测维度与 ONNX 输入维度不匹配时启动即报错。
- sim2real 只支持 154D mjlab ONNX。
