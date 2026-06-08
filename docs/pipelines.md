# ACT / DP / 4D Map Pipeline 总结

本文档汇总当前仓库中 ACT、Diffusion Policy（DP）以及不同 4D map 接入方式的训练和推理链路。实验结果另见 `docs/results.md`。

## 1. 统一数据与 4D Map 表示

当前主要使用 ManiSkill demonstration HDF5 数据。StackCube 使用 `StackCube-v1`、`pd_ee_delta_pos`、最多 1000 step；PlugCharger 使用 `PlugCharger-v1`、`pd_ee_delta_pose`、最多 400 step。数据采集脚本位于：

- `scripts/data_collection/collect_stackcube.sh`
- `scripts/data_collection/collect_plugcharger.sh`

训练时 policy 读取 RGB、可选 depth、robot state 和 action。启用 `--use-map4d` 后，会额外读取或构造 `map4d` 序列。当前主路径是 `--map4d-source maniskill_gt`，直接从 HDF5 的 `env_states/actors` 读取物体 GT pose，并拼成每帧张量：

```text
map4d frame: [num_objects, 12]
12 = size(3) + position(3) + rotation_6d(6)
```

StackCube 默认 3 个 object：`cubeA`、`cubeB`、`table-workspace`；PlugCharger 默认 2 个 object：`charger`、`receptacle`。

## 2. Diffusion Policy Pipeline

入口文件是 `baselines/diffusion_policy/train_rgbd.py`。baseline 流程：

```text
RGB / RGB-D + state
  -> visual encoder
  -> observation condition
clean action sequence
  -> DDPM add noise
noisy action + timestep + observation condition
  -> ConditionalUnet1D
  -> predict noise
  -> MSE loss
```

推理时从 Gaussian action noise 开始，按 scheduler 逐步 denoise，最后取 `act_horizon` 段动作执行。常用超参是 `obs_horizon=2`、`pred_horizon=16`、`act_horizon=8`、`batch_size=64`、`total_iters=100000`。

视觉 encoder 有两类：

- `plain_conv`：默认轻量 CNN，配置如 `stackcube_pos_dp_baseline.conf`。
- `dinov3_vits16`：DINOv3 视觉特征，配置如 `stackcube_pos_dp_dinov3.conf` 和 `stackcube_pos_dp_dinov3_map4d.conf`。

常用脚本：

```bash
bash scripts/dp_experiments/stackcube/run_baseline_100.sh
bash scripts/dp_experiments/stackcube/run_map4d_raw_aux_100.sh
bash scripts/dp_experiments/plugcharger/run_map4d_raw_1000.sh
```

## 3. ACT Pipeline

RGB-D ACT baseline 入口是 `baselines/act/train_rgbd.py`，4D map 版本入口是 `baselines/act/train_rgbd_map4d.py`。模型核心是 DETR-style CVAE：

```text
RGB / RGB-D + state
  -> ResNet backbone + transformer
action sequence
  -> CVAE encoder
  -> predict num_queries actions
  -> L1 action loss + KL loss * kl_weight
```

训练阶段使用 demonstration 中从当前时刻开始的 `num_queries` 段 action；不足长度会 padding。推理阶段不给 action，模型从 prior 输出 action chunk。常用超参是 `num_queries=30`、`kl_weight=10`、`batch_size=64`、`total_iters=100000`。

常用脚本：

```bash
bash scripts/act_experiments/stackcube/run_baseline_100.sh
bash scripts/act_experiments/stackcube/run_map4d_mlp_token_990.sh
bash scripts/act_experiments/plugcharger/run_map4d_raw_1000.sh
```

## 4. 4D Map 接入变体

### 4.1 Encoder Feature / Context Token

配置特征：`--use-map4d`，不加 raw/token 特殊开关。`map4d` 序列先进入 `Map4d_Encoder`，内部可选：

- `--map4d-encoder-type gru`：默认 `GeometricEncoder`。
- `--map4d-encoder-type transformer`：`GeometricTransformerEncoder`。

DP 中，map feature 与 visual feature、state 拼接后作为 diffusion 全局条件。ACT 中，最后一帧 map feature 作为 `map4d_feature` 注入 DETRVAE 的 transformer memory/context。训练时还会通过 `PhysicsLosses` 加入 pose、penetration、kinematic 辅助约束。

典型配置：

- DP GRU: `baselines/diffusion_policy/configs/stackcube_pos_dp_map4d.conf`
- DP Transformer: `baselines/diffusion_policy/configs/stackcube_pos_dp_map4d_transformer_100demos.conf`
- ACT GRU: `baselines/act/configs/stackcube_pos_act_map4d_100demos.conf`
- ACT Transformer: `baselines/act/configs/stackcube_pos_act_map4d_transformer_100demos.conf`

### 4.2 Raw Concat

配置特征：`--use-map4d --map4d-raw-concat`。该模式跳过 4D map encoder，直接把当前或最近 observation horizon 内的 map4d flatten 后拼到 robot state。

```text
map4d [T, N, 12]
  -> flatten
  -> concat(state)
  -> policy
```

DP raw concat 通常拼接最近 `obs_horizon` 帧；ACT raw concat 取最后一帧拼到 state。PlugCharger 当前主要使用这个变体。

典型配置/脚本：

- DP: `baselines/diffusion_policy/configs/stackcube_pos_dp_map4d_raw_100demos.conf`
- ACT: `baselines/act/configs/stackcube_pos_act_map4d_raw_100demos.conf`
- PlugCharger: `scripts/dp_experiments/plugcharger/run_map4d_raw_1000.sh`

### 4.3 Raw Concat + Auxiliary Future Loss

配置特征：`--map4d-raw-concat --map4d-aux-loss`。主 policy 仍使用 raw concat；额外从 policy feature 或 transformer memory 预测未来 `map4d_future_horizon` 帧物体状态。

监督目标为未来 position delta 和未来 rotation 6D：

```text
target: future_delta_pos(3) + future_rotation_6d(6)
shape: [future_horizon, num_objects, 9]
```

该 loss 用 `--map4d-aux-weight` 加权。StackCube 的 100-demo / 990-demo 实验均有该变体。

### 4.4 Raw Concat + Keyframe Future Loss + TCP Pose

该变体建立在 4.3 的 raw concat + auxiliary future loss 之上，但监督目标不再是连续的未来 `map4d_future_horizon` 帧，而是未来 `map4d_future_horizon` 个关键帧。每个关键帧同时包含物体 4D map 状态和 TCP pose。

```text
raw map4d
  -> concat(state)
  -> policy
policy feature / transformer memory
  -> future keyframe prediction head
  -> predict keyframe object states + TCP pose
```

目标张量可以按关键帧组织：

```text
keyframe target:
  object states: [num_keyframes, num_objects, 9]
    = future_delta_pos(3) + future_rotation_6d(6)
  tcp pose: [num_keyframes, tcp_dim]
```

关键帧提取先尝试 PerAct-style heuristic，而不是等间隔采样。对每条 demonstration 预先生成 keyframe index 序列：

```text
for each frame t:
  keep t if gripper open/close state changes
  keep t if TCP / joint velocity falls below a threshold after meaningful motion
  keep t if t is the final successful frame
deduplicate nearby frames
sort by time
```

训练样本位于当前时刻 `T` 时，只取 `T` 之后的关键帧作为 auxiliary target。关键帧数量由 `map4d_future_horizon` 控制。如果未来不足 `map4d_future_horizon` 个关键帧，则重复最后一个可用关键帧；如果没有未来关键帧，则使用当前帧作为 fallback 并重复到指定长度。这样可以把 auxiliary loss 从“逐帧轨迹预测”改成“未来阶段性状态预测”，同时显式约束末端执行器与物体状态的对应关系。

建议实现时保留 4.3 的原有开关语义，新增独立变体开关，例如：

```text
--map4d-keyframe-aux-loss
```

该变体适合 StackCube / PlugCharger 这类动作阶段明确的任务；关键帧来源可以是 dataset 中已有的阶段标注，或由任务规则、接触事件、成功子目标等离线生成。

### 4.5 ACT Map4D Tokens

配置特征：`--map4d-as-tokens`。该模式只在 ACT 中使用，把 `map4d` 展平为 token 序列：

```text
map4d [B, pre_horizon, N, 12]
  -> [B, pre_horizon * N, 12]
  -> DETRVAE transformer memory
```

对应配置：`baselines/act/configs/stackcube_pos_act_map4d_tokens_100demos.conf`。

### 4.6 ACT MLP Token

配置特征：`--map4d-mlp-token`。该模式只在 ACT 中使用，把 30 帧 map4d flatten 后经过 MLP 压成 1 个 128 维 context token：

```text
map4d [B, 30, N, 12]
  -> flatten
  -> MLP(hidden_dim)
  -> map4d_feature [B, 128]
```

对应配置：`baselines/act/configs/stackcube_pos_act_map4d_mlp_token_100demos.conf`。

## 5. 训练与评估入口

推荐优先使用 `scripts/*_experiments/` 下的封装脚本；它们会进入对应 baseline 目录、source 配置、设置日志路径，并支持部分 `DRY_RUN=1` 检查命令。

```bash
# DP
bash scripts/dp_experiments/stackcube/run_map4d_100.sh
bash scripts/dp_experiments/stackcube/run_map4d_transformer_990.sh
bash scripts/dp_experiments/plugcharger/run_baseline_1000.sh

# ACT
bash scripts/act_experiments/stackcube/run_map4d_raw_aux_100.sh
bash scripts/act_experiments/stackcube/run_map4d_tokens_990.sh
bash scripts/act_experiments/plugcharger/run_baseline_100.sh
```

评估指标会写入 TensorBoard，并在 DP 路径中额外写到 `outputs/eval_metrics/*.jsonl`。主要关注 `success_once` 和 `success_at_end`。

## 6. Smoke Test

DP 侧有 4D map pipeline smoke test：

```bash
python baselines/diffusion_policy/smoke_map4d_pipeline.py
```

它会检查 map4d tensor、数组、RGB 图和可视化产物，输出到 `outputs/map4d_pipeline_smoke/`。
