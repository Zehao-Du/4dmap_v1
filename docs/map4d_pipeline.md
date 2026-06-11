# Map4D DiT Pipeline

本文档记录当前 `Map4DDiT` 的训练和推理数据流。这里描述的是新的 Semantic Field + online `Map4DEncoder` 路径，不是旧 DP 里的离线 `map_feature` 路径。

核心约束：

- 数据集存 GT map，不存 map encoder feature。
- `traj_*/map_node_feature`、`obs.map_node_feature`、`obs.map_feature`、`obs.map_graph` 都不是合法输入。
- `Map4DEncoder` 在 `Map4DDiT` 内部在线运行，参数和 DiT 一起训练。
- `obs.dino_feature` 必须是真实 DINO/DINOv3 per-point feature，不能用 RGB placeholder。

## 训练输入

训练使用两个 HDF5 文件：

```text
MAP4D_DEMO_PATH
  ManiSkill demo HDF5

MAP4D_KEYFRAME_SIDECAR_PATH
  final GT-map context sidecar HDF5
```

demo HDF5 提供 Semantic Field 和 robot state：

```text
traj_*/actions
traj_*/obs/agent/*
traj_*/obs/extra/tcp_pose
traj_*/obs/point_cloud/fused      [T, P, 6]
traj_*/obs/dino_feature           [T, P, D_sem]
```

`obs/point_cloud/fused` 的布局是 `xyzrgb`，模型只消费前三维 `xyz`。`obs/dino_feature` 必须与 point cloud 的 `[T, P]` 对齐，并带有 DINO provenance metadata，例如：

```text
semantic_feature_model=dinov3_vits16
feature_type=dinov3
```

final sidecar 提供 GT map 和 keyframe targets：

```text
traj_*/map4d                              [T, N_target_node, 9]
traj_*/size_parameters                    [D_size]
traj_*/relation_parameters                [D_relation]
traj_*/future_keyframe_object_targets     [T, H_key, N_target_node, 9]
traj_*/future_keyframe_tcp_pose           [T, H_key, tcp_dim]
```

`size_parameters` / `relation_parameters` 在 keyframe sidecar 构建阶段从 ManiSkill env 读取。构建脚本要求 demo HDF5 旁边有同名 `.json` metadata，用其中的 `env_info.env_kwargs` 创建同任务 env，并对每个 `traj_*` 使用对应 episode seed/options reset 后读取 `env.unwrapped` 的几何属性。不会从 maps4d JSON default fallback。

注意：HDF5 里为了兼容已有 sidecar，字段仍叫 `traj_*/map4d` 和 `future_keyframe_object_targets`。进入训练 batch 后改名为更明确的接口。

## Dataset 输出

`ManiSkillMap4DDataset.__getitem__` 输出：

```text
sample["obs"]["robot_state"]              [T_obs, robot_state_dim]
sample["obs"]["point_cloud"]              [T_obs, P, 6]
sample["obs"]["dino_feature"]             [T_obs, P, D_sem]
sample["obs"]["node_poses"]               [T_obs, N_target_node, 9]
sample["obs"]["size_parameters"]          [D_size]
sample["obs"]["relation_parameters"]      [D_relation]

sample["action"]["trajectory"]            [H_action, 7]
sample["action"]["gripper_openness"]      [H_action, 1]

sample["keyframe"]["map4d"]               [H_key, N_target_node, 9]
sample["keyframe"]["tcp"]                 [H_key, tcp_dim]
```

batch 后典型 shape：

```text
obs.robot_state             [B, T_obs, robot_state_dim]
obs.point_cloud             [B, T_obs, P, 6]
obs.dino_feature            [B, T_obs, P, D_sem]
obs.node_poses              [B, n_map_step, N_target_node, 9]
obs.size_parameters         [B, D_size]
obs.relation_parameters     [B, D_relation]

action.trajectory           [B, H_action, 7]
action.gripper_openness     [B, H_action, 1]
keyframe.map4d              [B, H_key, N_target_node, 9]
keyframe.tcp                [B, H_key, tcp_dim]
```

当前实现要求 `n_map_step == T_obs`。如果以后让 map history 和 observation history 长度不同，需要在 dataset 里单独切 `node_poses` history。

## 训练流程

训练入口：

```text
map4d/backbone/train_map4d_dit.py
```

训练主循环：

```text
DataLoader
  -> batch
  -> Map4DDiTPolicy.forward(batch)
  -> loss.backward()
  -> optimizer.step()
  -> EMA update
```

`Map4DDiTPolicy.forward` 做四件事：

1. 规范化 obs。
   - 只规范化 `obs.robot_state`。
   - 不规范化 `obs.node_poses`、`size_parameters`、`relation_parameters`，因为 online `Map4DEncoder` 需要 GT 几何量。

2. 规范化 targets。
   - `action.trajectory[..., 0:3]` 用 `trajectory_pos` normalizer。
   - quaternion 规范到单位四元数。
   - `keyframe.map4d[..., 0:3]` 用 `keyframe_map4d_pos` normalizer。
   - `keyframe.tcp[..., 0:3]` 用 `keyframe_tcp_pos` normalizer。
   - gripper 用 `gripper_openness` normalizer。

3. 加 diffusion noise。
   - trajectory delta pose 加 position/rotation noise。
   - keyframe map4d pose 加 position/rot6d noise。
   - keyframe tcp 加 position/rotation 或 position/gripper noise。

4. 调 `Map4DDiT(noisy_targets, timestep, obs)` 预测 noise 和 gripper。

训练 loss：

```text
total_loss =
  trajectory_loss_weight    * L1(pred.trajectory, noise.trajectory)
  + keyframe_tcp_loss_weight  * L1(pred.keyframe_tcp, noise.keyframe_tcp)
  + keyframe_map4d_loss_weight * L1(pred.keyframe_map4d, noise.keyframe_map4d)
  + gripper_loss_weight       * L1(pred.gripper_openness, target.gripper_openness)
```

当前 metric key：

```text
trajectory_noise_l1
keyframe_tcp_noise_l1
keyframe_map4d_noise_l1
gripper_l1
bc_loss
```

## Map4DDiT 内部流程

### 1. Semantic Field context

输入：

```text
obs.point_cloud       [B, T_obs, P, 6]
obs.dino_feature      [B, T_obs, P, D_sem]
```

处理：

```text
xyz = obs.point_cloud[..., :3]
semantic_input = concat(xyz, dino_feature)
semantic_token = semantic_field_proj(semantic_input)
semantic_token += semantic_source_embed
```

flatten 后：

```text
semantic_xyz      [B, T_obs * P, 3]
semantic_token    [B, T_obs * P, D_model]
```

### 2. Online map context

输入：

```text
obs.node_poses              [B, n_map_step, N_target_node, 9]
obs.size_parameters         [B, D_size]
obs.relation_parameters     [B, D_relation]
```

`Map4DDiT` 先把历史 map 展平到 `B * n_map_step`：

```text
positions = node_poses[..., 0:3]
rotations = node_poses[..., 3:9]
```

然后根据任务构造 maps4d representation：

```text
StackCube-v1:
  Map4d_StackCube(size_parameters, positions, rotations)

PlugCharger-v1:
  Map4d_PlugCharger(positions, rotations, size_parameters, relation_parameters)
```

再在线调用：

```text
encoded_map_nodes = Map4DEncoder(map_representation)
```

输出必须是：

```text
[B * n_map_step, N_map_node, 3 + D_map]
```

reshape 成：

```text
encoded_map_nodes    [B, n_map_step, N_map_node, 3 + D_map]
map_xyz              [B, n_map_step, N_map_node, 3]
map_feature          [B, n_map_step, N_map_node, D_map]
```

map token：

```text
map_token = map_feature_proj(map_feature)
map_token += map_time_pos_embed
map_token += map_node_pos_embed
map_token += map_source_embed
```

flatten 后：

```text
map_xyz       [B, n_map_step * N_map_node, 3]
map_token     [B, n_map_step * N_map_node, D_model]
```

### 3. Unified 3D context

Semantic Field tokens 和 map tokens 合并成一个只读 3D context field：

```text
context_xyz =
  concat(semantic_xyz, map_xyz)
  [B, T_obs * P + n_map_step * N_map_node, 3]

context_token =
  concat(semantic_token, map_token)
  [B, T_obs * P + n_map_step * N_map_node, D_model]
```

这些 context tokens 只作为 cross-attention 的 key/value，不被 denoise，也不被 target branch 更新。

### 4. Staged PPI-style attention

`Map4DDiT` 不把所有 token 拼成一个大 self-attention 序列，而是使用 PPI-style staged dependency：

1. keyframe map4d branch。
   - query: noisy `keyframe_map4d`
   - key/value: unified 3D context
   - 输出 `pred.keyframe_map4d`

2. keyframe tcp branch。
   - query: noisy `keyframe_tcp`
   - key/value: unified 3D context + keyframe map4d hidden features
   - 输出 `pred.keyframe_tcp`

3. action branch。
   - query: noisy `trajectory`
   - key/value: unified 3D context + keyframe map4d hidden + keyframe tcp hidden
   - 输出 `pred.trajectory` 和 `pred.gripper_openness`

默认 downstream branch 使用 upstream hidden feature 的 `detach()`，对应 PPI 的稳定训练设计。

## 推理流程

推理入口：

```text
Map4DDiTPolicy.predict_action(obs_dict)
```

`obs_dict` 必须提供和训练一致的字段：

```text
obs_dict["robot_state"]              [B, T_obs, robot_state_dim]
obs_dict["point_cloud"]              [B, T_obs, P, 6]
obs_dict["dino_feature"]             [B, T_obs, P, D_sem]
obs_dict["node_poses"]               [B, n_map_step, N_target_node, 9]
obs_dict["size_parameters"]          [B, D_size]
obs_dict["relation_parameters"]      [B, D_relation]
```

推理时不需要 keyframe GT target。policy 会初始化 diffusion latent：

```text
sample["trajectory"]       random noise
sample["keyframe_map4d"]   random noise
sample["keyframe_tcp"]     random noise
```

然后按 scheduler timestep 反复调用：

```text
pred = Map4DDiT(sample, timestep, obs)
sample = scheduler.step(pred, sample)
```

反向 diffusion 结束后：

```text
trajectory_pred = unnormalize(sample["trajectory"])
gripper_pred = unnormalize(latest_pred["gripper_openness"])
action = concat(trajectory_pred, gripper_pred)
```

只有 `action[:, :n_action_steps]` 会发给控制器。`keyframe_map4d` 和 `keyframe_tcp` 是 latent plan，不直接执行。

## Rollout 时需要在线构造的观测

rollout/eval 环境不能使用训练 sidecar target，但必须在线构造和训练一致的 obs：

1. `robot_state`
   - 从 env obs 的 `agent/qpos`、`agent/qvel`、`extra/tcp_pose`、`extra/tcp_pos` 拼接并 pad/truncate。

2. `point_cloud`
   - 从当前 RGB-D、多相机内外参反投影得到 fused point cloud。
   - shape 必须是 `[B, T_obs, P, 6]`。

3. `dino_feature`
   - 必须由 DINO/DINOv3 对当前观测生成，并映射/对齐到 fused point cloud 的每个点。
   - shape 必须是 `[B, T_obs, P, D_sem]`。
   - 不能用 RGB 值、image-level pooled feature、zero tensor、random tensor 代替。

4. `node_poses`
   - 从环境状态或在线 Map4D construction 得到每个 target node 的 pose。
   - layout: `pos(3) + rot6d(6)`。

5. `size_parameters` / `relation_parameters`
   - 从 ManiSkill env/reset 读取，不从 maps4d task metadata default 补齐。
   - 即使 `D_relation=0`，也要传空 tensor `[B, 0]`，不能缺字段。

当前需要特别注意：如果 rollout 代码仍然传旧字段 `obs.map4d` 或 `obs.rgb_feature`，它不是新的 `Map4DDiT` 输入接口，需要改成 `obs.node_poses`、`obs.point_cloud`、`obs.dino_feature`。

## 数据构建流程

一键脚本：

```bash
bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh <task> <demos> [resolution]
```

当前步骤：

1. 收集 ManiSkill RGB-D demo。
2. 构造 fused point cloud：`traj_*/obs/point_cloud/fused`。
   - 同时写入 `traj_*/obs/point_cloud_source/fused/{camera_index,pixel_uv}`，用于 DINO patch token 到点云采样点的对齐。
3. 生成或校验 per-point DINO feature：`traj_*/obs/dino_feature`。
   - 默认 `SEMANTIC_FEATURE_MODE=dinov3`，加载一次冻结 DINOv3 backbone，从 RGB patch tokens 生成 `[T, P, D_sem]`。
   - 默认 `SEMANTIC_FEATURE_DTYPE=float32`；可以设为 `float16` 来减少 HDF5 写入量，训练读取时会转回 `float32`。
   - `SEMANTIC_FEATURE_MODE=existing_dino` 只用于显式复用已经存在的 per-point DINO feature。
   - 不生成 fallback RGB feature。
4. 构造 keyframe sidecar。
5. 构造 final GT-map context sidecar。
   - 复制 keyframe sidecar。
   - 校验 demo HDF5 中存在 point cloud 和 DINO feature。
   - 拒绝 `traj_*/map_node_feature`。

## 禁止的旧接口

以下字段不应出现在新训练/推理输入里：

```text
obs.map4d                  # 已改名为 obs.node_poses
obs.rgb_feature            # image-level feature，不是 per-point Semantic Field
obs.map_node_feature       # 离线 map encoder feature
obs.map_feature            # 离线 map encoder feature
obs.map_graph
obs.map_graph_seq
traj_*/map_node_feature
```

如果出现这些字段，应该直接报错，而不是 fallback。
