# 4D Map Encoder 训练流程

本文档记录当前 4D map encoder 在 DP + 4D map 训练中的实际数据流和网络结构。

## 概览

```
┌─────────────────────────────────────────────────────────────────┐
│ train_rgbd.py :: encode_obs (训练模式)                           │
│                                                                 │
│  obs_seq["map4d"]        (2帧 GT map4d)                         │
│  obs_seq["future_map4d"] (未来3帧 GT map4d, dataloader切出)      │
│         │                                                       │
│         ▼                                                       │
│  map4d_encoder.forward_with_aux(map4d_seq, future_map4d_seq)    │
│         │                                                       │
│         ├──→ map_feature [B, obs_horizon, feature_dim]  → policy│
│         └──→ pred (辅助预测) → physics_losses → auxiliary loss   │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Dataloader 准备输入

`train_rgbd.py` 第 470-480 行:

- `obs_seq["map4d"]`: obs_horizon=2 帧的 4D map tensor(per-object position + rotation_6d + size）
- `obs_seq["future_map4d"]`: 当前帧之后 future_horizon=3 帧的 GT 4D map tensor

## Step 2: GeometricEncoder 编码 obs 窗口

`map4d_encoder.py` 第 66 行 → `geometric_encoder.py :: _encode_sequence_parts`:

```
map4d_seq [B, T=2, N_obj, 12]   (pos:3 + rot_6d:6 + size:3)
  → _parse_representation
    sizes [B, T, N, 3], positions [B, T, N, 3], rotations [B, T, N, 6]

  → node_embed: Linear(12 → node_dim)
    node_feat [B, T, N, node_dim]

  → relation_net: pairwise interaction
    relation_feat [B, T, N, relation_dim]

  → concat → Linear → [B, T, N, temporal_dim]
    → transpose → [B*N, T, temporal_dim]

  → temporal_gru: GRU over time
    temporal_out [B, T, N, temporal_dim]

  → obj_proj: Linear(temporal_dim → feature_dim)
    obj_feat [B, T, N, feature_dim]

  → scene_feat = mean(obj_feat, dim=N)
    scene_feat [B, T, feature_dim]

  → scene_proj: Linear(feature_dim → feature_dim)
    map_feature_seq [B, T, feature_dim]   ← 给 policy fusion 用
```

## Step 3: Future Prediction Head

`map4d_encoder.py` 第 71-92 行 `predict_future_from_features`:

```
输入: obj_feat[:, -1] (最后帧的 per-object feature) [B, N, feature_dim]
  → mean(dim=1)  (objects 维度 mean pool)
    scene_summary [B, feature_dim]

  → future_head: Linear(feature_dim, future_horizon * num_objects * 9)
    pred_flat [B, future_horizon * num_objects * 9]

  → reshape [B, future_horizon=3, num_objects=3, 9]
  → split:
    pred_delta_pos [B, 3, 3, 3]   (未来3帧 × 3物体 × xyz)
    pred_delta_rot [B, 3, 3, 6]   (未来3帧 × 3物体 × rot_6d)

  → 从最后观测帧出发 cumsum 累加:
    pred_pos = positions[:, -1:] + cumsum(pred_delta_pos)
    pred_rot = rotations[:, -1:] + cumsum(pred_delta_rot)   ⚠️ 6D加法bug
```

## Step 4: Physics Losses 计算

`physics_losses.py :: PhysicsLosses.forward(pred)`:

GT 构造 (在 predict_future_from_features 中完成):
```
gt_positions = cat([positions[:, -1:], future_positions[:, :3]])  → [B, 4, N, 3]
gt_rotations = cat([rotations[:, -1:], future_rotations[:, :3]])  → [B, 4, N, 6]
```

4 个 loss:

| Loss | 权重 | 计算 |
|------|------|------|
| pose_loss | 1.0 | L1(pred_delta_pos, gt_delta_pos) + L1(pred_delta_rot, gt_delta_rot) |
| kinematic_loss | 0.1 | 惩罚超出速度/加速度物理限制的 delta |
| penetration_loss | 0.1 | 惩罚预测位姿下物体 AABB 的穿透量 |
| pointcloud_loss | 0.1 | 预测表面采样点与 GT 表面点的距离一致性 |

其中:
```
gt_delta_pos = gt_positions[:, 1:] - gt_positions[:, :-1]   → [B, 3, N, 3]
gt_delta_rot = gt_rotations[:, 1:] - gt_rotations[:, :-1]   → [B, 3, N, 6]  ⚠️ 6D减法bug
```

## Step 5: 与 Diffusion Loss 合并

`train_rgbd.py` 第 665+ 行:

```python
total_loss = diffusion_loss + map4d_total_loss
```

map4d_total_loss = 1.0×pose + 0.1×kinematic + 0.1×penetration + 0.1×pointcloud

## 推理流程

推理时只调用 `map4d_encoder.forward(map4d_seq)`:
- 不走 future prediction head
- 不计算 auxiliary loss
- 只返回 map_feature_seq 给 policy 做 action denoising

## 已知问题

1. **Rotation 6D 加法 bug**: cumsum 和 GT delta 都对 6D rotation 直接加减,无几何意义
2. **obs_horizon 绑定**: map encoder 的输入帧数 = DP obs_horizon = 2,太短无法捕捉运动趋势
3. **future_head 容量**: 单层 Linear 从 1 个 mean-pooled vector 预测 3×3×9=81 维,表达能力弱

详见 `docs/planned_changes.md`。

---

## Encoder 版本对比

### v1: GeometricEncoder (GRU)

文件: `map4d/encoder/geometric_encoder.py`

```
node_feat [B, T, N, node_dim]
  → transpose → [B*N, T, node_dim]
  → GRU(node_dim → temporal_dim)
  → temporal_out [B*N, T, temporal_dim]
  → reshape → [B, T, N, temporal_dim]
  → obj_proj → obj_feat [B, T, N, feature_dim]
  → mean(dim=N) → scene_feat [B, T, feature_dim]
  → scene_proj → map_feature [B, T, feature_dim]
```

特点:
- 每个物体独立做temporal建模（per-object GRU），物体间交互仅通过前端relation MLP
- 参数量: ~166K
- 问题: per-object GRU无法在temporal维度上建模跨物体的联合变化（如cubeA接近cubeB时的协同运动）

### v2: GeometricTransformerEncoder

文件: `map4d/encoder/geometric_transformer_encoder.py`

```
node_feat [B, T, N, node_dim]
  → input_proj → [B, T, N, temporal_dim]
  → + temporal_pos_embed(T) + object_pos_embed(N)
  → reshape → [B, T*N, temporal_dim]
  → TransformerEncoder (num_layers=2, num_heads=4, gelu)
  → reshape → [B, T, N, temporal_dim]
  → obj_proj → obj_feat [B, T, N, feature_dim]
  → mean(dim=N) → scene_feat [B, T, feature_dim]
  → scene_proj → map_feature [B, T, feature_dim]
```

特点:
- 将 T×N 展平为序列，self-attention同时建模空间（物体间）和时间（帧间）关系
- 双重位置编码: learned temporal embedding + learned object identity embedding
- 参数量: ~489K
- 优势: 任意物体在任意时间步之间可以直接attend，捕捉跨物体的时序协同模式

### 共享前端

两个版本共享相同的前端处理:
- node_mlp: 将 (size, pos, rot, vel, acc) 30维 → node_dim
- relation_mlp: 计算 pairwise relation (相对位置、相对旋转、距离) → 聚合到 node_feat
- velocity/acceleration: 通过 time_diff 计算一阶和二阶时间导数

### 共享后端

两个版本共享相同的输出接口:
- `_encode_sequence_parts()` 返回 `(map_feature_seq, obj_feat, scene_feat, sizes, positions, rotations)`
- `map_feature_seq` shape: `[B, T, feature_dim]`
- 在 `Map4d_Encoder` 中可直接替换，无需修改下游代码
