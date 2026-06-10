---
name: map4d-horizon-decouple
description: Planned change to decouple 4D map encoder's temporal horizon from DP obs_horizon
metadata:
  type: project
---

## Map4D Encoder Horizon 解耦

**问题**: 当前 4D map encoder 的输入帧数绑定了 DP 的 `obs_horizon`(=2)。2 帧输入几乎没有运动趋势信息(只有一个 delta),很难预测未来 3 帧的物体运动,导致辅助 loss 信号接近噪声。

**决定**: 将 4D map encoder 的输入时序窗口与 DP 的 `obs_horizon` 解耦。

- DP 保持 `obs_horizon=2`(控制 action 生成的条件窗口,短一点 fine)
- 4D map encoder 单独设置更长的 `map4d_pre_horizon`(如 4~8 帧),让 encoder 能看到加速/减速趋势
- 需要改动:dataloader 加载更长的 map4d 历史窗口;encoder 输入逻辑独立于 obs_horizon

**Why:** 从第一轮实验(stackcube_pos 4 组对比)看到 DINOv3+Map4D 的辅助 loss 可能干扰主 diffusion loss,部分原因是 2 帧输入下 future prediction 太难、loss 信号质量差。

**How to apply:** 等第一轮 4 组实验全部跑完(400k iter)后再动手改。改动涉及 `train_rgbd.py` 的 dataloader、`Map4d_Encoder` 的输入接口、以及新增 CLI 参数 `--map4d-pre-horizon`。

**状态**: 待实施(等 DINOv3+Map4D 400k 实验结果确认后)

---

## Pose Loss Rotation 计算 Bug

**问题**: `physics_losses.py` 第 78 行对 6D rotation 表示直接做减法:
```python
gt_delta_rot = rotations[:, 1:] - rotations[:, :-1]
```
6D rotation 是旋转矩阵前两列 flatten 成 6 维向量,直接做差没有几何意义(不等价于旋转的差)。同理 `pred_delta_rot` 的监督目标也是错的。

**正确做法**:
1. 6D → 旋转矩阵: `R1 = rotation_6d_to_matrix(rot_t)`, `R2 = rotation_6d_to_matrix(rot_{t+1})`
2. 计算相对旋转: `R_delta = R2 @ R1^T`
3. Loss 可以用:
   - Frobenius norm: `||R_delta_pred - R_delta_gt||_F`
   - 或 geodesic distance: `arccos((tr(R_delta_pred^T @ R_delta_gt) - 1) / 2)`

**影响文件**: `map4d/encoder/physics_losses.py` 第 78、80-81 行,以及 `map4d/encoder/geometric_encoder.py` 的 `predict_pose_deltas` 和 `pred_head` 输出格式。

**Why:** 当前 loss 对旋转的监督信号有误,可能导致 map4d feature 学到的旋转动态不正确,间接影响 policy 性能。

**状态**: 待修复

---

## ~~Future Prediction 只预测了 1 步而非 N 步~~ (已澄清,非 bug)

**澄清**: 训练时走的是 `map4d_encoder.py` 的 `forward_with_aux` → `predict_future_from_features` 路径,**确实预测了 future_horizon=3 帧**。dataloader 会切出未来 3 帧 GT map4d 传入。

`geometric_encoder.py` 的 `predict_pose_deltas`(obs 窗口内 1 步预测)只在 `future_map4d_seq=None` 时的 fallback 路径才会用到,正常训练不走这条路。

**实际的 future_head 实现问题**:
- 网络是单层 `nn.Linear(feature_dim, future_horizon * num_objects * 9)`,从最后帧 obj_feat 的 mean pool 一次性输出所有未来帧所有物体的 delta — 表达能力弱
- Rotation 部分仍然有 6D cumsum 加法的 bug(同第 2 条)

**状态**: 不需要修复路径问题;rotation bug 和网络容量问题待改进

---

## Future Head 改为 Per-Object MLP

**问题**: 当前 `future_head` 是一个 Linear,从 mean-pooled scene feature 一次性输出所有物体所有未来帧的 delta。不同物体的运动模式差异很大(如 cubeA 被抓取移动,cubeB 静止,table 永远不动),一个共享 Linear 难以同时学好所有物体。

**改动**: 将 `future_head` 替换为 `nn.ModuleList`,每个物体一个独立 MLP:

```python
self.future_heads = nn.ModuleList([
    nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, future_horizon * 9),
    )
    for _ in range(num_objects)
])
```

每个 MLP 以**该物体最后帧的 obj_feat**(而非所有物体 mean pool)为输入,输出该物体未来 `future_horizon` 帧的 `[delta_pos(3), delta_rot(6)]`。

**优势**:
- 每个物体的预测头独立,能学到不同运动模式
- 输入从 mean-pooled scene feature 改为 per-object feature,信息更精准
- 参数量适中:3 个 MLP × (128×128 + 128×27) ≈ 60k params

**状态**: 待实施

---

## 去掉 Pointcloud Loss

**问题**: StackCube 任务目标是把 cubeA 堆叠到 cubeB 上面,成功状态下两个方块表面距离为 0。Pointcloud loss 惩罚表面点距离 < margin=0.002 的情况,等于在惩罚正确的目标状态(贴合)。

**修改方案**: 去掉 pointcloud loss,保留 penetration loss(防止穿透仍然有意义)、pose loss 和 kinematic loss。

**状态**: 待实施
