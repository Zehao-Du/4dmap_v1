# RLBench2 Map4D DiT Algorithm Pipeline

本文档描述 RLBench2 `bimanual_push_box` 上当前 Map4D DiT 的算法流程，而不是运行命令。

核心问题是：普通点云 token 数量远大于 Map4D node token 数量。旧 DiT 将二者直接拼接后做 cross-attention，导致 map/node token 容易被视觉点云淹没。新 pipeline 保留旧架构作为默认选项，同时新增独立 map cross-attention 分支，让动作预测显式读取 Map4D 结构 token。

## 1. 输入表示

每个训练样本包含三类信息：

```text
observation: 当前和历史观测
action:      连续动作轨迹监督
keyframe:    Map4D node 和 TCP 的关键帧监督
```

对 RLBench2 push box，主要 tensor 为：

```text
obs.robot_state            [B, Tobs, 16]
obs.point_cloud            [B, Tobs, P, 6]
obs.dino_feature           [B, Tobs, P, 288]
obs.node_position          [B, Tobs, N, 3]
obs.node_rotation          [B, Tobs, N, 4]
obs.size_parameters        [B, 3]
obs.relation_parameters    [B, 0]

action.trajectory          [B, A, Hact, 7]
action.gripper_openness    [B, A, Hact, 1]

keyframe.map4d             [B, Hkey, N, 7]
keyframe.tcp               [B, A, Hkey, 7]
```

当前默认超参数：

```text
Tobs = 2
P = 6144
N = 1
A = 2   # two arms: right arm, left arm
Hact = 50
Hkey = 4
```

其中：

```text
point_cloud[..., 0:3] 是点坐标
dino_feature 是每个点的语义特征
node_position/node_rotation/size_parameters 是 Map4D 结构输入
obs.robot_state 是双臂状态，raw 维度为 16
```

RLBench2 raw action 也是双臂 16 维：

```text
raw action [16] =
  right_xyz(3) + right_quat_xyzw(4) + right_gripper(1)
+ left_xyz(3)  + left_quat_xyzw(4)  + left_gripper(1)
```

进入 dataset 后会转成模型使用的结构化双臂格式：

```text
action.trajectory[:, 0]       = right_xyz + right_quat_wxyz
action.gripper_openness[:, 0] = right_gripper

action.trajectory[:, 1]       = left_xyz + left_quat_wxyz
action.gripper_openness[:, 1] = left_gripper
```

## 2. Observation Encoder

点云和 DINO 特征先进入 `ObservationEncoder`：

```text
point_cloud + dino_feature + robot_state
        |
        v
ObservationEncoder
        |
        +-- semantic_xyz
        +-- semantic_token
        +-- sampled_semantic_xyz
        +-- sampled_semantic_token
        +-- state_feat
```

输入在进入 encoder 前会先把 batch 和观测时间维展平：

```text
point_cloud:   [B, Tobs, P, 6]   -> [B * Tobs, P, 3]
dino_feature:  [B, Tobs, P, 288] -> [B * Tobs, P, 288]
robot_state:   [B, Tobs, 16]     -> [B * Tobs, 16]
```

其中 `point_cloud` 只把前 3 维 xyz 送入几何分支；rgb 目前不直接作为点特征参与 `Map4DDiT` 的 context 构造。每个点的语义来自预计算的 `dino_feature`。

### 2.1 Semantic Point Tokens

`ObservationEncoder` 会把每个点的几何坐标和 DINO 语义特征编码成 point-wise semantic token：

```text
point input per point = xyz(3) + dino_feature(288)
                     = 291 dims

semantic_token dim = C = 240
```

输出再 reshape 回 batch 维：

```text
semantic_xyz/token:         [B, Tobs * P, 3 / C]
```

当前 push box 配置中：

```text
Tobs * P = 2 * 6144 = 12288 tokens
C = 240
```

这一路 token 是后续 cross-attention 的主视觉语义上下文。旧架构里它会和 map token 直接 concat；新架构里它只作为 semantic context，被 map cross-attention 之后的 query 再读取。

### 2.2 Sampled Semantic Tokens

除了完整 point-wise semantic tokens，`ObservationEncoder` 还会通过 pointcloud encoder 产生下采样后的 sampled tokens：

```text
sampled_semantic_xyz/token: [B, Tobs * Ps, 3 / C]
```

其中 `Ps` 是每帧采样后的点数。当前默认 pointcloud encoder 配置是：

```text
in_channels = 291
out_channels = 240
npoint1 = 1024
npoint2 = 512
```

因此 sampled semantic token 约为：

```text
Tobs * Ps = 2 * 512 = 1024 tokens
```

这一路 token 不直接作为第一层 cross-attention 的 full context，而是在每个 stage 的 self-attention planning 阶段被拼进去：

```text
sampled_context = sampled_semantic_tokens + map_tokens
```

它的作用是提供一个更小、更适合 self-attention 的场景摘要，避免在 self-attention 中引入 12288 个完整点 token。

### 2.3 Robot State Token

`robot_state` 是双臂 16 维状态：

```text
robot_state [16] =
  right_state(8) + left_state(8)
```

在当前 RLBench2 数据中，它和 raw action 的顺序一致，都是 right 在前、left 在后。`ObservationEncoder` 会把每一帧 robot state 编成：

```text
state_feat: [B, Tobs, C]
```

`state_feat` 不作为普通 context token 拼到点云里，而是用于构造 diffusion timestep condition：

```text
time_feat = time_encoder(diffusion_timestep)
cond = time_feat + mean(state_feat over Tobs)
```

这个 `cond` 会作为 AdaLN / diffusion conditioning 输入到后续 cross-attention 和 self-attention 模块中。

### 2.4 Encoder 输出如何进入 DiT

`ObservationEncoder` 输出后，Map4D DiT 内部有两套视觉上下文：

```text
semantic context:
  semantic_xyz
  semantic_token

sampled semantic context:
  sampled_semantic_xyz
  sampled_semantic_token
```

旧架构：

```text
context_token = concat(semantic_token, map_token)
sampled_context_token = concat(sampled_semantic_token, map_token)
```

新架构：

```text
map_token      -> independent map cross-attention
semantic_token -> semantic cross-attention
sampled_semantic_token + map_token -> stage self-attention
```

也就是说，完整 semantic tokens 负责高分辨率视觉读取，sampled semantic tokens 负责紧凑的 planning context，robot state 负责 diffusion condition。

## 3. Map4D Encoder

Map4D 结构输入不从点云中隐式学习，而是由 GT pose 和 task representation 显式构造：

```text
node_position
node_rotation
size_parameters
relation_parameters
        |
        v
Map4d_RLBench2PushBox
        |
        v
Map4DEncoder
        |
        v
encoded_map_nodes: [B, Tobs, N, 3 + Dmap]
```

随后拆成：

```text
map_xyz   = encoded_map_nodes[..., 0:3]
map_feat  = encoded_map_nodes[..., 3:]
map_token = map_feature_proj(map_feat)
```

最终 map token 形状：

```text
map_xyz/token: [B, Tobs * N, 3 / C]
```

对 push box：

```text
Tobs * N = 2 * 1 = 2 map tokens
```

这就是旧架构的主要问题：`2` 个 map tokens 和 `12288` 个 semantic point tokens 在同一个 cross-attention context 里竞争。

## 4. DiT Denoiser 输入

`Map4DDiTPolicy.forward(batch)` 先把 dataset batch 拆成三部分：

```text
obs:
  robot_state
  point_cloud
  dino_feature
  node_position
  node_rotation
  size_parameters
  relation_parameters

action:
  trajectory
  gripper_openness

keyframe:
  map4d
  tcp
```

进入 DiT 之前，policy 会做 normalization 和 diffusion 加噪。对 push box，关键 shape 是：

```text
normalized obs.robot_state            [B, Tobs=2, 16]
normalized obs.point_cloud            [B, Tobs=2, P=6144, 6]
normalized obs.dino_feature           [B, Tobs=2, P=6144, 288]
normalized obs.node_position          [B, Tobs=2, N=1, 3]
normalized obs.node_rotation          [B, Tobs=2, N=1, 4]

normalized trajectory target          [B, A=2, Hact=50, 7]
normalized gripper target             [B, A=2, Hact=50, 1]
normalized keyframe TCP target        [B, A=2, Hkey=4, 7]
normalized keyframe Map4D target      [B, Hkey=4, N=1, 7]
```

DiT 本体的 forward 输入不是完整 batch，而是：

```text
noisy_targets = {
  trajectory:   noisy trajectory,   [B, A, Hact, 7]
  keyframe_tcp: noisy keyframe TCP, [B, A, Hkey, 7]
}

timestep: [B]
obs: normalized observation dict
```

注意：

```text
keyframe.map4d 不作为 noisy target 输入。
keyframe.map4d[..., 0:3] 只作为 keyframe_node_position 的监督信号。
```

也就是说，DiT 同时完成三个预测任务：

```text
1. 给 continuous trajectory 去噪
2. 给 keyframe TCP 去噪
3. 预测 Map4D node keyframe xyz
```

## 5. Diffusion 加噪流程

训练时 policy 随机采样 diffusion timestep：

```text
timesteps: [B]
```

然后分别给 trajectory 和 keyframe TCP 加噪：

```text
trajectory_pos = add_position_noise(trajectory[..., 0:3])
trajectory_rot = add_rotation_noise(trajectory[..., 3:7])

tcp_pos = add_position_noise(keyframe_tcp[..., 0:3])
tcp_rot = add_rotation_noise(keyframe_tcp[..., 3:7])
```

对当前 7 维 pose：

```text
trajectory[..., 0:3] 是 xyz
trajectory[..., 3:7] 是 quat_wxyz
keyframe_tcp[..., 0:3] 是 xyz
keyframe_tcp[..., 3:7] 是 quat_wxyz
```

gripper 不作为 diffusion sample 的一部分，而是由 `gripper_head` 从 trajectory feature 直接预测：

```text
gripper_openness target: [B, A, Hact, 1]
gripper prediction:      [B, A, Hact, 1]
```

当前 condition mask 也支持双臂 action：

```text
condition_mask: [B, A, Hact, 7]
```

实现上会临时 flatten arm 维：

```text
[B, A, Hact, 7] -> [B * A, Hact, 7] -> mask -> [B, A, Hact, 7]
```

## 6. DiT 内部输入编码

`Map4DDiT.forward(noisy_targets, timestep, obs)` 内部先准备三类 context 和三类 target query。

### 6.1 Semantic Context

来自 `ObservationEncoder`：

```text
semantic_xyz       [B, Tobs * P, 3]   = [B, 12288, 3]
semantic_token     [B, Tobs * P, C]   = [B, 12288, 240]

sampled_semantic_xyz    [B, Tobs * Ps, 3] = [B, 1024, 3]
sampled_semantic_token  [B, Tobs * Ps, C] = [B, 1024, 240]

state_feat         [B, Tobs, C]       = [B, 2, 240]
```

### 6.2 Map Context

来自 `Map4DEncoder`：

```text
encoded_map_nodes [B, Tobs, N, 3 + Dmap]
map_xyz           [B, Tobs * N, 3] = [B, 2, 3]
map_token         [B, Tobs * N, C] = [B, 2, 240]
```

其中 `map_token` 通过 `map_feature_proj` 投影到 DiT hidden dim：

```text
map_feature [B, Tobs, N, 240]
  -> LayerNorm
  -> Linear
  -> SiLU
  -> Linear
  -> LayerNorm
  -> map_token [B, Tobs * N, 240]
```

### 6.3 Diffusion Condition

DiT 把 timestep 和 robot state 合成一个 condition 向量：

```text
time_feat = time_encoder(timestep)      [B, C]
state_ctx = mean(state_feat over Tobs)  [B, C]
cond = time_feat + state_ctx            [B, C]
```

这个 `cond` 会传入每个 cross-attention 和 self-attention block，用于 AdaLN / diffusion timestep conditioning。

## 7. Target Query Tokenization

DiT 构造三类 query token：trajectory、Map4D node keyframe、TCP keyframe。

### 7.1 Trajectory Tokens

输入：

```text
noisy trajectory: [B, A, Hact, 7]
```

投影：

```text
trajectory_proj(7 -> C)
```

加 embedding：

```text
trajectory_token =
  trajectory_proj(noisy_trajectory)
  + trajectory_pos_embed      # action horizon position
  + arm_embed                 # right/left arm id
  + target_type_embed[action]
```

输出：

```text
traj_tokens: [B, A * Hact, C]
traj_xyz:    [B, A * Hact, 3]
```

对 push box：

```text
traj_tokens: [B, 2 * 50, 240] = [B, 100, 240]
traj_xyz:    [B, 100, 3]
```

### 7.2 Node Keyframe Tokens

node query 不是从 noisy target 来的，而是从当前观测里的 node position 构造：

```text
current_node_position = obs.node_position[:, -1]  [B, N, 3]
keyframe_node_query = repeat over Hkey            [B, Hkey, N, 3]
```

投影和 embedding：

```text
node_token =
  node_proj(current node xyz)
  + keyframe_time_embed
  + node_id_embed
  + target_type_embed[node]
```

输出：

```text
node_tokens: [B, Hkey * N, C]
node_xyz:    [B, Hkey * N, 3]
```

对 push box：

```text
node_tokens: [B, 4 * 1, 240] = [B, 4, 240]
node_xyz:    [B, 4, 3]
```

### 7.3 TCP Keyframe Tokens

输入：

```text
noisy keyframe_tcp: [B, A, Hkey, 7]
```

投影和 embedding：

```text
tcp_token =
  tcp_proj(noisy_keyframe_tcp)
  + tcp_pos_embed
  + arm_embed
  + target_type_embed[tcp]
```

输出：

```text
tcp_tokens: [B, A * Hkey, C]
tcp_xyz:    [B, A * Hkey, 3]
```

对 push box：

```text
tcp_tokens: [B, 2 * 4, 240] = [B, 8, 240]
tcp_xyz:    [B, 8, 3]
```

所有 query 和 context 都带 3D 坐标，attention 里用 `RotaryPositionEncoding3D` 提供相对 3D 位置信息：

```text
query_pos = rotary_3d(query_xyz)
value_pos = rotary_3d(context_xyz)
```

## 8. DiT 三阶段预测流程

Map4D DiT 是 staged denoiser，不是把所有 query 一次性扔进同一个 transformer。它按下面顺序预测：

```text
stage 1: Map4D node keyframe
stage 2: TCP keyframe
stage 3: continuous trajectory/action
```

这个顺序体现了结构先验：

```text
先规划物体/node 的未来关键位置
再规划双臂 TCP keyframe
最后规划连续动作轨迹
```

### 8.1 Stage 1: Node Keyframe

旧架构中：

```text
node_tokens
  -> node_context_attn(context_token = semantic_token + map_token)
  -> concat(node_feat, sampled_semantic_token + map_token)
  -> node_self_attn
  -> node_feat
```

新 `separate_map_cross_attn=true` 中：

```text
node_tokens
  -> node_map_attn(map_token)
  -> node_context_attn(semantic_token)
  -> concat(node_feat, sampled_semantic_token + map_token)
  -> node_self_attn
  -> node_feat
```

输出：

```text
node_feat: [B, Hkey * N, C]
```

### 8.2 Stage 2: TCP Keyframe

TCP stage 会读取 node stage 的 planning result：

```text
node_condition = node_plan.detach() if detach_stage_features else node_plan
```

旧架构：

```text
tcp_tokens
  -> tcp_context_attn(context_token = semantic_token + map_token)
  -> concat(tcp_feat, node_condition)
  -> tcp_self_attn
  -> tcp_feat
```

新架构：

```text
tcp_tokens
  -> tcp_map_attn(map_token)
  -> tcp_context_attn(semantic_token)
  -> concat(tcp_feat, node_condition)
  -> tcp_self_attn
  -> tcp_feat
```

输出：

```text
tcp_feat: [B, A * Hkey, C]
```

### 8.3 Stage 3: Continuous Trajectory

trajectory/action stage 会读取 TCP stage 的 planning result：

```text
tcp_condition = tcp_plan.detach() if detach_stage_features else tcp_plan
```

旧架构：

```text
traj_tokens
  -> action_context_attn(context_token = semantic_token + map_token)
  -> concat(traj_feat, tcp_condition)
  -> action_self_attn
  -> traj_feat
```

新架构：

```text
traj_tokens
  -> action_map_attn(map_token)
  -> action_context_attn(semantic_token)
  -> concat(traj_feat, tcp_condition)
  -> action_self_attn
  -> traj_feat
```

输出：

```text
traj_feat: [B, A * Hact, C]
```

## 9. 旧架构与新架构差异

旧架构中，semantic tokens 和 map tokens 被直接拼接：

```text
context_xyz   = concat(semantic_xyz, map_xyz)
context_token = concat(semantic_token, map_token)
```

然后所有 stage 都从同一个 context 里 cross-attend：

```text
query -> cross_attn([semantic tokens + map tokens])
```

问题是 push box 里 token 数量极不平衡：

```text
semantic tokens: 2 * 6144 = 12288
map tokens:      2 * 1    = 2
ratio:           1 : 6144
```

新架构由下面开关控制：

```yaml
policy.model_cfg.separate_map_cross_attn: true
```

启用后，每个 stage 都先单独读取 map tokens，再读取 semantic tokens：

```text
query -> map_cross_attn(map_token) -> semantic_cross_attn(semantic_token)
```

因此 map tokens 不再和上万个视觉点 token 在同一个 attention softmax 中竞争。旧架构仍然是默认值：

```yaml
policy.model_cfg.separate_map_cross_attn: false
```

这样已有旧架构训练和 checkpoint 不会被默认破坏。

## 10. Prediction Heads

三阶段 attention 输出后，会先经过 timestep/state condition 调制：

```text
shift, scale = final_modulation(cond)

node_feat = final_norm(node_feat) * (1 + scale) + shift
tcp_feat  = final_norm(tcp_feat)  * (1 + scale) + shift
traj_feat = final_norm(traj_feat) * (1 + scale) + shift
```

然后进入各自输出头：

```text
node_feat -> node_head       -> keyframe_node_position
tcp_feat  -> tcp_head        -> keyframe_tcp
traj_feat -> trajectory_head -> trajectory
traj_feat -> gripper_head    -> gripper_openness
```

输出 shape：

```text
trajectory                 [B, A, Hact, 7]
keyframe_node_position     [B, Hkey, N, 3]
keyframe_tcp               [B, A, Hkey, 7]
gripper_openness           [B, A, Hact, 1]
```

对 push box：

```text
trajectory                 [B, 2, 50, 7]
keyframe_node_position     [B, 4, 1, 3]
keyframe_tcp               [B, 2, 4, 7]
gripper_openness           [B, 2, 50, 1]
```

注意：

```text
keyframe_node_position 当前只预测 node xyz。
keyframe.map4d 里的 node rotation 目前没有作为输出监督。
```

## 11. DiT Forward 伪代码

下面伪代码对应当前实现的两层调用：

```text
Map4DDiTPolicy.forward(batch)
  -> normalize / add noise / compute loss

Map4DDiT.forward(noisy_targets, timestep, obs)
  -> encode context / run staged attention / predict outputs
```

伪代码重点表达 RLBench2 双臂路径；单臂输入的 squeeze/unsqueeze 兼容逻辑在这里省略。

### 11.1 Training Step

```python
def policy_forward(batch):
    # ----- normalize observations -----
    nobs = normalizer.normalize(batch["obs"])

    # ----- normalize supervised targets -----
    ntrajectory = normalize_trajectory(batch["action"]["trajectory"])
    # [B, A, Hact, 7]

    ngripper = normalizer["gripper_openness"].normalize(batch["action"]["gripper_openness"])
    # [B, A, Hact, 1]

    nkeyframe_tcp = normalize_keyframe_tcp(batch["keyframe"]["tcp"])
    # [B, A, Hkey, 7]

    nkeyframe_node_position = normalizer["keyframe_map4d_pos"].normalize(
        batch["keyframe"]["map4d"][..., 0:3]
    )
    # [B, Hkey, N, 3]

    # ----- sample diffusion timestep -----
    timesteps = randint(0, num_train_timesteps, shape=[B])

    # ----- add diffusion noise -----
    trajectory_noise = randn_like(ntrajectory)
    keyframe_tcp_noise = randn_like(nkeyframe_tcp)

    noisy_trajectory_pos = position_scheduler.add_noise(
        ntrajectory[..., 0:3],
        trajectory_noise[..., 0:3],
        timesteps,
    )
    noisy_trajectory_rot = rotation_scheduler.add_noise(
        ntrajectory[..., 3:7],
        trajectory_noise[..., 3:7],
        timesteps,
    )
    noisy_trajectory = concat(
        noisy_trajectory_pos,
        noisy_trajectory_rot,
        dim=-1,
    )

    noisy_tcp_pos = position_scheduler.add_noise(
        nkeyframe_tcp[..., 0:3],
        keyframe_tcp_noise[..., 0:3],
        timesteps,
    )
    noisy_tcp_rot = rotation_scheduler.add_noise(
        nkeyframe_tcp[..., 3:7],
        keyframe_tcp_noise[..., 3:7],
        timesteps,
    )
    noisy_keyframe_tcp = concat(noisy_tcp_pos, noisy_tcp_rot, dim=-1)

    # ----- apply conditional action mask -----
    # mask shape is [B, A, Hact, 7]
    condition_mask = make_lowdim_mask(ntrajectory.shape)
    noisy_trajectory = where(condition_mask, ntrajectory, noisy_trajectory)

    noisy_targets = {
        "trajectory": noisy_trajectory,
        "keyframe_tcp": noisy_keyframe_tcp,
    }

    # ----- DiT predicts noise and auxiliary keyframe outputs -----
    pred = map4d_dit(noisy_targets, timesteps, nobs)

    # ----- losses -----
    trajectory_loss = l1(pred["trajectory"], trajectory_noise)
    keyframe_tcp_loss = l1(pred["keyframe_tcp"], keyframe_tcp_noise)
    keyframe_node_loss = l1(
        pred["keyframe_node_position"],
        nkeyframe_node_position,
    )
    gripper_loss = l1(pred["gripper_openness"], ngripper)

    loss = (
        trajectory_loss_weight * trajectory_loss
        + keyframe_tcp_loss_weight * keyframe_tcp_loss
        + keyframe_map4d_loss_weight * keyframe_node_loss
        + gripper_loss_weight * gripper_loss
    )
    return loss
```

### 11.2 Map4DDiT Forward

```python
def map4d_dit_forward(noisy_targets, timestep, obs):
    trajectory = noisy_targets["trajectory"]
    keyframe_tcp = noisy_targets["keyframe_tcp"]

    # expected push-box shapes:
    # trajectory:   [B, A=2, Hact=50, 7]
    # keyframe_tcp: [B, A=2, Hkey=4, 7]

    # ----- semantic observation context -----
    (
        semantic_xyz,
        semantic_token,
        lang_feat,
        state_feat,
        sampled_semantic_xyz,
        sampled_semantic_token,
    ) = observation_encoder(obs)

    # semantic_xyz/token:
    #   [B, Tobs * P, 3 / C] = [B, 12288, 3 / 240]
    # sampled_semantic_xyz/token:
    #   [B, Tobs * Ps, 3 / C] = [B, 1024, 3 / 240]

    # ----- Map4D context -----
    map_representation = Map4d_RLBench2PushBox(
        obs.node_position,
        obs.node_rotation,
        obs.size_parameters,
        obs.relation_parameters,
    )
    encoded_map_nodes = map_encoder(map_representation)
    # [B, Tobs, N, 3 + Dmap]

    map_xyz = encoded_map_nodes[..., 0:3]
    map_feat = encoded_map_nodes[..., 3:]
    map_token = map_feature_proj(map_feat)

    map_xyz = flatten_time_and_nodes(map_xyz)
    map_token = flatten_time_and_nodes(map_token)
    # [B, Tobs * N, 3 / C] = [B, 2, 3 / 240]

    # ----- query tokens -----
    traj_tokens, traj_xyz = build_trajectory_tokens(trajectory)
    # [B, A * Hact, C], [B, A * Hact, 3]

    node_query_xyz = repeat_current_node_position(
        obs.node_position[:, -1],
        repeats=Hkey,
    )
    node_tokens, node_xyz = build_node_tokens(node_query_xyz)
    # [B, Hkey * N, C], [B, Hkey * N, 3]

    tcp_tokens, tcp_xyz = build_tcp_tokens(keyframe_tcp)
    # [B, A * Hkey, C], [B, A * Hkey, 3]

    # ----- staged prediction head -----
    traj_feat, node_feat, tcp_feat = prediction_head(
        traj_tokens,
        traj_xyz,
        node_tokens,
        node_xyz,
        tcp_tokens,
        tcp_xyz,
        semantic_xyz,
        semantic_token,
        sampled_semantic_xyz,
        sampled_semantic_token,
        map_xyz,
        map_token,
        timestep,
        state_feat,
    )

    # ----- output heads -----
    cond = encode_denoising_timestep(timestep, state_feat)
    # [B, C], time embedding + mean recent robot-state feature

    node_feat = final_adaln(node_feat, cond)
    tcp_feat = final_adaln(tcp_feat, cond)
    traj_feat = final_adaln(traj_feat, cond)

    pred_node_position = node_head(node_feat)
    pred_tcp = tcp_head(tcp_feat)
    pred_trajectory = trajectory_head(traj_feat)
    pred_gripper = gripper_head(traj_feat)

    return {
        "trajectory": reshape(pred_trajectory, [B, A, Hact, 7]),
        "keyframe_tcp": reshape(pred_tcp, [B, A, Hkey, 7]),
        "keyframe_node_position": reshape(pred_node_position, [B, Hkey, N, 3]),
        "gripper_openness": reshape(pred_gripper, [B, A, Hact, 1]),
    }
```

### 11.3 Prediction Head

```python
def prediction_head(...):
    cond = encode_denoising_timestep(timestep, state_feat)

    if separate_map_cross_attn:
        # ----- stage 1: node -----
        node_tokens = node_map_attn(
            query=node_tokens,
            value=map_token,
            query_pos=rotary3d(node_xyz),
            value_pos=rotary3d(map_xyz),
            cond=cond,
        )
        node_feat = node_context_attn(
            query=node_tokens,
            value=semantic_token,
            query_pos=rotary3d(node_xyz),
            value_pos=rotary3d(semantic_xyz),
            cond=cond,
        )

        # ----- stage 2: tcp -----
        tcp_tokens = tcp_map_attn(
            query=tcp_tokens,
            value=map_token,
            query_pos=rotary3d(tcp_xyz),
            value_pos=rotary3d(map_xyz),
            cond=cond,
        )
        tcp_feat = tcp_context_attn(
            query=tcp_tokens,
            value=semantic_token,
            query_pos=rotary3d(tcp_xyz),
            value_pos=rotary3d(semantic_xyz),
            cond=cond,
        )

        # ----- stage 3: action -----
        traj_tokens = action_map_attn(
            query=traj_tokens,
            value=map_token,
            query_pos=rotary3d(traj_xyz),
            value_pos=rotary3d(map_xyz),
            cond=cond,
        )
        traj_feat = action_context_attn(
            query=traj_tokens,
            value=semantic_token,
            query_pos=rotary3d(traj_xyz),
            value_pos=rotary3d(semantic_xyz),
            cond=cond,
        )

    else:
        context_token = concat(semantic_token, map_token)
        context_xyz = concat(semantic_xyz, map_xyz)

        node_feat = node_context_attn(
            node_tokens, context_token, node_xyz, context_xyz, cond
        )
        tcp_feat = tcp_context_attn(
            tcp_tokens, context_token, tcp_xyz, context_xyz, cond
        )
        traj_feat = action_context_attn(
            traj_tokens, context_token, traj_xyz, context_xyz, cond
        )

    # stage 1 self-attention with compact scene context
    sampled_context_token = concat(sampled_semantic_token, map_token)
    sampled_context_xyz = concat(sampled_semantic_xyz, map_xyz)

    node_plan = concat(node_feat, sampled_context_token)
    node_plan_xyz = concat(node_xyz, sampled_context_xyz)
    node_plan = node_self_attn(node_plan, node_plan_xyz, cond)
    node_feat = node_plan[:, :num_node_query_tokens]

    # stage 2 self-attention conditioned on node plan
    if detach_stage_features:
        node_condition = detach(node_plan)
        node_condition_xyz = detach(node_plan_xyz)
    else:
        node_condition = node_plan
        node_condition_xyz = node_plan_xyz

    tcp_plan = concat(tcp_feat, node_condition)
    tcp_plan_xyz = concat(tcp_xyz, node_condition_xyz)
    tcp_plan = tcp_self_attn(tcp_plan, tcp_plan_xyz, cond)
    tcp_feat = tcp_plan[:, :num_tcp_query_tokens]

    # stage 3 self-attention conditioned on tcp plan
    if detach_stage_features:
        tcp_condition = detach(tcp_plan)
        tcp_condition_xyz = detach(tcp_plan_xyz)
    else:
        tcp_condition = tcp_plan
        tcp_condition_xyz = tcp_plan_xyz

    action_plan = concat(traj_feat, tcp_condition)
    action_plan_xyz = concat(traj_xyz, tcp_condition_xyz)
    action_plan = action_self_attn(action_plan, action_plan_xyz, cond)
    traj_feat = action_plan[:, :num_action_query_tokens]

    return traj_feat, node_feat, tcp_feat
```

## 12. Training Objective

训练目标分为四部分：

```text
trajectory diffusion noise loss
keyframe TCP diffusion noise loss
keyframe Map4D node position loss
gripper openness loss
```

当前权重：

```text
trajectory_loss_weight:     30.0
keyframe_tcp_loss_weight:   1.0
keyframe_map4d_loss_weight: 0.3
gripper_loss_weight:        30.0
```

trajectory 和 TCP 的 position 部分使用 position noise scheduler；rotation/quaternion 部分使用 rotation noise scheduler。

训练时的高层流程：

```text
batch
  -> normalize obs/action/keyframe
  -> sample diffusion timestep
  -> add noise to trajectory and keyframe TCP
  -> Map4DDiT(noisy_targets, timestep, obs)
  -> compute trajectory noise loss
  -> compute TCP noise loss
  -> compute keyframe node xyz loss
  -> compute gripper loss
  -> weighted sum
```

## 13. Inference Flow

推理时没有 GT trajectory/TCP，因此从 Gaussian noise 开始 denoise：

```text
trajectory   ~ N(0, I), shape [B, A, Hact, 7]
keyframe_tcp ~ N(0, I), shape [B, A, Hkey, 7]
```

然后循环 DDPM timesteps：

```text
for t in scheduler.timesteps:
    pred = Map4DDiT(sample, t, obs)
    trajectory = scheduler.step(pred.trajectory, trajectory)
    keyframe_tcp = scheduler.step(pred.keyframe_tcp, keyframe_tcp)
```

最后反归一化并拼接 gripper：

```text
trajectory_pred       [B, A, Hact, 7]
gripper_pred          [B, A, Hact, 1]
action_pred           [B, A, Hact, 8]
```

返回给 policy 的最终动作：

```text
action = action_pred[:, :, :n_action_steps]
```

## 14. Why Separate Map Cross-Attention

旧架构的问题不是 Map4D token 没有进入模型，而是它进入模型的方式不稳定：

```text
2 map tokens + 12288 semantic tokens -> one attention context
```

在这种情况下，map token 的梯度和注意力分配都容易被大规模点云 token 稀释。

新架构将 map context 变成显式条件：

```text
map tokens are attended independently before semantic point tokens
```

它保留点云语义感知能力，同时让 node/TCP/action 的 query 在每个 stage 都必须先读取 Map4D 结构信息。

## 15. Backward Compatibility

默认配置：

```yaml
separate_map_cross_attn: false
```

默认时：

```text
不创建 node_map_attn/tcp_map_attn/action_map_attn 参数
旧 forward 路径保持不变
旧训练和旧 checkpoint 不受影响
```

启用新架构时：

```yaml
separate_map_cross_attn: true
```

新增参数：

```text
node_map_attn
tcp_map_attn
action_map_attn
```

因此新架构需要从头训练，不能直接无损加载旧架构 checkpoint 的完整模型权重。
