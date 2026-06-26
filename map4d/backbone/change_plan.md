# DiT Input

Hyper parameters:
- `h_obs`: observation horizon
- `h_action`: action horizon
- `h_keyframe`: keyframe horizon
- `P`: RGB-D point count
- `N`: Map4D node count, read from `{task_name}.json`
- `D`: DINO feature dim

## Noised Target

```json
{
  "trajectory_action": "[B, h_action, action_dim]  // normalized noisy action",
  "keyframe_tcp_action": "[B, num_arms, h_keyframe, tcp_dim]  // normalized noisy keyframe TCP action"
}
```

## Context Input

```json
{
  "semantic_field_xyz": "[B, h_obs, P+N, 3]  // normalized RGB-D point xyz + Map4D node-center xyz",
  "semantic_field_dino": "[B, h_obs, P+N, D]  // normalized DINO feature aligned with semantic_field_xyz"
}
```

## Conditional Input

```json
{
  "diffusion_timestep": "[B]  // timestep -> FiLM/timestep feature",
  "robot_state": "[B, h_obs, robot_state_dim]  // normalized robot state -> FiLM robot-state feature",
  "map4d": {
    "node_position": "[B, h_obs, N, 3]  // normalized with point-cloud xyz stats",
    "node_rotation": "[B, h_obs, N, 4]  // quaternion_wxyz, not normalized by graph normalizer",
    "size_parameters": "[B, size_dim]  // normalized",
    "relation_parameters": "[B, relation_dim]  // normalized if relation_dim > 0",
    "map_feature": "Map4D graph -> map encoder -> node/map features -> LayerNorm"
  }
}
```

## DiT Output

```json
{
  "trajectory_action": "[B, h_action, action_dim]  // predicted action in normalized space",
  "keyframe_tcp_action": "[B, num_arms, h_keyframe, tcp_dim]  // predicted keyframe TCP action in normalized space"
}
```

## Auxiliary Output

```json
{
  "keyframe_node_position": "[B, h_keyframe, N, 3]  // normalized auxiliary node position target"
}
```

## Notes

- Noised targets are the tokens denoised by DiT.
- Semantic field is the normalized cross-attention context.
- Robot state and diffusion timestep enter as FiLM-style conditioning.
- Map4D enters as normalized graph parameters; the map encoder converts them to node/map features, followed by LayerNorm.
- Node rotation stays as canonical quaternion_wxyz representation and is not normalized by `normalize_map4d_graph_data`.
- Final action and keyframe TCP outputs are predicted in normalized space and must be unnormalized before metric-space execution or visualization.

# PPI Algorithm Flow

## Inputs

```json
{
  "obs": {
    "agent_pos": "[B, h_obs, state_dim]  // normalized robot state",
    "point_cloud": "[B, h_obs, P, 6]  // normalized point cloud; model uses xyz only",
    "dino_feature": "[B, h_obs, P, D]  // normalized point feature",
    "lang": "[B, h_obs, lang_dim]  // optional normalized language feature",
    "initial_point_flow": "[B, h_obs, F, 3]  // normalized initial point-flow query coordinates"
  },
  "targets": {
    "action": "[B, h_continuous+h_keyframe, 16]  // left xyz+quat, right xyz+quat, left/right openness",
    "point_flow": "[B, h_keyframe, F, 3]"
  }
}
```

## Training

1. Normalize `batch["obs"]`, `batch["action"]`, and `batch["point_flow"]`.
2. Keep only `point_cloud[..., :3]` for model geometry.
3. Encode observations with `ObservationEncoder`:
   - full point context: `context_coord`, `context_feat`
   - sampled point context: `pn_coord`, `pn_feat`
   - robot/language condition: `state_feat`, `lang_feat`
   - point-flow query: `pointflow_coords`, `pointflow_feat`
4. Sample diffusion timestep `t` and Gaussian noise.
5. Add noise separately to:
   - left/right xyz with position scheduler
   - left/right quaternion with rotation scheduler
   - openness is kept as direct regression target, not diffused.
6. Run `DiffusionHeadPPI*` on noisy left/right trajectories plus encoded observation context.
7. Inside the diffusion head:
   - gripper tokens cross-attend to full point context.
   - point-flow query tokens cross-attend to full point context.
   - point-flow tokens self-attend with sampled context, then predict keyframe point flow.
   - keyframe gripper tokens attend with point-flow features, then predict keyframe pose/action.
   - continuous gripper tokens attend with detached keyframe features, then predict continuous pose/action.
8. Loss:
   - xyz noise L1 loss, weight `30`
   - quaternion noise L1 loss, weight `10`
   - openness L1 loss, weight `30`
   - point-flow position L1 loss, weight `600`

## Inference

1. Normalize observation dict.
2. Keep `point_cloud[..., :3]` and encode the first `h_obs` observations.
3. Initialize action trajectory as Gaussian noise.
4. For every DDPM reverse timestep:
   - predict left/right action noise and point flow.
   - update xyz with position scheduler.
   - update quaternion with rotation scheduler.
   - copy predicted openness directly from model output.
5. Unnormalize final action and point-flow predictions.
6. Return execution action slice:
   - `start = h_obs - 1`
   - `end = start + n_action_steps`

## Key Idea

PPI uses predicted point flow as an intermediate geometric plan. The model predicts point flow first, uses it to strengthen keyframe gripper prediction, then uses keyframe features to guide dense continuous action prediction.

# Map4D DiT Algorithm Flow

基本与PPI相同，下面是PPI中变量与map4d dit变量对应：

map4d dit中 map4d node center对应PPI中point flow
map4d dit在condition侧新加了map4d feature
map4d dit使用单臂，但保留双臂接口
其余流程一致