# Key Components in Map4D_DiT Framework

## Dataset Format

Unified Format：
- task_name
- obs:
  - robot state: [T, robot_state_dim]
  - semantic field xyz: [T, num_point+Num_node, 3(xyz)]
  - semantic field dino feature: [T, num_point+Num_node, dino_dim]
  - {camera}_rgb: [T, H, W, 3]
  - {camera}_depth: [T, H, W, 1]
- map4d:
  - size parameter: [size_dim]
  - relation parameter: [relation_dim]
  - position parameter: [T, Num_node, 3]
  - rotation parameter: [T, Num_node, 4]  // quaternion_wxyz
- keyframe:
  - keyframe indices: [T, H_key]
  - future_keyframe_map4d_targets: [T, H_key, Num_node, 7]
  - future_keyframe_tcp_pose: [T, H_key, tcp_dim]
- actions: [T-1, action_dim]

Note: (TODO)
- `semantic field xyz` is a unified token set: RGB-D point xyz plus Map4D node-center xyz.
- `semantic field dino feature` uses the same rule for both RGB-D points and node centers:
  project xyz to camera RGB views, sample/interpolate DINO features, then combine multi-view
  features with distance/visibility weights.
- The model does not need token-type labels for RGB-D points vs node centers; Map4D structure is
  provided separately through Map4D features in cross attention.
- Still store a token-type/mask key for debugging and visualization. This key is dataset metadata,
  not a default model input.
- ManiSkill implementation mapping:
  - `obs/point_cloud/fused[..., :3]` -> RGB-D point xyz.
  - `map4d[..., 0:3]` -> node-center xyz.
  - `obs/dino_feature` currently covers RGB-D points only; add node-center projected DINO features.
  - Concatenate RGB-D point tokens and node-center tokens into unified `semantic field`.
- Keep the current model-facing keys for the first implementation:
  - write/read unified xyz through `obs/point_cloud/fused` with shape `[T, P+N, >=3]`.
  - write/read unified DINO through `obs/dino_feature` with shape `[T, P+N, D]`.
  - this avoids adding token-type labels or changing `Map4DDiT._semantic_context`, because it already
    treats `point_cloud[..., :3]` and `dino_feature` as one semantic field.
- `map4d` remains a separate graph/condition source:
  - do not remove `map4d[..., 0:3]` from `map4d`.
  - node centers are duplicated intentionally: once as semantic-field xyz tokens, once as graph node positions.
  - graph/node features still enter through map encoder/cross attention.
- Node-center DINO generation:
  - RGB-D point DINO can keep using `obs/point_cloud_source/fused/{camera_index,pixel_uv}`.
  - node centers need new projection metadata or direct projected features, because they do not have
    existing `point_cloud_source` rows.
  - project every `map4d[..., 0:3]` node center into the same ManiSkill cameras used by the fused point cloud.
  - sample DINO patch tokens at projected uv using the same patch-token rule as
    `build_point_semantic_features.py`.
  - if several cameras see the node center, combine visible-camera features with the same visibility/depth
    and distance weighting rule used for semantic-field points; do not silently fall back to zeros.
- Normalization rule:
  - fit `point_cloud` xyz limits on the unified xyz set `[RGB-D points; node centers]`.
  - fit `dino_feature` limits on the unified DINO set `[RGB-D point features; node-center features]`.
  - use the same `point_cloud` normalizer for Map4D graph positions and keyframe Map4D positions.
  - keep quaternion rotations, semantic ids, type/index-like fields unnormalized.
- Dataset validation checks:
  - `obs/point_cloud/fused.shape[:2] == obs/dino_feature.shape[:2]`.
  - `obs/semantic_field_source/token_type.shape == [T, P+N]`.
  - `token_type[..., :P] == 0` for RGB-D points and `token_type[..., P:] == 1` for Map4D node centers.
  - after unification, point count must be `P+N`, where `N == num_map_nodes`.
  - the last `N` tokens should equal `map4d[..., 0:3]` before normalization.
  - visual validation should render raw and normalized semantic-field points, with node-center tokens
    highlighted from `token_type` only for debugging, not used as a default model input.

Benchmark ManiSkill (.h5)

```json
{
  "demo_h5": {
    "attrs": {
      "task_name": "StackCube-v1",
      "semantic_field_format": "rgbd_points_plus_map4d_node_centers_v1",
      "semantic_field_node_count": "N",
      "semantic_feature_source": "dinov3_patch_token_projected_to_semantic_field",
      "pointcloud_cameras": ["base_camera", "hand_camera"]
    },
    "traj_*": {
      "actions": "[T-1, action_dim]",
      "obs": {
        "agent": "robot state fields",
        "extra/tcp_pose": "[T, 7]",
        "sensor_data/{camera}/rgb": "[T, H, W, 3]",
        "sensor_data/{camera}/depth": "[T, H, W, 1]",
        "point_cloud/fused": "[T, P+N, 6]  // unified semantic-field xyz+rgb; last N xyz are map4d node centers",
        "dino_feature": "[T, P+N, D]  // DINO features aligned with point_cloud/fused",
        "point_cloud_source/fused": {
          "camera_index": "[T, P, 1]  // RGB-D points only",
          "pixel_uv": "[T, P, 2]  // RGB-D points only"
        },
        "semantic_field_source": {
          "token_type": "[T, P+N]  // 0=RGB-D point, 1=Map4D node center; debug/visualization only",
          "node_center_token_indices": "[N]  // usually [P, P+1, ..., P+N-1]",
          "node_center_camera_index": "[T, N, K]",
          "node_center_pixel_uv": "[T, N, K, 2]",
          "node_center_camera_weight": "[T, N, K]"
        }
      },
      "env_states/actors/{actor_name}": "[T, actor_state_dim]"
    }
  },
  "map4d_sidecar_h5": {
    "attrs": {
      "task_name": "{task_name}",
      "representation_json": "map4d/representation/maps4d/{task}.json",
      "map4d_format": "pose_graph_parameters_v1",
      "target_format": "map4d_dit_local_delta_relative_quaternion_tcp_pos_gripper_v1",
      "keyframe_horizon": "H_key",
      "keyframe_tcp_dim": "tcp_dim"
    },
    "traj_*": {
      "map4d": "[T, N, 7]  // position(3) + quaternion_wxyz(4)",
      "size_parameters": "[size_dim]",
      "relation_parameters": "[relation_dim]",
      "future_keyframe_object_targets": "[T, H_key, N, 7]",
      "future_keyframe_tcp_pose": "[T, H_key, tcp_dim]",
      "future_keyframe_indices": "[T, H_key]"
    }
  }
}
```

`map4d_sidecar_h5.attrs.task_name` points to the representation json. Task-specific metadata
such as `node_names`, optional `actor_names`, `num_map_nodes`, `map4d_dim`,
`size_parameter_dim`, and `relation_parameter_dim` should be read from that json instead of
duplicated by hand in the sidecar attrs.

For current StackCube no-table data: `N=2`, `size_dim=6`, `relation_dim=0`, `tcp_dim=4`, `P=6144`, `P+N=6146`, `D=384`.


## Normalizer

Code in `4dmap_policy/map4d/backbone/model/common/normalizer.py`

```json
{
  "method": {
    "default_mode": "limits",
    "operation": "per-channel affine normalization",
    "limits": "x -> 2 * (x - min) / (max - min) - 1",
    "constant_channel": "map to 0 when range < eps",
    "gaussian_optional": "x -> (x - mean) / std"
  },
  "obs": {
    "robot_state": "normalize all state channels independently",
    "point_cloud": "normalize all 6 channels independently; xyz stats are also reused for Map4D coordinates",
    "dino_feature": "normalize every DINO channel independently when using precomputed features",
    "node_position": "normalize xyz with point_cloud xyz stats",
    "node_rotation": "quaternion_wxyz is canonicalized and is not normalized by graph normalizer",
    "size_parameters": "normalize each size channel independently",
    "relation_parameters": "normalize each relation channel independently if relation_dim > 0"
  },
  "targets": {
    "trajectory_pos": "normalize action xyz independently",
    "gripper_openness": "normalize gripper scalar independently",
    "keyframe_map4d_pos": "normalize keyframe node xyz; should share point_cloud xyz coordinate space",
    "keyframe_tcp_pos": "normalize keyframe TCP xyz independently",
    "keyframe_tcp_gripper": "normalize keyframe gripper scalar independently when tcp_dim == 4"
  },
  "not_normalized": {
    "actions_rotation": "quaternion / rotation fields are kept as rotation representation",
    "map4d_quaternion": "quaternion_wxyz is not normalized by normalize_map4d_graph_data",
    "rgb_image": "raw image tensor is not handled by LinearNormalizer",
    "depth_image": "raw depth image is not handled by LinearNormalizer",
    "semantic_field_source": "token_type, indices, uv, camera ids and weights are metadata/debug fields",
    "keyframe_indices": "index field",
    "map4d_features": "learned map encoder features use network normalization such as LayerNorm, not dataset LinearNormalizer"
  },
  "map4d_graph_data": {
    "absolute_xyz": ["node_pos", "x_pos", "x_aff", "edge_anchor position part"],
    "delta_xyz": ["edge_pose[..., 0:3]"],
    "rule": "absolute xyz uses point_cloud xyz scale+offset; delta xyz uses point_cloud xyz scale only"
  }
}
```

## Model Input

Hyper parameters:
- `h_obs`: observation horizon
- `h_action`: action horizon
- `h_keyframe`: keyframe horizon
- `P`: RGB-D point count
- `N`: Map4D node count (Read from {task_name}.json)
- `D`: DINO feature dim

```json
{
  "noised_target": {
    "trajectory_action": "[B, h_action, action_dim]  // normalized noisy action",
    "keyframe_tcp_action": "[B, num_arms, h_keyframe, tcp_dim]  // normalized noisy keyframe TCP action"
  },
  "context_input": {
    "semantic_field_xyz": "[B, h_obs, P+N, 3]  // normalized xyz from RGB-D points + Map4D node centers",
    "semantic_field_dino": "[B, h_obs, P+N, D]  // normalized DINO feature aligned with semantic_field_xyz"
  },
  "conditional_input": {
    "diffusion_timestep": "[B]  // timestep -> FiLM/timestep feature",
    "robot_state": "[B, h_obs, robot_state_dim]  // normalized robot state -> FiLM robot-state feature",
    "map4d": {
      "node_position": "[B, h_obs, N, 3]  // normalized with point-cloud xyz stats",
      "node_rotation": "[B, h_obs, N, 4]  // quaternion_wxyz, not normalized by graph normalizer",
      "size_parameters": "[B, size_dim]  // normalized",
      "relation_parameters": "[B, relation_dim]  // normalized if relation_dim > 0",
      "map_feature": "Map4D graph -> map encoder -> node/map features -> LayerNorm"
    }
  },
  "model_output": {
    "trajectory_action": "[B, h_action, action_dim]  // predicted action in normalized space",
    "keyframe_tcp_action": "[B, num_arms, h_keyframe, tcp_dim]  // predicted keyframe TCP action in normalized space"
  },
  "auxiliary_output": {
    "keyframe_node_position": "[B, h_keyframe, N, 3]  // normalized auxiliary node position target when enabled"
  },
  "debug_only": {
    "semantic_field_source/token_type": "[B, h_obs, P+N]  // 0=RGB-D point, 1=Map4D node center; not a model input by default",
    "camera_rgb_depth": "used to generate/precompute semantic field and DINO features; not a default DiT input when semantic_field_dino is precomputed"
  }
}
```

Input grouping:
- Noised targets are the tokens denoised by DiT.
- Semantic field is the normalized cross-attention context.
- Robot state and diffusion timestep enter as FiLM-style conditioning.
- Map4D enters as normalized graph parameters, then becomes map encoder features for conditioning/cross attention.
- All final action/keyframe outputs are predicted in normalized space and must be unnormalized before metric-space execution or visualization.

## Model
