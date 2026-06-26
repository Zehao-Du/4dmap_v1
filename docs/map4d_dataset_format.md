# Map4D DiT Dataset Format

This document describes the dataset produced by:

```bash
bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh <task> <demos> [resolution]
```

The current builder creates a ManiSkill demo HDF5, a raw keyframe sidecar, a final GT-map sidecar, and a `.env` manifest. Training uses the demo HDF5 plus the final GT-map sidecar.

Important: the dataset stores GT map data, not learned map features. `traj_*/map_node_feature` and `obs.map_node_feature` must not exist. `Map4DEncoder` runs online inside `Map4DDiT` and is trained jointly with the denoising network.

Naming convention:

- HDF5 sidecar storage keeps the historical dataset name `traj_*/map4d`.
- The training batch exposes the same tensor as `sample["obs"]["node_poses"]`.

## Files

Default output files:

- Demo HDF5:
  - `${DATA_ROOT}/ManiSkill/${TASK_NAME}/motionplanning/${TRAJ_NAME}.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.filtered.h5`
- Raw keyframe sidecar:
  - `${demo_stem}.map4d_dit_h${FUTURE_HORIZON}.keyframe.h5`
- Final GT-map sidecar:
  - `${demo_stem}.map4d_dit_h${FUTURE_HORIZON}.context.h5`
- Manifest:
  - `${final_sidecar_stem}.env`

Training should set:

```bash
MAP4D_DEMO_PATH=<demo h5>
MAP4D_KEYFRAME_SIDECAR_PATH=<final GT-map context h5>
MAP4D_NUM_TRAJ=<num trajectories>
```

## Demo HDF5

Each trajectory is stored under `traj_*`.

Top-level trajectory fields:

- `[TRAIN USED] traj_*/actions`: `[T_action, action_dim]`
- `traj_*/env_states`
- `traj_*/success`
- `traj_*/terminated`
- `traj_*/truncated`
- `[PARTLY TRAIN USED] traj_*/obs`

Observation fields used by training:

- `[TRAIN USED] traj_*/obs/agent/*`
  - Used to build `obs.robot_state`.
  - Current loader prefers `qpos`, `qvel`, `tcp_pose`, and `tcp_pos`, then pads/truncates to `robot_state_dim`.
- `[TRAIN USED] traj_*/obs/extra/tcp_pose`: `[T_frame, 7]`
  - Used when keyframe TCP targets need to be materialized from the demo.
  - Layout: `pos(3) + quat_wxyz(4)`.
- `[TRAIN USED] traj_*/obs/point_cloud/fused`: `[T_frame, P, 6]`
  - Used as `obs.point_cloud`.
  - Layout: `xyzrgb`.
  - The model consumes only `xyz = point_cloud[..., :3]`.
- `[TRAIN USED] traj_*/obs/dino_feature`: `[T_frame, P, D_sem]`
  - Used as `obs.dino_feature`.
  - Must be point-aligned with `obs/point_cloud/fused`.
  - Must be produced by a DINO/DINOv3 model. RGB-derived placeholder features are rejected.
  - Must carry DINO provenance metadata, for example `semantic_feature_model=dinov3_*` or `feature_type=dinov3`.

Observation fields kept for data construction or debugging:

- `traj_*/obs/sensor_data/<camera>/rgb`
- `traj_*/obs/sensor_data/<camera>/depth`
- `traj_*/obs/sensor_param/<camera>/intrinsic_cv`
- `traj_*/obs/sensor_param/<camera>/extrinsic_cv`
- `traj_*/obs/point_cloud/<camera>`: `[T_frame, P_camera, 6]`
- `traj_*/obs/point_cloud_source/fused/camera_index`: `[T_frame, P]`
- `traj_*/obs/point_cloud_source/fused/pixel_uv`: `[T_frame, P, 2]`
  - Used by the dataset builder to align DINO patch tokens to sampled point-cloud points.
- `traj_*/obs/tcp_trajectory`

Current Semantic Field modes:

- `SEMANTIC_FEATURE_MODE=dinov3` (default)
  - Loads a frozen DINOv3 model once per builder process.
  - Extracts RGB patch tokens per camera and frame.
  - Assigns the nearest patch token to each sampled point using `point_cloud_source`.
  - Writes `traj_*/obs/dino_feature`.
  - `SEMANTIC_FEATURE_DTYPE=float32` by default; `float16` can be used to reduce HDF5 size/write time. The training dataset casts loaded features to `float32`.
- `SEMANTIC_FEATURE_MODE=existing_dino`
  - Requires `traj_*/obs/dino_feature` to already exist.
  - Validates shape `[T_frame, P, D_sem]`.
  - Validates that the feature has DINO provenance metadata.
  - Does not generate fallback RGB features.

## Raw Keyframe Sidecar HDF5

The raw keyframe sidecar is produced by `helper/build_keyframe_aux_dataset.py`. The final sidecar copies these datasets, so training normally reads them from the final sidecar.

Per-trajectory fields:

- `[TRAIN USED as obs.node_poses] traj_*/map4d`: `[T_frame, N_target_node, 9]`
  - GT node pose.
  - Layout: `pos(3) + rot6d(6)`.
- `[TRAIN USED] traj_*/size_parameters`: `[D_size]`
  - Static geometry parameters required by `Map4DEncoder`.
  - Produced by resetting the matching ManiSkill task env episode and reading geometry from `env.unwrapped`.
- `[TRAIN USED] traj_*/relation_parameters`: `[D_relation]`
  - Static relation parameters required by `Map4DEncoder`.
  - Produced from ManiSkill env relation geometry, for example PlugCharger `_peg_gap`.
- `traj_*/tcp_pose`: `[T_frame, 7]`
- `traj_*/keyframe_indices`: `[N_keyframe]`
- `traj_*/future_keyframe_indices`: `[T_frame, H_key]`
- `[TRAIN USED] traj_*/future_keyframe_object_targets`: `[T_frame, H_key, N_target_node, 9]`
  - Loaded as `keyframe.map4d`.
  - Layout: `local_delta_pos(3) + delta_rot6d(6)`.
- `[TRAIN USED] traj_*/future_keyframe_tcp_pose`: `[T_frame, H_key, tcp_dim]`
  - Loaded as `keyframe.tcp`.
  - StackCube current format: `tcp_dim=4`, `local_delta_pos(3) + gripper(1)`.
  - PlugCharger current format: `tcp_dim=7`, `local_delta_pos(3) + delta_quat(4)`.

Task-specific extra fields may exist:

- `traj_*/tcp_pos_gripper`: `[T_frame, 4]`
- `traj_*/gripper_target`: `[T_frame, 1]`
- `traj_*/keyframe_tcp_pose`: `[N_keyframe, tcp_dim]`

The keyframe sidecar stores `structural_parameter_source = "maniskill_env"` in file/group attrs. There is no maps4d JSON default fallback: missing adjacent ManiSkill metadata JSON, missing episode seed/options records, or missing required env attributes should fail during dataset construction.

Current extraction rules:

- StackCube `size_parameters`: `[cubeA_size(3), cubeB_size(3), table_size(3)]`, from `cube_half_size` and `table_scene.table_length/width/height`.
- StackCube `relation_parameters`: empty `[0]`.
- PlugCharger `size_parameters`: `[charger_body(3), charger_prong(3), receptacle_center_divider(3), receptacle_face_loop(5)]`, from `_base_size`, `_peg_size`, `_receptacle_size`, `_peg_gap`, and `_clearance`.
- PlugCharger `relation_parameters`: `[prong_gap_half]`, from `_peg_gap`.

## Final GT-Map Sidecar HDF5

The final sidecar is produced by `scripts/data_collection/helpers/build_map4d_context_dataset.py`.

It copies the raw keyframe sidecar and validates that the demo HDF5 contains:

- `traj_*/obs/point_cloud/fused`
- `traj_*/obs/dino_feature`

Required file attrs:

- `task_name`
- `actor_names`
- `map_context_format = "map4d_gt_pose_sidecar_v1"`
- `map_encoder_location = "online_in_Map4DDiT"`
- `num_map_nodes`
- `target_format`
- `tcp_dim`
- `tcp_target`

Forbidden fields:

- `[FORBIDDEN] traj_*/map_node_feature`

If `traj_*/map_node_feature` exists, the context builder should fail. Map node features are internal online tensors produced by `Map4DEncoder` during training.

## Training Batch

`ManiSkillMap4DDataset.__getitem__` returns:

```text
sample["obs"]["robot_state"]
sample["obs"]["point_cloud"]
sample["obs"]["dino_feature"]
sample["obs"]["node_poses"]
sample["obs"]["size_parameters"]
sample["obs"]["relation_parameters"]

sample["action"]["trajectory"]
sample["action"]["gripper_openness"]

sample["keyframe"]["map4d"]
sample["keyframe"]["tcp"]
```

For `n_obs_steps=2` and `n_map_step=2`, typical batch shapes are:

- `[TRAIN USED] obs.robot_state`: `[B, 2, robot_state_dim]`
- `[TRAIN USED] obs.point_cloud`: `[B, 2, P, 6]`
- `[TRAIN USED] obs.dino_feature`: `[B, 2, P, D_sem]`
- `[TRAIN USED] obs.node_poses`: `[B, 2, N_target_node, 9]`
- `[TRAIN USED] obs.size_parameters`: `[B, D_size]`
- `[TRAIN USED] obs.relation_parameters`: `[B, D_relation]`
- `[TRAIN USED] action.trajectory`: `[B, H_action, 7]`
- `[TRAIN USED] action.gripper_openness`: `[B, H_action, 1]`
- `[TRAIN USED] keyframe.map4d`: `[B, H_key, N_target_node, 9]`
- `[TRAIN USED] keyframe.tcp`: `[B, H_key, tcp_dim]`

Inside `Map4DDiT`, the batch is converted to context tokens:

- Semantic Field context:
  - `obs.point_cloud[..., :3]`
  - `obs.dino_feature`
- GT map context:
  - `obs.node_poses`
  - `obs.size_parameters`
  - `obs.relation_parameters`
  - online `Map4DEncoder(...) -> [B, n_map_step, N_map_node, 3 + D_map]`

The online encoder output is not stored in HDF5.

## Example StackCube Shapes

A valid one-trajectory StackCube dataset should have:

```text
Demo HDF5:
traj_0/obs/point_cloud/fused       [108, 6144, 6]
traj_0/obs/dino_feature            [108, 6144, D_dino]

Final GT-map sidecar:
traj_0/map4d                       [108, 3, 9]
traj_0/size_parameters             [9]
traj_0/relation_parameters         [0]
traj_0/tcp_pose                    [108, 7]
traj_0/future_keyframe_indices     [108, 4]
traj_0/future_keyframe_object_targets [108, 4, 3, 9]
traj_0/future_keyframe_tcp_pose    [108, 4, 4]
```
