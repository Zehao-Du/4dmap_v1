# map4d backbone

single robot arm first, with a preserved dual-arm interface.

## DiT

### Input

transformer tokens

noisy target tokens

- Noised trajectory [7] = delta_pos(3) + delta_quat(4)
- Noised Keyframe node pose [n*9] = n node \* local_delta_pos(3)+delta_rot6d(6)
- Noised Keyframe tcp pose [7] = local_delta_pos(3) + delta_quat(4)

Target tensor interface:

- `N_arm` is an explicit arm axis for robot targets. The first implementation uses `N_arm=1`; the tensor layout keeps this axis so left/right arms can be added later without changing the denoising order.
- `trajectory`: canonical shape `[B, N_arm, H_action, 7]`. Single-arm shorthand `[B, H_action, 7]` may be accepted by inserting `N_arm=1`.
- `keyframe_node`: `[B, H_key, N_target_node, 9]`. Map node targets are scene-level and are not duplicated per arm.
- `keyframe_tcp`: canonical shape `[B, N_arm, H_key, tcp_dim]`. Single-arm shorthand `[B, H_key, tcp_dim]` may be accepted by inserting `N_arm=1`.
- `gripper_openness`: `[B, N_arm, H_action, 1]`, predicted by direct regression from action token features.
- `arm_embed`: `[1, N_arm, 1, D_model]` is added to arm-specific trajectory and tcp tokens. For `N_arm=2`, the implementation can use PPI-style separate left/right heads or shared heads with arm embeddings; the interface must not hard-code single-arm-only shapes.

context tokens

- unified 3D context field tokens
  - Semantic Field points: point cloud xyz + per-point DINOv3 feature
  - map feature points: maps4d node coordinate + map encoder feature

Context token sources:

- Semantic Field point tokens: visual-geometric context follows PPI's representation. Each observation frame provides a point cloud and aligned per-point semantic feature. The point input is `point_cloud[..., 0:3]`; the semantic input is `dino_feature` with one feature vector per point. Coordinates and semantic features are jointly projected into DiT context tokens.
- RGB Image Features: not used as the primary Map4D DiT context in the Semantic Field version. RGB is used upstream to build per-point DINOv3 features.
- Depth Image Features: not inserted as raw depth tokens. Depth is used upstream to back-project RGB-D observations into the Semantic Field point cloud.
- map feature points: map4d object pose, object geometry, semantic node features, affordance points, and relation/edge parameters are encoded by `map4d/encoder/map_encoder.py`. Each output node is treated as one sparse point in the same 3D context field as Semantic Field points: xyz is the node coordinate, and feature is the learned map node feature.
- robot state: encoded by MLP and used as FiLM/AdaLN global conditioning.
- denoising step: timestep embedding used as FiLM/AdaLN global conditioning.
- Context tokens are read-only key/value features. They are not denoised, decoded, or updated by target token self-attention.

Semantic Field interface:

- `obs.point_cloud`: `[B, T_obs, P, 3]` or `[B, T_obs, P, >=3]`; only xyz is consumed by the network.
- `obs.dino_feature`: `[B, T_obs, P, D_sem]`; feature dimension is configured by `semantic_feature_dim`.
- Tokens are flattened over observation time and points: `[B, T_obs * P, D_model]`.
- Each Semantic Field token is produced from `concat(xyz, dino_feature)` by an MLP.
- `max_context_tokens` must cover `T_obs * P + n_map_step * N_node`, where `N_node` is the number of map nodes for the task.
- Optional point subsampling can happen in the dataset or in the model, but the default plan is to pre-sample a fixed `P` in the dataset, matching PPI-style preprocessing.

Map feature interface:

- The dataset provides GT maps, not map encoder features:
  - `obs.node_poses`: `[B, n_map_step, N_target_node, 9]`
  - `obs.size_parameters`: `[B, D_size]` or `[D_size]`
  - `obs.relation_parameters`: `[B, D_relation]` or `[D_relation]`
- `traj_*/map_node_feature` and `obs.map_node_feature` are not valid dataset/model input fields.
- `Map4DDiT` owns `Map4DEncoder` from `map4d/encoder/map_encoder.py`; its weights are trained jointly with the denoising transformer.
- Inside `Map4DDiT`, the map encoder converts GT map inputs into per-node coordinate + feature:
  - per frame internal tensor: `[B, N_node, 3 + D_map]`
  - history internal tensor: `[B, n_map_step, N_node, 3 + D_map]`
  - the first 3 dimensions are the node coordinate `node_pos`
  - the remaining `D_map` dimensions are the learned node feature
- `Map4DDiT` treats each map node output as a 3D context point:
  - `map_point_xyz`: `[B, n_map_step, N_node, 3]`
  - `map_point_feature`: `[B, n_map_step, N_node, D_map]`
- `Map4DDiT` owns a source-specific projection MLP:
  - `map_feature_proj: D_map -> D_model`
- Map feature points are flattened over map history and nodes:
  - xyz: `[B, n_map_step, N_node, 3] -> [B, n_map_step * N_node, 3]`
  - token: `[B, n_map_step, N_node, D_model] -> [B, n_map_step * N_node, D_model]`
- Because map feature points come from multiple timesteps, they must receive temporal position encoding.
  - `map_time_pos_embed`: `[1, n_map_step, 1, D_model]`
  - optional `map_node_pos_embed`: `[1, 1, N_node, D_model]` if node identity/order should be explicitly encoded
  - map token before flattening: `map_feature_proj(map_feature) + map_time_pos_embed + map_node_pos_embed + source_embed`
- Raw map4d object history, size parameters, semantic node features, and relation parameters remain part of the online map encoder input. They should not be separately inserted into `Map4DDiT._context_tokens`.
- The number of map nodes is task-fixed within a batch, so no node padding/mask is required in the first implementation.
- `num_target_nodes` and `num_map_nodes` are separate concepts:
  - `num_target_nodes` controls keyframe node pose targets
  - `num_map_nodes` controls map encoder context tokens
- `n_map_step` can be different from `T_obs`; it controls how many historical map frames are encoded as map context.

Unified 3D context field interface:

- Semantic Field and map feature points are concatenated into one read-only 3D field:
  - `context_xyz`: `[B, T_obs * P + n_map_step * N_node, 3]`
  - `context_token`: `[B, T_obs * P + n_map_step * N_node, D_model]`
- Semantic Field token:
  - xyz: `point_cloud[..., :3]`
  - token: `semantic_field_proj(concat(xyz, dino_feature)) + semantic_source_embed`
- Map feature token:
  - xyz: map node coordinate from `Map4DEncoder`
  - token: `map_feature_proj(encoded_map_nodes[..., 3:]) + map_time_pos_embed + map_node_pos_embed + map_source_embed`
- Cross-attention receives only this unified field:
  - key/value: `context_token`
  - value position: `context_xyz`
- Source embeddings let the network distinguish Semantic Field points from map feature points, but attention sees both as points in the same 3D space.

global conditioning / modulation

- robot state (MLP+FiLM)
- Denoising Step (MLP+FiLM)
- Default `D_model=240`, matching PPI-style attention and satisfying the 3D rotary position encoding requirement.

### PPI-style attention design

Map4DDiT follows PPI's staged, unidirectional attention pattern instead of concatenating all tokens into one full self-attention sequence.

Read-only context:

- `context_token`: `[B, T_obs * P + n_map_step * N_node, D_model]`
- `context_xyz`: `[B, T_obs * P + n_map_step * N_node, 3]`
- Semantic Field points and map feature points are concatenated into this single 3D context field.
- Cross-attention uses `context_token` as key/value and `context_xyz` as the 3D positional input.

Prediction stages:

1. `keyframe_node` stage.
   - Query: noisy keyframe node pose tokens.
   - Key/value: unified 3D context field.
   - Output hidden features are decoded by the node pose head.

2. `keyframe_tcp` stage.
   - Query: noisy keyframe tcp pose tokens, with arm embedding.
   - Key/value: unified 3D context field and keyframe node hidden features.
   - Output hidden features are decoded by the tcp pose head.

3. `trajectory/action` stage.
   - Query: noisy trajectory tokens, with arm embedding.
   - Key/value: unified 3D context field, keyframe node hidden features, and keyframe tcp hidden features.
   - Output hidden features are decoded by the trajectory head and gripper openness head.

Unidirectional dependency:

- `keyframe_node` cannot attend to tcp or trajectory/action tokens.
- `keyframe_tcp` can attend to keyframe node features, but cannot attend to trajectory/action tokens.
- `trajectory/action` can attend to both keyframe node and keyframe tcp features.
- Context features are never updated by target branches.
- By default, downstream stages consume upstream hidden features with `detach()`, matching PPI's stabilizing design for continuous action prediction. Removing detach should be treated as an ablation.

Each stage uses the same diffusion timestep and robot-state AdaLN/FiLM conditioning. The stages are ordered inside a single denoising network call; they do not use separate schedulers.

### Map4DDiT v1 algorithm

1. Build noisy target tokens.
   - `trajectory`: `[B, N_arm, H_action, 7] -> [B, N_arm * H_action, D_model]`
   - `keyframe_node`: `[B, H_key, N_target, 9] -> [B, H_key * N_target, D_model]`
   - `keyframe_tcp`: `[B, N_arm, H_key, tcp_dim] -> [B, N_arm * H_key, D_model]`
   - Add target positional embedding and target type embedding.
   - Add arm embedding to trajectory and tcp tokens.

2. Build Semantic Field context tokens.
   - Take `point_cloud[..., :3]` as xyz.
   - Concatenate xyz with aligned `dino_feature`.
   - Flatten observation time and points: `[B, T_obs, P, 3 + D_sem] -> [B, T_obs * P, 3 + D_sem]`.
   - Project with `semantic_field_proj` to `[B, T_obs * P, D_model]`.
   - Add Semantic Field source embedding.
   - Keep point xyz as the Semantic Field part of `context_xyz`.

3. Build historical map node context tokens.
   - Take the last `n_map_step` GT node pose frames from `obs.node_poses`.
   - Combine them with `obs.size_parameters` and `obs.relation_parameters` to construct the maps4d representation inside `Map4DDiT`.
   - Per map frame: `Map4DEncoder(map_representation_t) -> [B, N_node, 3 + D_map]`.
   - Stack history: `[B, n_map_step, N_node, 3 + D_map]`.
   - Split output into map node xyz and map node feature.
   - Project feature with `map_feature_proj` to `[B, n_map_step, N_node, D_model]`.
   - Add map temporal position embedding so the model can distinguish map history order.
   - Add optional map node position embedding so the model can distinguish task-fixed node identity/order.
   - Add map source embedding.
   - Flatten xyz to `[B, n_map_step * N_node, 3]`.
   - Flatten tokens to `[B, n_map_step * N_node, D_model]`.

4. Build unified 3D context field.
   - Concatenate Semantic Field xyz and map node xyz:
     - `context_xyz`: `[B, T_obs * P + n_map_step * N_node, 3]`
   - Concatenate Semantic Field tokens and map feature tokens:
     - `context_token`: `[B, T_obs * P + n_map_step * N_node, D_model]`

5. Build global conditioning.
   - Take the last `T_obs` robot states.
   - Pad on the left if the available history is shorter than `T_obs`.
   - Flatten robot state history and encode with `robot_encoder`.
   - Encode denoising timestep with `time_encoder`.
   - Add them to form the AdaLN/FiLM conditioning vector.

6. Run staged PPI-style attention.
   - Node branch:
     - keyframe node tokens cross-attend to the unified 3D context field.
     - keyframe node tokens run branch self-attention.
     - node hidden features are decoded by the node pose head.
   - TCP branch:
     - tcp tokens cross-attend to the unified 3D context field.
     - tcp tokens attend to keyframe node hidden features.
     - tcp tokens run branch self-attention.
     - tcp hidden features are decoded by the tcp pose head.
   - Action branch:
     - trajectory tokens cross-attend to the unified 3D context field.
     - trajectory tokens attend to keyframe node hidden features and tcp hidden features.
     - trajectory tokens run branch self-attention.
     - trajectory hidden features are decoded by the trajectory and gripper heads.

7. Predict denoised targets.
   - trajectory head predicts `[B, N_arm, H_action, 7]`.
   - node head predicts `[B, H_key, N_target, 9]`.
   - tcp head predicts `[B, N_arm, H_key, tcp_dim]`.
   - gripper openness head predicts `[B, N_arm, H_action, 1]` from trajectory token features.

### Output

- DeNoised trajectory [7] = delta_pos(3) + delta_quat(4)
- DeNoised Keyframe node pose [n*9] = n node \* local_delta_pos(3)+delta_rot6d(6)
- DeNoised Keyframe tcp pose [7] = local_delta_pos(3) + delta_quat(4)

Pose target convention:

- Trajectory target uses delta pose in formal settings: delta_pos(3) + delta_quat(4). StackCube may use delta_pos only for debugging/compatibility.
- Keyframe node pose target uses local_delta_pos(3) + delta_rot6d(6).
- Keyframe tcp pose target uses local_delta_pos(3) + delta_quat(4).
- Trajectory and keyframe tcp targets keep an explicit `N_arm` dimension. Single-arm runs use `N_arm=1`.
- local_delta_pos is expressed in the current pose frame: `R_current^T * (p_future - p_current)`.
- Rotation targets are relative rotations from current pose to future keyframe pose.

Gripper openness:

- map4d follows PPI's gripper openness design.
- Gripper openness is not part of the noisy diffusion target.
- It is predicted by a direct regression head from trajectory token features.
- It is supervised with L1 loss against GT openness.
- During reverse diffusion, only delta_pos and delta_quat are scheduler-stepped; openness is read from the latest model output.

Inference:

- Keyframe node/tcp tokens are latent plan tokens.
- Trajectory, keyframe node, and keyframe tcp tokens are jointly denoised during training and inference.
- Keyframe node/tcp hidden features guide trajectory generation through the staged unidirectional attention path.
- Only trajectory tokens are sent to the controller.
  
### Loss

diffusion losses are L1 losses on predicted noise

weight

- trajectory delta pose noise: 1
- keyframe tcp delta pose noise: 1
- keyframe node delta pose noise: 0.3

direct regression losses

- gripper openness: L1 loss on GT openness

### Scheduler

Rotation representation:

- TCP/action rotation uses normalized quaternion rot(4). Quaternions are canonicalized to `w >= 0` before training and renormalized after each denoising step.
- Object rotation uses rot6d(6). rot6d is denoised in 6D Euclidean space and orthogonalized with `rotation_6d_to_matrix` before geometric use.
- Position and rotation use separate schedulers. Quaternion/rot6d fields are not mean-std normalized as ordinary vectors.
- Gaussian diffusion noise moves quaternions off the unit sphere, so every noisy/denoised quaternion must be projected back to a canonical unit quaternion. This is a required map4d addition over the original PPI implementation.

### Normalizer

- map4d follows PPI's `LinearNormalizer` design for non-rotation numeric fields.
- delta_pos/local_delta_pos fields use the PPI limits normalizer and are scaled to `[-1, 1]`.
- robot state, map4d pose fields, point cloud xyz, and other non-rotation numeric context fields follow the existing limits normalizer where available.
- Per-point DINOv3 semantic features are learned features and are not min-max normalized as pose/action fields. They may be passed through as extracted, or standardized by a feature-specific normalizer if dataset statistics are provided.
- Quaternion and rot6d fields are not mean-std normalized as ordinary vectors.
- Quaternion fields are canonicalized to `w >= 0` and renormalized after each denoising step.
- Noisy quaternion inputs are also renormalized before entering the denoising transformer.
- rot6d fields are denoised in raw 6D representation space and orthogonalized with `rotation_6d_to_matrix` before geometric use.
- Gripper openness follows PPI's direct regression path and uses the action/task normalizer if the dataset provides one.

## Dataset Built From Scratch

The one-click builder is:

```bash
bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh <task> <demos> [resolution]
```

Supported tasks are `stackcube` / `StackCube-v1` and `plugcharger` / `PlugCharger-v1`. The script runs inside the `4dmap` conda environment. It builds the training data in three HDF5 layers:

1. ManiSkill demo HDF5, augmented in place with point-cloud Semantic Field inputs.
2. Keyframe sidecar HDF5 with future keyframe node/TCP targets.
3. Final GT-map sidecar HDF5, copied from the keyframe sidecar and validated against the Semantic Field inputs.

Default output paths:

- Demo HDF5:
  - `${DATA_ROOT}/ManiSkill/${TASK_NAME}/motionplanning/${TRAJ_NAME}.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.filtered.h5`
  - `StackCube-v1`: `StackCube.rgb+depth.pd_ee_delta_pos.physx_cpu.filtered.h5`
  - `PlugCharger-v1`: `PlugCharger.rgb+depth.pd_ee_delta_pose.physx_cpu.filtered.h5`
- Raw keyframe sidecar:
  - `${demo_stem}.map4d_dit_h${FUTURE_HORIZON}.keyframe.h5`
- Final GT-map context sidecar:
  - `${demo_stem}.map4d_dit_h${FUTURE_HORIZON}.context.h5`
- Manifest:
  - `${final_sidecar_stem}.env`

### Demo HDF5

The demo file keeps the standard ManiSkill trajectory layout and adds the model's per-frame visual-geometric inputs under each `traj_*` group.

Standard ManiSkill data used by the dataset:

- `traj_*/actions`: `[T_action, action_dim]`
  - StackCube uses `pd_ee_delta_pos`, typically delta position plus gripper.
  - PlugCharger uses `pd_ee_delta_pose`, typically delta position, axis-angle rotation, and gripper.
- `traj_*/obs/agent/*`: robot proprioception such as `qpos` and `qvel`.
- `traj_*/obs/extra/tcp_pose`: `[T_frame, 7]`, TCP pose in `[pos(3), quat_wxyz(4)]`.
- `traj_*/obs/sensor_data/<camera>/rgb`: RGB images from replay.
- `traj_*/obs/sensor_data/<camera>/depth`: depth images from replay.
- `traj_*/obs/sensor_param/<camera>/intrinsic_cv`
- `traj_*/obs/sensor_param/<camera>/extrinsic_cv`
- `traj_*/env_states/actors/<actor_name>`: actor states used to reconstruct Map4D pose when needed.

Point-cloud data added by `build_pointcloud_dataset.py`:

- `traj_*/obs/point_cloud/<camera>`: `[T_frame, P_camera, 6]`
- `traj_*/obs/point_cloud/fused`: `[T_frame, P, 6]`
  - channel layout: `xyzrgb`
  - `xyz`: world coordinates
  - `rgb`: sampled RGB values from the source RGB-D images
  - default `P = POINTCLOUD_NUM_POINTS = 6144`
- `traj_*/obs/point_cloud_source/fused/camera_index`: `[T_frame, P]`
- `traj_*/obs/point_cloud_source/fused/pixel_uv`: `[T_frame, P, 2]`
  - used only during dataset construction to project DINO patch tokens back to sampled points.

Semantic Field feature added by `build_point_semantic_features.py`:

- `traj_*/obs/dino_feature`: `[T_frame, P, D_sem]`

`SEMANTIC_FEATURE_MODE` controls this dataset:

- `dinov3` (default):
  - loads one frozen DINOv3 backbone per script process.
  - extracts patch tokens from `obs/sensor_data/<camera>/rgb`.
  - uses `obs/point_cloud_source/fused/{camera_index,pixel_uv}` to assign nearest patch token features to each sampled point.
  - writes `traj_*/obs/dino_feature` with DINO provenance metadata.
  - `SEMANTIC_FEATURE_DTYPE=float32` by default; `float16` can be used to reduce HDF5 size/write time. The training dataset casts loaded features to `float32`.
- `existing_dino`:
  - requires `traj_*/obs/dino_feature` to already exist.
  - validates that it is a per-point dataset with shape `[T_frame, P, D_sem]`.
  - validates that its first two dimensions match `obs/point_cloud/fused`.
  - validates that it carries DINO/DINOv3 provenance metadata.
  - does not generate RGB fallback features.

The model dataset reads:

- `obs.point_cloud <- traj_*/obs/point_cloud/fused`
- `obs.dino_feature <- traj_*/obs/dino_feature`

For a training sample with `n_obs_steps=2`, the batch tensor shapes are:

- `obs.point_cloud`: `[B, 2, P, 6]`
- `obs.dino_feature`: `[B, 2, P, D_sem]`

### Keyframe Sidecar HDF5

The raw keyframe sidecar is created by `helper/build_keyframe_aux_dataset.py`. It is aligned by `traj_*` name with the demo HDF5.

Per-trajectory datasets:

- `traj_*/map4d`: `[T_frame, N_target_node, 9]`
  - node pose layout: `pos(3) + rot6d(6)`
- `traj_*/size_parameters`: `[D_size]`
- `traj_*/relation_parameters`: `[D_relation]`
- `traj_*/tcp_pose`: `[T_frame, 7]`
- `traj_*/keyframe_indices`: `[N_keyframe]`
- `traj_*/future_keyframe_indices`: `[T_frame, H_key]`
- `traj_*/future_keyframe_object_targets`: `[T_frame, H_key, N_target_node, 9]`
- `traj_*/future_keyframe_tcp_pose`: `[T_frame, H_key, tcp_dim]`

Task-specific target settings:

- StackCube:
  - `tcp_target=pos_gripper`
  - target format: `map4d_dit_local_delta_relative_rotation_tcp_pos_gripper_v1`
  - `future_keyframe_tcp_pose`: `[T_frame, H_key, 4]`, `local_delta_pos(3) + gripper(1)`
- PlugCharger:
  - `tcp_target=pose`
  - target format: `map4d_dit_local_delta_relative_rotation_v1`
  - `future_keyframe_tcp_pose`: `[T_frame, H_key, 7]`, `local_delta_pos(3) + delta_quat(4)`

The model dataset reads:

- `keyframe.map4d <- future_keyframe_object_targets`
- `keyframe.tcp <- future_keyframe_tcp_pose`

`size_parameters` / `relation_parameters` are not read from maps4d JSON defaults. During keyframe sidecar construction, `helper/build_keyframe_aux_dataset.py` requires the ManiSkill metadata JSON next to the demo HDF5, creates the matching ManiSkill task env, resets it with each episode seed/options, and reads the task geometry from the env. Missing metadata, missing episode seeds, or missing env attributes are hard errors.

Current extraction:

- StackCube:
  - cube size: `2 * env.unwrapped.cube_half_size`
  - table size: `env.unwrapped.table_scene.table_length/width/height`
  - `relation_parameters`: empty `[0]`
- PlugCharger:
  - charger body: `2 * env.unwrapped._base_size`
  - charger prong: `2 * env.unwrapped._peg_size`
  - charger prong relation: `env.unwrapped._peg_gap`
  - receptacle primitive parameters are derived from `_receptacle_size`, `_peg_size`, `_peg_gap`, and `_clearance`

### Final GT-Map Sidecar HDF5

The final sidecar is created by `build_map4d_context_dataset.py`. It copies the raw keyframe sidecar and validates that the demo HDF5 contains `obs/point_cloud/fused` and `obs/dino_feature`.

It does not run `Map4DEncoder` and does not write `traj_*/map_node_feature`. The map encoder is part of `Map4DDiT` and is trained jointly with the denoising network.

Per-trajectory GT map datasets are copied from the raw keyframe sidecar:

- `traj_*/map4d`: `[T_frame, N_target_node, 9]`
- `traj_*/size_parameters`: `[D_size]`
- `traj_*/relation_parameters`: `[D_relation]`

The model dataset reads these as:

- `obs.node_poses`
- `obs.size_parameters`
- `obs.relation_parameters`

For a training sample with `n_map_step=2`, the batch tensor shapes are:

- `obs.node_poses`: `[B, 2, N_target_node, 9]`
- `obs.size_parameters`: `[B, D_size]` or `[D_size]`
- `obs.relation_parameters`: `[B, D_relation]` or `[D_relation]`

Inside `Map4DDiT`, `Map4DEncoder` converts this GT map into model context:

- online encoder output: `[B, 2, N_map_node, 3 + D_map]`
- first 3 channels: map node coordinate
- remaining channels: trainable map node feature
- default `D_map = MAP_FEATURE_DIM = 240`

`N_target_node` and `N_map_node` are allowed to be different:

- `N_target_node` is the number of keyframe node pose targets. It comes from maps4d task metadata.
- `N_map_node` is the number of context nodes emitted by online `Map4DEncoder`. It comes from maps4d task metadata and is written into the context summary JSON and manifest.

### Manifest And Summaries

The script writes a `.env` manifest next to the final context sidecar. It is the handoff from data construction to training.

Important manifest keys:

- `MAP4D_DEMO_PATH`: demo HDF5 path.
- `MAP4D_KEYFRAME_SIDECAR_PATH`: final GT-map context sidecar path.
- `MAP4D_RAW_KEYFRAME_SIDECAR_PATH`: raw keyframe sidecar path.
- `MAP4D_NUM_TRAJ`: number of trajectories.
- `SEMANTIC_FEATURE_DIM`: `D_sem`.
- `MAP_FEATURE_DIM`: `D_map`.
- `NUM_MAP_NODES`: `N_map_node`.
- `TASK_OVERRIDE`: Hydra task override, for example `task=stackcube_map4d_dit`.

The script also writes summary JSON files:

- `${demo_stem}.pointcloud.summary.json`
- `${demo_stem}.semantic_field_${SEMANTIC_FEATURE_MODE}.summary.json`
- `${keyframe_sidecar_stem}.summary.json`
- `${final_sidecar_stem}.summary.json`

Minimal training invocation:

```bash
source /path/to/final_context.env
MAP4D_DEMO_PATH="$MAP4D_DEMO_PATH" \
MAP4D_KEYFRAME_SIDECAR_PATH="$MAP4D_KEYFRAME_SIDECAR_PATH" \
MAP4D_NUM_TRAJ="$MAP4D_NUM_TRAJ" \
  bash scripts/map4d_backbone/run_map4d_dit_train.sh \
  --config-name map4d_dit "$TASK_OVERRIDE" \
  policy.model_cfg.semantic_feature_dim="$SEMANTIC_FEATURE_DIM" \
  policy.model_cfg.map_feature_dim="$MAP_FEATURE_DIM" \
  policy.model_cfg.num_map_nodes="$NUM_MAP_NODES"
```

### Planned Code Changes

- Remove the abandoned `DiffusionHeadMap4D` path and maintain `Map4DDiT` as the single Map4D DiT network.
- Replace `use_rgb` / `rgb_feature_dim` model options with Semantic Field options:
  - `use_semantic_field`
  - `semantic_feature_dim`
  - `point_dim`, default `3`
  - optional `max_semantic_points`
- Add `semantic_field_proj = MLP(point_dim + semantic_feature_dim -> D_model)` inside `Map4DDiT`.
- In `_context_tokens`, replace `obs.rgb_feature` handling with `obs.point_cloud` and `obs.dino_feature`.
- Add a map encoder module from `map4d/encoder/map_encoder.py` that produces `[B, N_node, 3 + D_map]`.
- Add `map_feature_proj = MLP(D_map -> D_model)` inside `Map4DDiT`.
- Build a unified 3D context field:
  - `context_xyz = concat(semantic_point_xyz, map_node_xyz)`
  - `context_token = concat(semantic_point_token, map_feature_token)`
- Add historical map-token support:
  - `n_map_step`
  - `map_time_pos_embed`
  - optional `map_node_pos_embed`
- Replace concat-all-token DiT blocks with PPI-style staged attention:
  - node branch: context -> keyframe node pose
  - tcp branch: context + keyframe node hidden -> keyframe tcp pose
  - action branch: context + keyframe node hidden + keyframe tcp hidden -> trajectory/action
- Preserve the dual-arm interface:
  - explicit `N_arm` axis for trajectory, tcp, and gripper targets
  - `arm_embed`
  - no hard-coded single-arm-only tensor assumptions
- In `_context_tokens`, replace direct raw map4d/size/relation token projection with projected map feature points in the unified 3D context field.
- Split `num_objects` into target and map concepts where needed:
  - `num_target_nodes` for keyframe node prediction
  - `num_map_nodes` for map encoder validation/configuration
- Update `ManiSkillMap4DDataset` to load:
  - `traj_*/obs/point_cloud/fused` as `obs.point_cloud`
  - `traj_*/obs/dino_feature` or compatible per-point semantic features as `obs.dino_feature`
- Update dataset or collate code to build the batched input required by `map4d/encoder/map_encoder.py`.
- Keep robot state conditioning, target tokens, losses, and schedulers unchanged.
