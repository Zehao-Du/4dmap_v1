# RLBench2 Map4D DiT Pipeline

This document describes the current RLBench2 `bimanual_push_box` pipeline for
training the Map4D DiT backbone in `4dmap_policy`.

The current wired task is:

```text
RLBench2 task: bimanual_push_box
Hydra task:    task=rlbench2_push_box_map4d_dit
dataset root:  dataset/rlbench2/map4d_dit/bimanual_push_box
```

## Environment

Use the `4dmap` conda environment for all Python commands.

For local commands:

```bash
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
conda activate 4dmap
```

For copied job commands, do not rely on `~/.bashrc`. Wrap the actual command in
`/bin/bash -lc`, activate conda explicitly, and export CoppeliaSim paths
explicitly.

## Dataset Builder

The builder is:

```text
scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh
```

Default push-box command:

```bash
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy

bash scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh \
  dataset/rlbench2/bimanual_push_box.train.squashfs \
  dataset/rlbench2/map4d_dit \
  rgb_pcd_rps6144
```

With this input mode, the builder writes to a task-named subdirectory:

```text
dataset/rlbench2/map4d_dit/bimanual_push_box
```

The builder does the following:

1. Reads the RLBench2 `.train.squashfs`.
2. Builds or reuses raw RLBench2 episodes under `raw/all_variations/episodes`.
3. Builds point clouds under `point_cloud`.
4. Builds DINO-style semantic features under `dino_feature`.
5. Extracts push-box Map4D poses into `.npz` and `.csv` sidecars.
6. Writes a `.env` manifest for training.
7. Runs a dataset smoke check unless `RUN_SMOKE=0`.

The visual feature builder has `tqdm` as a hard dependency. There is no fallback:
missing dependencies or missing data should fail loudly.

## Output Layout

For push box, the expected directory is:

```text
dataset/rlbench2/map4d_dit/bimanual_push_box/
  raw/all_variations/episodes/
  point_cloud/
    episode*/rgb_pcd_rps6144/step*.npy
  dino_feature/
    episode*/rgb_pcd_rps6144/step*.npy
  bimanual_push_box_train_poses.npz
  bimanual_push_box_train_poses.csv
  rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.env
  rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.summary.txt
```

Current validated push-box dataset values:

```text
episodes:             100
pose frames:          14099
point cloud files:    14099
dino feature files:   14099
point cloud shape:    (6144, 6)
dino feature shape:   (6144, 288)
box size xyz:         [0.192, 0.384, 0.128]
```

## Manifest

The training manifest is:

```text
dataset/rlbench2/map4d_dit/bimanual_push_box/rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.env
```

It provides the training paths and shape metadata:

```bash
TASK_OVERRIDE=task=rlbench2_push_box_map4d_dit
RLBENCH2_DATA_PATH=...
RLBENCH2_POSE_PATH=...
RLBENCH2_PCD_PATH=...
RLBENCH2_DINO_PATH=...
RLBENCH2_LANG_EMB_PATH=...
RLBENCH2_PCD_TYPE=rgb_pcd_rps6144
SEMANTIC_FEATURE_DIM=288
MAP_FEATURE_DIM=240
NUM_MAP_NODES=1
POINTCLOUD_NUM_POINTS=6144
```

Source it with `set -a` so variables are exported for Hydra:

```bash
set -a
source dataset/rlbench2/map4d_dit/bimanual_push_box/rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.env
set +a
```

## Dataset Class

The training dataset implementation is:

```text
map4d/backbone/dataset/rlbench2_map4d_dataset.py
```

It is configured by:

```text
map4d/backbone/config/task/rlbench2_push_box_map4d_dit.yaml
```

The batch format consumed by Map4D DiT is:

```text
obs.robot_state            [B, 2, 16]
obs.point_cloud            [B, 2, 6144, 6]
obs.dino_feature           [B, 2, 6144, 288]
obs.node_position          [B, 2, 1, 3]
obs.node_rotation          [B, 2, 1, 4]
obs.size_parameters        [B, 3] or [3]
obs.relation_parameters    zero-dimensional for push box

action.trajectory          [B, 2, 50, 7]
action.gripper_openness    [B, 2, 50, 1]

keyframe.map4d             [B, 4, 1, 7]
keyframe.tcp               [B, 2, 4, 7]
```

For push box:

```text
n_obs_steps = 2
horizon_action = 50
horizon_keyframe = 4
num_map_nodes = 1
num_arms = 2
trajectory_dim = 7
keyframe_tcp_dim = 7
```

## Training

Base training command:

```bash
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy

set -a
source dataset/rlbench2/map4d_dit/bimanual_push_box/rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.env
set +a

python map4d/backbone/train_map4d_dit.py \
  --config-name map4d_dit "${TASK_OVERRIDE}" \
  policy.model_cfg.semantic_feature_dim="${SEMANTIC_FEATURE_DIM}" \
  policy.model_cfg.map_feature_dim="${MAP_FEATURE_DIM}" \
  policy.model_cfg.num_map_nodes="${NUM_MAP_NODES}" \
  dataloader.batch_size=128 \
  val_dataloader.batch_size=128
```

Default training config:

```text
map4d/backbone/config/map4d_dit.yaml
```

Important defaults:

```text
batch_size:                 128
num_train_timesteps:        1000
num_inference_steps:        1000
optimizer:                  AdamW, lr=1e-4
use_ema:                    true
trajectory_loss_weight:     30.0
keyframe_tcp_loss_weight:   1.0
keyframe_map4d_loss_weight: 0.3
gripper_loss_weight:        30.0
```

## Job Format

The copy-paste job examples live in:

```text
scripts/job
```

Dataset build job:

```bash
#!/bin/bash

/bin/bash -lc '
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy \
&& source /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/miniconda3/etc/profile.d/conda.sh \
&& conda activate 4dmap \
&& bash scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh \
dataset/rlbench2/bimanual_push_box.train.squashfs \
dataset/rlbench2/map4d_dit rgb_pcd_rps6144
'
```

Training job:

```bash
#!/bin/bash

/bin/bash -lc '
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy \
&& source /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/miniconda3/etc/profile.d/conda.sh \
&& conda activate 4dmap \
&& export COPPELIASIM_ROOT=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/codes/CoppeliaSim \
&& export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
&& export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}" \
&& set -a \
&& source dataset/rlbench2/map4d_dit/bimanual_push_box/rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.env \
&& set +a \
&& python map4d/backbone/train_map4d_dit.py \
--config-name map4d_dit "${TASK_OVERRIDE}" \
policy.model_cfg.semantic_feature_dim="${SEMANTIC_FEATURE_DIM}" \
policy.model_cfg.map_feature_dim="${MAP_FEATURE_DIM}" \
policy.model_cfg.num_map_nodes="${NUM_MAP_NODES}" \
dataloader.batch_size=128 \
val_dataloader.batch_size=128
'
```

The explicit CoppeliaSim exports are required because job shells may not source
`~/.bashrc`, and an outer `sh` shell will not understand `source` or bash-only
syntax.

## DiT Architecture Switches

The original Map4D DiT architecture is preserved as the default.

Default behavior:

```yaml
policy:
  model_cfg:
    separate_map_cross_attn: false
```

In this mode, semantic point tokens and map node tokens are concatenated into a
single cross-attention context:

```text
query -> cross_attn([semantic point tokens + map node tokens])
```

For push box, this means the map tokens are very sparse:

```text
semantic tokens: 2 * 6144 = 12288
map tokens:      2 * 1    = 2
```

To test the new map-aware architecture, explicitly enable:

```bash
policy.model_cfg.separate_map_cross_attn=true
```

With this enabled, node, TCP, and action stages each get a separate map
cross-attention path before attending to semantic point tokens:

```text
node   -> map cross-attn -> semantic cross-attn -> self-attn
tcp    -> map cross-attn -> semantic cross-attn -> self-attn
action -> map cross-attn -> semantic cross-attn -> self-attn
```

This keeps map tokens from competing directly with thousands of point tokens.
The old architecture remains the default so existing training runs and old
checkpoints are not changed by default.

Training command for the new architecture:

```bash
python map4d/backbone/train_map4d_dit.py \
  --config-name map4d_dit "${TASK_OVERRIDE}" \
  policy.model_cfg.semantic_feature_dim="${SEMANTIC_FEATURE_DIM}" \
  policy.model_cfg.map_feature_dim="${MAP_FEATURE_DIM}" \
  policy.model_cfg.num_map_nodes="${NUM_MAP_NODES}" \
  policy.model_cfg.separate_map_cross_attn=true \
  dataloader.batch_size=128 \
  val_dataloader.batch_size=128
```

## Smoke Checks

The builder runs a dataset smoke check by default. A successful push-box sample
should contain shapes like:

```text
obs.point_cloud              (2, 6144, 6)
obs.dino_feature             (2, 6144, 288)
action.trajectory            (2, 50, 7)
action.gripper_openness      (2, 50, 1)
keyframe.map4d               (4, 1, 7)
keyframe.tcp                 (2, 4, 7)
```

For a very small training smoke, restrict the environment variables before
sourcing the manifest or override the dataset range:

```bash
RLBENCH2_END=0 RLBENCH2_MAX_TRAIN_EPISODES=1 \
python map4d/backbone/train_map4d_dit.py \
  --config-name map4d_dit "${TASK_OVERRIDE}" \
  policy.model_cfg.semantic_feature_dim="${SEMANTIC_FEATURE_DIM}" \
  policy.model_cfg.map_feature_dim="${MAP_FEATURE_DIM}" \
  policy.model_cfg.num_map_nodes="${NUM_MAP_NODES}" \
  training.max_train_steps=1 \
  training.max_val_steps=1 \
  dataloader.batch_size=2 \
  val_dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0
```

## Common Failures

`source: not found` or `[[: not found`:

```text
The job was run by sh instead of bash.
Use /bin/bash -lc '...'.
```

`libcoppeliaSim.so.1: cannot open shared object file`:

```text
CoppeliaSim is not in LD_LIBRARY_PATH.
Export COPPELIASIM_ROOT and LD_LIBRARY_PATH inside the job command.
Do not rely on ~/.bashrc.
```

Missing point cloud or DINO feature files:

```text
The visual feature builder did not finish or was pointed at the wrong output dir.
Check RLBENCH2_PCD_PATH, RLBENCH2_DINO_PATH, and RLBENCH2_PCD_TYPE.
```

Missing pose sidecar:

```text
For bimanual_push_box, bimanual_push_box_train_poses.npz is required.
The dataset config uses allow_missing_pose: false.
```

Wrong semantic feature dimension:

```text
The model expects SEMANTIC_FEATURE_DIM from the manifest.
For the current push-box dataset this is 288.
```
