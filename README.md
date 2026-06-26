# 4D Map Policy

This repository contains the active 4D-map policy learning code: ManiSkill data
collection, Map4D context dataset construction, Map4DDiT training, and rollout
evaluation.

## Quick Start

Initialize submodules:

```bash
git submodule update --init --recursive
```

Use the `4dmap` conda environment:

```bash
conda activate 4dmap
```

Install project requirements manually from `requirements.txt`. The file is a
shell-oriented recipe, not a plain `pip install -r` file.

## ManiSkill

Use the vendored ManiSkill fork in `third_party/maniskill_map4d`. It includes
the camera resolution overrides needed by the data generation scripts.

If `mani_skill` was previously installed from pip, replace it with the local
editable install:

```bash
conda activate 4dmap
pip uninstall -y mani_skill
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/third_party/maniskill_map4d
pip install -e . --no-build-isolation
```

Verify that Python imports the local fork:

```bash
python -c "import mani_skill, inspect; print(inspect.getfile(mani_skill))"
```

Expected path:

```text
.../4dmap_policy/third_party/maniskill_map4d/mani_skill/__init__.py
```

The fork supports these resolution flags in both motion planning collection and
trajectory replay:

```bash
--image-size 224
--image-size 320x240
--camera-width 224 --camera-height 224
```

## Data Collection

Build a StackCube training dataset from scratch:

```bash
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh stackcube 1000 224 false
```

Arguments:

```text
<task> <demos> [resolution] [generate_dino_feature]
```

- `task`: `stackcube` or `plugcharger`
- `demos`: number of successful demos to collect
- `resolution`: `native`, `SIZE`, or `WIDTHxHEIGHT`; default is `224`
- `generate_dino_feature`: `true` to precompute per-point DINO features,
  `false` to build an online-DINO dataset manifest

For online-DINO training, use `false`:

```bash
bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh stackcube 1000 224 false
```

Important environment variables:

```bash
DATA_ROOT=/path/to/dataset/root
NUM_PROCS=10
DINOV3_WEIGHTS_PATH=/path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
DINOV3_THIRD_PARTY_DIR=/path/to/dinov3/source
SKIP_COLLECT=1
DEMO_PATH=/path/to/existing_demo.h5
```

The script writes a manifest next to the generated dataset, for example:

```text
dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb+depth.pd_ee_delta_pos.physx_cpu.filtered.map4d_dit_h4.context.env
```

Training scripts read this manifest and fail fast if required files or fields
are missing.

## Train Map4DDiT

Train StackCube with online DINO features:

```bash
ONLINE_DINO=1 \
DINOV3_THIRD_PARTY_DIR=/data2/zehao/MAP4D/4dmap_v1/map4d/backbone/model/vision/dinov3 \
bash scripts/map4d_experiments/train_stackcube_map4d_dit.sh 1000
```

Use shared memory to stage the read-heavy HDF5 files before training:

```bash
SHM_ROOT=/dev/shm/4dmap_stackcube_dit \
ONLINE_DINO=1 \
DINOV3_THIRD_PARTY_DIR=/data2/zehao/MAP4D/4dmap_v1/map4d/backbone/model/vision/dinov3 \
bash scripts/map4d_experiments/train_stackcube_map4d_dit_shm.sh 1000
```

The shared-memory script copies the demo HDF5 and keyframe sidecar HDF5 into
`SHM_ROOT`, rewrites a temporary manifest to point at those files, and then
calls `train_stackcube_map4d_dit.sh`. It does not fall back to disk paths when
required files are missing.

### Multi-GPU

Use `torchrun` through the script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
BATCH_SIZE=32 \
ONLINE_DINO=1 \
DINOV3_THIRD_PARTY_DIR=/data2/zehao/MAP4D/4dmap_v1/map4d/backbone/model/vision/dinov3 \
bash scripts/map4d_experiments/train_stackcube_map4d_dit_shm.sh 1000
```

`BATCH_SIZE` is per GPU. For example, `NPROC_PER_NODE=4 BATCH_SIZE=32` gives a
global batch size of 128.

The DDP config defaults to:

```yaml
training.ddp_find_unused_parameters: true
```

This is needed for the current online-DINO Map4DDiT graph, where some
`requires_grad=True` parameters are not used by every loss path.

### Common Overrides

```bash
BATCH_SIZE=16
NUM_WORKERS=8
NUM_EPOCHS=2000
ROLLOUT_ENABLED=false
ROLLOUT_NUM_EVAL_ENVS=1
ROLLOUT_NUM_EVAL_EPISODES=5
ROLLOUT_EVERY=100
CHECKPOINT_SAVE_CKPT=false
MASTER_PORT=29570
MANIFEST_PATH=/path/to/context.env
```

Hydra overrides can also be appended after `<num_demos>`:

```bash
ONLINE_DINO=1 bash scripts/map4d_experiments/train_stackcube_map4d_dit.sh 1000 \
  rollout.enabled=false training.max_train_steps=10 training.max_val_steps=2
```

## Roadmap

TODO: validate 4D map representation and losses in **ManiSkill** on
**StackCube-v1** first.

- [x] 4D map representation, including losses (`map4d/representation/maps4d`)
- [x] 4D map construction (`map4d/construction`)
- [ ] 4D map encoder (`map4d/encoder`)
- [ ] Insert into baselines, such as diffusion policy and ACT

StackCube 的 4D map representation 已放在
`map4d/representation/maps4d/maniskill_stackcube.py`，基于 `Map_4d` 保存
scene-level Objects，并用 red cube / green cube / desk 的 Cuboid 节点描述结构参数。
Construction 入口在 `map4d/construction/map_constructor.py`，当前流程是 RGB-D ->
Grounded-SAM2/manual masks -> structural parameter estimator -> FoundationPose
poses -> instantiated Map4d，后续 encoder 开发可直接通过
`map4d/map4d_encoder.py` 的 `Map4d_Encoder.construction(...)` 调用。
