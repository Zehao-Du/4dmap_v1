#!/bin/bash
# Collect StackCube-v1 demos via motion planning + replay to pd_ee_delta_pos
# Only keeps trajectories <= 400 steps (filters out long retry trajectories)
# Reference: 4dmap_policy/README.md
set -e
ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dataset/ManiSkill/StackCube-v1/motionplanning}"
RECORD_DIR="${RECORD_DIR:-$ROOT_DIR/dataset/ManiSkill}"
POLICY_DIR="${POLICY_DIR:-$ROOT_DIR/4dmap_policy}"

echo "Cleaning old data in $DATASET_DIR ..."
rm -f "$DATASET_DIR"/StackCube*.h5 "$DATASET_DIR"/StackCube*.json
mkdir -p "$DATASET_DIR"

TARGET_DEMOS=${1:-1000}
MAX_STEPS=400
NUM_PROCS=${2:-10}
IMAGE_SIZE=${3:-224}
OBS_MODE="${OBS_MODE:-rgb+depth}"
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-$NUM_PROCS}"
# Collect extra to account for filtering (empirically ~65% pass the filter)
COLLECT_N=$(( TARGET_DEMOS * 2 ))
PATCH_CAMERA_CONFIG=0
if [[ -n "$IMAGE_SIZE" && "$IMAGE_SIZE" != "native" ]]; then
  if [[ "$IMAGE_SIZE" =~ ^[0-9]+$ ]]; then
    CAMERA_WIDTH="$IMAGE_SIZE"
    CAMERA_HEIGHT="$IMAGE_SIZE"
  elif [[ "$IMAGE_SIZE" =~ ^[0-9]+x[0-9]+$ ]]; then
    CAMERA_WIDTH="${IMAGE_SIZE%x*}"
    CAMERA_HEIGHT="${IMAGE_SIZE#*x}"
  else
    echo "Invalid IMAGE_SIZE=$IMAGE_SIZE. Use native, SIZE, or WIDTHxHEIGHT." >&2
    exit 1
  fi
  PATCH_CAMERA_CONFIG=1
fi

echo "[$(date +%H:%M:%S)] Step 1: Collecting $COLLECT_N StackCube-v1 trajectories (targeting $TARGET_DEMOS after filtering)..."
python -m mani_skill.examples.motionplanning.panda.run \
  -e StackCube-v1 \
  -n "$COLLECT_N" \
  --only-count-success \
  -b cpu \
  --traj-name StackCube \
  --record-dir "$RECORD_DIR" \
  --num-procs "$NUM_PROCS" \
  --image-size "$IMAGE_SIZE"

if [[ "$PATCH_CAMERA_CONFIG" == "1" ]]; then
  python -c "
import json
path = '$DATASET_DIR/StackCube.json'
with open(path, 'r') as f:
    data = json.load(f)
env_kwargs = data.setdefault('env_info', {}).setdefault('env_kwargs', {})
sensor_configs = env_kwargs.setdefault('sensor_configs', {})
sensor_configs['width'] = int('$CAMERA_WIDTH')
sensor_configs['height'] = int('$CAMERA_HEIGHT')
with open(path, 'w') as f:
    json.dump(data, f)
"
fi

echo "[$(date +%H:%M:%S)] Step 2: Replaying to ${OBS_MODE} + pd_ee_delta_pos..."
echo "  replay_num_envs: $REPLAY_NUM_ENVS"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$DATASET_DIR/StackCube.h5" \
  -o "$OBS_MODE" \
  -c pd_ee_delta_pos \
  --image-size "$IMAGE_SIZE" \
  --no-verbose \
  --max-retry 3 \
  --no-allow-failure \
  --save-traj \
  -n "$REPLAY_NUM_ENVS"

echo "[$(date +%H:%M:%S)] Step 3: Filtering trajectories > $MAX_STEPS steps..."
python -c "
import h5py, json, os, sys

src = '$DATASET_DIR/StackCube.$OBS_MODE.pd_ee_delta_pos.physx_cpu.h5'
dst = '$DATASET_DIR/StackCube.$OBS_MODE.pd_ee_delta_pos.physx_cpu.filtered.h5'
max_steps = $MAX_STEPS
target = $TARGET_DEMOS

with h5py.File(src, 'r') as f_in, h5py.File(dst, 'w') as f_out:
    keys = sorted([k for k in f_in.keys() if k.startswith('traj')], key=lambda x: int(x.split('_')[-1]))
    kept = 0
    for k in keys:
        if f_in[k]['actions'].shape[0] <= max_steps:
            f_in.copy(f_in[k], f_out, name=f'traj_{kept}')
            kept += 1
            if kept >= target:
                break
    print(f'Kept {kept}/{len(keys)} trajectories (<={max_steps} steps)')
    if kept < target:
        print(f'WARNING: only got {kept} trajectories, wanted {target}. Re-run with more collections.')
        sys.exit(1)

json_src = src.replace('.h5', '.json')
json_dst = dst.replace('.h5', '.json')
if os.path.exists(json_src):
    with open(json_src, 'r') as f:
        meta = json.load(f)
    if 'episodes' in meta:
        meta['episodes'] = meta['episodes'][:kept]
    with open(json_dst, 'w') as f:
        json.dump(meta, f)
"

echo "[$(date +%H:%M:%S)] Done."
echo "Filtered dataset: $DATASET_DIR/StackCube.$OBS_MODE.pd_ee_delta_pos.physx_cpu.filtered.h5"
ls -la "$DATASET_DIR"/StackCube*.filtered.*
