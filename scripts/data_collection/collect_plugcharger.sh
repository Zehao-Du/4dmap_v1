#!/bin/bash
# Collect PlugCharger-v1 demos via motion planning + replay to pd_ee_delta_pose
# Only keeps trajectories <= 400 steps (filters out long retry trajectories)
# Reference: 4dmap_policy/README.md
set -e
ROOT_DIR=/data2/zehao/MAP4D
DATASET_DIR="$ROOT_DIR/dataset/ManiSkill/PlugCharger-v1/motionplanning"
RECORD_DIR="$ROOT_DIR/dataset/ManiSkill"
POLICY_DIR="$ROOT_DIR/4dmap_v1"

echo "Cleaning old data in $DATASET_DIR ..."
rm -f "$DATASET_DIR"/PlugCharger*.h5 "$DATASET_DIR"/PlugCharger*.json
mkdir -p "$DATASET_DIR"

TARGET_DEMOS=${1:-1000}
MAX_STEPS=400
NUM_PROCS=${2:-10}
# Collect extra to account for filtering (empirically ~65% pass the filter)
COLLECT_N=$(( TARGET_DEMOS * 2 ))

echo "[$(date +%H:%M:%S)] Step 1: Collecting $COLLECT_N PlugCharger-v1 trajectories (targeting $TARGET_DEMOS after filtering)..."
python -m mani_skill.examples.motionplanning.panda.run \
  -e PlugCharger-v1 \
  -n "$COLLECT_N" \
  --only-count-success \
  -b cpu \
  --traj-name PlugCharger \
  --record-dir "$RECORD_DIR" \
  --num-procs "$NUM_PROCS"

echo "[$(date +%H:%M:%S)] Step 2: Replaying to rgb + pd_ee_delta_pose..."
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$DATASET_DIR/PlugCharger.h5" \
  -o rgb \
  -c pd_ee_delta_pose \
  --no-verbose \
  --max-retry 3 \
  --no-allow-failure \
  --save-traj \
  -n 1

echo "[$(date +%H:%M:%S)] Step 3: Filtering trajectories > $MAX_STEPS steps..."
python -c "
import h5py, json, os, sys

src = '$DATASET_DIR/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.h5'
dst = '$DATASET_DIR/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5'
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
echo "Filtered dataset: $DATASET_DIR/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5"
ls -la "$DATASET_DIR"/PlugCharger*.filtered.*
