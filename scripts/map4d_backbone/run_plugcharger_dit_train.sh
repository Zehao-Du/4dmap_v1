#!/usr/bin/env bash
# Train Map4D DiT backbone on PlugCharger-v1 with DINOv3 features.
# Requires: keyframe sidecar + DINOv3 features already merged into sidecar.
#
# Usage:
#   bash scripts/map4d_backbone/run_plugcharger_dit_train.sh [NUM_TRAJ] [SEED]
#
# Prerequisites:
#   1. bash scripts/data_collection/extract_dinov3_features.sh PlugCharger-v1
#   2. bash scripts/data_collection/build_map4d_dit_dataset.sh  (with TASK_NAME=PlugCharger-v1)
#   3. Merge DINOv3 features into sidecar (see below)
#
# To merge DINOv3 features into sidecar:
#   python -c "
#   import h5py, numpy as np
#   with h5py.File('...filtered.dinov3_s16.h5','r') as fd, h5py.File('...filtered.map4d_dit_h4.h5','a') as fs:
#       for k in fd.keys():
#           if k in fs:
#               if 'rgb_feature' in fs[k]: del fs[k]['rgb_feature']
#               fs[k].create_dataset('rgb_feature', data=fd[k]['rgb_feature'][()], dtype=np.float32)
#   "
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap}"
POLICY_DIR="${ROOT_DIR}/4dmap_policy"
NUM_TRAJ="${1:-${NUM_TRAJ:-1000}}"
SEED="${2:-${SEED:-0}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

DEMO_PATH="${DEMO_PATH:-${ROOT_DIR}/dataset/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5}"
SIDECAR_PATH="${SIDECAR_PATH:-${DEMO_PATH%.h5}.map4d_dit_h4.h5}"

if [[ ! -f "$DEMO_PATH" ]]; then
  echo "Demo file not found: $DEMO_PATH" >&2
  exit 1
fi
if [[ ! -f "$SIDECAR_PATH" ]]; then
  echo "Sidecar not found: $SIDECAR_PATH" >&2
  echo "Build it with: TASK_NAME=PlugCharger-v1 bash scripts/data_collection/build_map4d_dit_dataset.sh" >&2
  exit 1
fi

RUN_NAME="map4d_dit_plugcharger_${NUM_TRAJ}demos_seed${SEED}"
LOG_FILE="${POLICY_DIR}/outputs/map4d_backbone_train_logs/${RUN_NAME}.log"

cd "$POLICY_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date +%H:%M:%S)] Training Map4D DiT on PlugCharger-v1"
echo "  num_traj: $NUM_TRAJ"
echo "  seed: $SEED"
echo "  demo: $DEMO_PATH"
echo "  sidecar: $SIDECAR_PATH"
echo "  log: $LOG_FILE"

MAP4D_DEMO_PATH="$DEMO_PATH" \
MAP4D_KEYFRAME_SIDECAR_PATH="$SIDECAR_PATH" \
MAP4D_NUM_TRAJ="$NUM_TRAJ" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
torchrun --nproc_per_node="$NPROC_PER_NODE" \
  map4d/backbone/train_map4d_dit.py \
  --config-name map4d_dit \
  task=plugcharger_map4d_dit \
  "seed=${SEED}" \
  "addition_info=plugcharger_${NUM_TRAJ}demos" \
  "task.dataset.num_traj=${NUM_TRAJ}" \
  "policy.model_cfg.use_rgb=true" \
  "policy.model_cfg.rgb_feature_dim=384" \
  "training.debug=false" \
  "logging.mode=${WANDB_MODE}" \
  2>&1 | tee "$LOG_FILE"
