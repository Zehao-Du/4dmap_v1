#!/bin/bash
# Launch 4 controlled DP V2 experiments on StackCube-v1 pd_ee_delta_pos demos.
# V2 changes: rotation 6D fix, per-object MLP future heads, no pointcloud loss, map4d_pre_horizon=6.
set -o pipefail

ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$DP_DIR"
mkdir -p outputs/train_logs

CONFS=(
  stackcube_pos_v2_dp_map4d
  stackcube_pos_v2_dp_dinov3_map4d
  stackcube_pos_v2_dp_dinov3
)

MODE="${1:-sequential}"

if [ "$MODE" = "parallel" ]; then
  for conf in "${CONFS[@]}"; do
    source "$DP_DIR/configs/${conf}.conf"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date +%H:%M:%S)] launching $conf -> $LOG_FILE"
    python train_rgbd.py "${TRAIN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    sleep 30
  done
  echo "all 4 launched, waiting..."
  wait
  echo "all 4 finished."
else
  for conf in "${CONFS[@]}"; do
    source "$DP_DIR/configs/${conf}.conf"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date +%H:%M:%S)] running $conf -> $LOG_FILE"
    python train_rgbd.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
  done
fi
