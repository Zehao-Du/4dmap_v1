#!/bin/bash
# Launch 4 controlled DP experiments on StackCube-v1 pd_ee_delta_pos demos.
# By default runs all 4 sequentially in foreground.
# Pass "parallel" as $1 to background them (requires job scheduler that doesn't kill children).
set -o pipefail

ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$DP_DIR"
mkdir -p outputs/train_logs

CONFS=(
  stackcube_pos_dp_baseline
  stackcube_pos_dp_dinov3
  stackcube_pos_dp_map4d
  stackcube_pos_dp_dinov3_map4d
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
