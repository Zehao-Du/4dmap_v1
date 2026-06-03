#!/bin/bash
# DP + DINOv3 + 4D map, 100 demos
set -o pipefail
ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$DP_DIR"
mkdir -p outputs/train_logs
source "$DP_DIR/configs/stackcube_pos_dp_dinov3_map4d.conf"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$(date +%H:%M:%S)] running $RUN_NAME -> $LOG_FILE"
python train_rgbd.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
