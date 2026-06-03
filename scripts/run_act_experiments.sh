#!/bin/bash
# Run ACT experiments on StackCube-v1: baseline (100/990 demos) + map4d (100 demos).
set -o pipefail

ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
ACT_DIR="$ROOT_DIR/baselines/act"

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$ACT_DIR"
mkdir -p outputs/train_logs

# --- ACT baseline 100 demos ---
source "$ACT_DIR/configs/stackcube_pos_act_100demos.conf"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$(date +%H:%M:%S)] running ACT baseline 100 demos -> $LOG_FILE"
python train_rgbd.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# --- ACT + map4d 100 demos ---
source "$ACT_DIR/configs/stackcube_pos_act_map4d_100demos.conf"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$(date +%H:%M:%S)] running ACT + map4d 100 demos -> $LOG_FILE"
python train_rgbd_map4d.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# --- ACT baseline 990 demos ---
source "$ACT_DIR/configs/stackcube_pos_act_1000demos.conf"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$(date +%H:%M:%S)] running ACT baseline 990 demos -> $LOG_FILE"
python train_rgbd.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
