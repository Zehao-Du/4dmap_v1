#!/bin/bash
# DP + raw concat + aux future prediction, 100 demos
set -o pipefail
ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$DP_DIR"
mkdir -p outputs/train_logs

BASE_RUN_NAME="stackcube_pos_dp_map4d_raw_aux_100demos_seed1"
RUN_NAME="${BASE_RUN_NAME}"
LOG_FILE="outputs/train_logs/${RUN_NAME}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date +%H:%M:%S)] running $RUN_NAME -> $LOG_FILE"
python train_rgbd.py \
  --exp-name "${RUN_NAME}" \
  --seed 1 \
  --env-id StackCube-v1 \
  --demo-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.h5 \
  --num-demos 100 \
  --control-mode pd_ee_delta_pos \
  --sim-backend physx_cpu \
  --obs-mode rgb \
  --max-episode-steps 1000 \
  --total-iters 100000 \
  --batch-size 64 \
  --obs-horizon 2 \
  --act-horizon 8 \
  --pred-horizon 16 \
  --lr 0.0001 \
  --visual-encoder plain_conv \
  --use-map4d \
  --map4d-raw-concat \
  --map4d-aux-loss \
  --map4d-aux-weight 1.0 \
  --map4d-source maniskill_gt \
  --map4d-task-name StackCube-v1 \
  --map4d-pre-horizon 30 \
  --map4d-future-horizon 30 \
  --map4d-num-objects 3 \
  --log-freq 100 \
  --eval-freq 10000 \
  --num-eval-episodes 100 \
  --num-eval-envs 10 \
  --num-dataload-workers 0 \
  --no-track \
  --no-capture-video 2>&1 | tee "$LOG_FILE"
