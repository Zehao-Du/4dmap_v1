#!/bin/bash
# Evaluate ACT + 4D map checkpoint with GT map4d from environment
set -o pipefail
ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
ACT_DIR="$ROOT_DIR/baselines/act"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$ACT_DIR"

CHECKPOINT="${1:-runs/stackcube_pos_act_map4d_990demos_seed1/checkpoints/best_eval_success_once.pt}"
DEMO_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.h5"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_NAME="act_map4d_eval_${RUN_TIMESTAMP}"
LOG_FILE="outputs/eval_logs/${EVAL_NAME}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date +%H:%M:%S)] Evaluating ACT+map4d: $CHECKPOINT -> $LOG_FILE"
python eval_map4d.py \
  --checkpoint "$CHECKPOINT" \
  --demo-path "$DEMO_PATH" \
  --num-demos 990 \
  --env-id StackCube-v1 \
  --control-mode pd_ee_delta_pos \
  --sim-backend physx_cpu \
  --no-include-depth \
  --max-episode-steps 1000 \
  --num-eval-episodes 100 \
  --num-eval-envs 10 \
  --num-queries 30 \
  --temporal-agg \
  --use-map4d \
  --map4d-source maniskill_gt \
  --map4d-task-name StackCube-v1 \
  --map4d-pre-horizon 6 \
  --map4d-future-horizon 3 \
  --map4d-num-objects 3 \
  --map4d-feature-dim 128 \
  --map4d-node-dim 128 \
  --map4d-relation-dim 64 \
  --map4d-temporal-dim 128 2>&1 | tee "$LOG_FILE"
