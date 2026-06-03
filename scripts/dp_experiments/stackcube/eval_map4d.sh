#!/bin/bash
# Evaluate DP + 4D map checkpoint with GT map4d from environment
set -o pipefail
ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$DP_DIR"

CHECKPOINT="${1:-runs/stackcube_pos_dp_map4d_seed1/checkpoints/370000.pt}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_NAME="dp_map4d_eval_${RUN_TIMESTAMP}"
LOG_FILE="outputs/eval_logs/${EVAL_NAME}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date +%H:%M:%S)] Evaluating DP+map4d: $CHECKPOINT -> $LOG_FILE"
python eval_map4d.py \
  --checkpoint "$CHECKPOINT" \
  --env-id StackCube-v1 \
  --control-mode pd_ee_delta_pos \
  --sim-backend physx_cpu \
  --obs-mode rgb \
  --max-episode-steps 1000 \
  --num-eval-episodes 100 \
  --num-eval-envs 10 \
  --obs-horizon 2 \
  --act-horizon 8 \
  --pred-horizon 16 \
  --use-map4d \
  --map4d-source maniskill_gt \
  --map4d-task-name StackCube-v1 \
  --map4d-future-horizon 3 \
  --map4d-num-objects 3 \
  --map4d-feature-dim 128 \
  --map4d-node-dim 128 \
  --map4d-relation-dim 64 \
  --map4d-temporal-dim 128 2>&1 | tee "$LOG_FILE"
