#!/bin/bash
# Fast ACT + Map4D smoke test using the datasets that exist in this workspace.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy}"
DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset}"
ACT_DIR="$ROOT_DIR/baselines/act"

TASK_NAME="${TASK_NAME:-PlugCharger-v1}"
NUM_DEMOS="${NUM_DEMOS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_ITERS="${TOTAL_ITERS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

case "$TASK_NAME" in
  StackCube-v1)
    DEMO_PATH="${DEMO_PATH:-$DATA_ROOT/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb+depth.pd_ee_delta_pos.physx_cpu.h5}"
    CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pos}"
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
    MAP4D_NUM_OBJECTS="${MAP4D_NUM_OBJECTS:-3}"
    ;;
  PlugCharger-v1)
    DEMO_PATH="${DEMO_PATH:-$DATA_ROOT/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb+depth.pd_ee_delta_pose.physx_cpu.h5}"
    CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-400}"
    MAP4D_NUM_OBJECTS="${MAP4D_NUM_OBJECTS:-2}"
    ;;
  *)
    echo "Unsupported TASK_NAME=$TASK_NAME" >&2
    exit 2
    ;;
esac

JSON_PATH="${DEMO_PATH%.h5}.json"
if [[ ! -f "$DEMO_PATH" ]]; then
  echo "Demo file not found: $DEMO_PATH" >&2
  exit 1
fi
if [[ ! -f "$JSON_PATH" ]]; then
  echo "Demo JSON sidecar not found: $JSON_PATH" >&2
  exit 1
fi

cd "$ACT_DIR"
mkdir -p outputs/train_logs

RUN_NAME="${RUN_NAME:-act_map4d_smoke_${TASK_NAME}_cuda${CUDA_VISIBLE_DEVICES}}"
RUN_NAME="${RUN_NAME//[^A-Za-z0-9_.-]/_}"
LOG_FILE="${LOG_FILE:-outputs/train_logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "$LOG_FILE")"

TRAIN_ARGS=(
  --exp-name "$RUN_NAME"
  --seed 1
  --env-id "$TASK_NAME"
  --demo-path "$DEMO_PATH"
  --num-demos "$NUM_DEMOS"
  --control-mode "$CONTROL_MODE"
  --sim-backend physx_cpu
  --no-include-depth
  --max-episode-steps "$MAX_EPISODE_STEPS"
  --total_iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --num-queries 30
  --kl-weight 10
  --lr 1e-4
  --lr-backbone 1e-5
  --log_freq 1
  --eval_freq 10000
  --num-eval-episodes 1
  --num-eval-envs 1
  --num-dataload-workers 0
  --no-track
  --no-capture-video
  --no-eval
  --no-backbone-pretrained
  --use-map4d
  --map4d-raw-concat
  --map4d-source maniskill_gt
  --map4d-task-name "$TASK_NAME"
  --map4d-pre-horizon 6
  --map4d-future-horizon 3
  --map4d-num-objects "$MAP4D_NUM_OBJECTS"
  --map4d-feature-dim 128
  --map4d-node-dim 128
  --map4d-relation-dim 64
  --map4d-temporal-dim 128
)

echo "[$(date +%H:%M:%S)] TASK_NAME=$TASK_NAME CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$(date +%H:%M:%S)] demo=$DEMO_PATH"
echo "[$(date +%H:%M:%S)] running $RUN_NAME -> $LOG_FILE"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'cd %s\npython train_rgbd_map4d.py' "$ACT_DIR"
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

python train_rgbd_map4d.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
