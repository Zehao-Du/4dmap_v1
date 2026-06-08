#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <task[,task...]> <num_demo[,num_demo...]>" >&2
  echo "Example: $0 stackcube,plugcharger 100,990" >&2
  echo "Tasks: stackcube, plugcharger, all" >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/data2/zehao/MAP4D/4dmap_v1}"
DATA_ROOT="${DATA_ROOT:-/data2/zehao/MAP4D/dataset}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

split_csv() {
  local raw="$1"
  raw="${raw// /}"
  IFS=',' read -r -a SPLIT_RESULT <<< "$raw"
}

task_config() {
  local task="$1"
  case "${task,,}" in
    all)
      echo "stackcube plugcharger"
      ;;
    stackcube|stackcube-v1)
      echo "stackcube"
      ;;
    plugcharger|plugcharger-v1)
      echo "plugcharger"
      ;;
    *)
      echo "Unsupported task: ${task}. Use stackcube, plugcharger, or all." >&2
      return 1
      ;;
  esac
}

task_env() {
  local task="$1"
  case "$task" in
    stackcube)
      TASK_OVERRIDE="task=stackcube_map4d_dit"
      TASK_NAME="StackCube-v1"
      DEMO_PATH="${STACKCUBE_DEMO_PATH:-${DATA_ROOT}/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.filtered.h5}"
      SIDECAR_PATH="${STACKCUBE_MAP4D_DIT_SIDECAR_PATH:-${DEMO_PATH%.h5}.map4d_dit_h${FUTURE_HORIZON}.h5}"
      ;;
    plugcharger)
      TASK_OVERRIDE="task=plugcharger_map4d_dit"
      TASK_NAME="PlugCharger-v1"
      DEMO_PATH="${PLUGCHARGER_DEMO_PATH:-${DATA_ROOT}/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5}"
      SIDECAR_PATH="${PLUGCHARGER_MAP4D_DIT_SIDECAR_PATH:-${DEMO_PATH%.h5}.map4d_dit_h${FUTURE_HORIZON}.h5}"
      ;;
    *)
      echo "Internal error: unknown normalized task ${task}" >&2
      return 1
      ;;
  esac
}

split_csv "$1"
RAW_TASKS=("${SPLIT_RESULT[@]}")
split_csv "$2"
NUM_DEMOS=("${SPLIT_RESULT[@]}")

TASKS=()
for raw_task in "${RAW_TASKS[@]}"; do
  normalized="$(task_config "$raw_task")"
  read -r -a normalized_tasks <<< "$normalized"
  TASKS+=("${normalized_tasks[@]}")
done

for num_demo in "${NUM_DEMOS[@]}"; do
  if [[ ! "$num_demo" =~ ^[0-9]+$ ]]; then
    echo "Invalid num_demo: ${num_demo}. Use integers, e.g. 100,990." >&2
    exit 1
  fi
done

cd "$ROOT_DIR"
mkdir -p outputs/map4d_backbone_train_logs

for task in "${TASKS[@]}"; do
  task_env "$task"

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ ! -f "$DEMO_PATH" ]]; then
      echo "Demo file not found for ${TASK_NAME}: ${DEMO_PATH}" >&2
      exit 1
    fi
    if [[ ! -f "$SIDECAR_PATH" ]]; then
      echo "Keyframe sidecar not found for ${TASK_NAME}: ${SIDECAR_PATH}" >&2
      echo "Build it with:" >&2
      echo "  TASK_NAME=${TASK_NAME} FUTURE_HORIZON=${FUTURE_HORIZON} bash scripts/data_collection/build_map4d_dit_dataset.sh" >&2
      exit 1
    fi
  fi

  for num_demo in "${NUM_DEMOS[@]}"; do
    RUN_NAME="${RUN_NAME_PREFIX:-map4d_dit}_${task}_${num_demo}demos_seed${SEED:-0}"
    THIS_LOG_FILE="${LOG_FILE:-outputs/map4d_backbone_train_logs/${RUN_NAME}.log}"

    echo "[$(date +%H:%M:%S)] running ${RUN_NAME}"
    echo "  task: ${TASK_NAME}"
    echo "  num_demo: ${num_demo}"
    echo "  demo: ${DEMO_PATH}"
    echo "  sidecar: ${SIDECAR_PATH}"
    echo "  log: ${THIS_LOG_FILE}"

    cmd=(
      torchrun --nproc_per_node="${NPROC_PER_NODE}"
      map4d/backbone/train_map4d_dit.py
      --config-name map4d_dit
      "${TASK_OVERRIDE}"
      "seed=${SEED:-0}"
      "addition_info=${task}_${num_demo}demos"
      "task.dataset.num_traj=${num_demo}"
      "training.debug=${DEBUG:-false}"
      "logging.mode=${WANDB_MODE}"
    )

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'MAP4D_DEMO_PATH=%q MAP4D_KEYFRAME_SIDECAR_PATH=%q CUDA_VISIBLE_DEVICES=%q ' \
        "$DEMO_PATH" "$SIDECAR_PATH" "$CUDA_VISIBLE_DEVICES"
      printf '%q ' "${cmd[@]}"
      printf '\n'
      continue
    fi

    mkdir -p "$(dirname "$THIS_LOG_FILE")"
    MAP4D_DEMO_PATH="$DEMO_PATH" \
    MAP4D_KEYFRAME_SIDECAR_PATH="$SIDECAR_PATH" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "${cmd[@]}" 2>&1 | tee "$THIS_LOG_FILE"
  done
done
