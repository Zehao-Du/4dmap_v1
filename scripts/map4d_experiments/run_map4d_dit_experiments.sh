#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  echo "Configure TASK_LIST and NUM_DEMOS_LIST inside the script, or override them with environment variables." >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/data2/zehao/MAP4D/4dmap_v1}"
DATA_ROOT="${DATA_ROOT:-/data2/zehao/MAP4D/dataset}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29570}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

# ======================== EDIT HERE: experiment selection ========================
# TASK_LIST controls which task(s) to train.
#   Options: stackcube, plugcharger, all
# NUM_DEMOS_LIST controls how many trajectories are used.
#   Use one value or comma-separated values, e.g. 1000 or 100,990,1000.
#
# You can edit the defaults below, or override them from the command line:
#   TASK_LIST=plugcharger NUM_DEMOS_LIST=1000 CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29571 bash scripts/map4d_backbone/run_map4d_dit_experiments.sh
TASK_LIST="${TASK_LIST:-stackcube}"
NUM_DEMOS_LIST="${NUM_DEMOS_LIST:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
# ====================== END EDIT: experiment selection ===========================

# =========================== EDIT HERE: dataset paths ============================
# StackCube inputs.
#   STACKCUBE_DEMO_PATH is the main ManiSkill .h5 with DINOv3 features.
#   STACKCUBE_MAP4D_DIT_SIDECAR_PATH is the keyframe/object/tcp sidecar .h5.
#   STACKCUBE_RGB_FEATURE_DIM must match the feature dimension in DEMO_PATH.
STACKCUBE_DEMO_PATH="${STACKCUBE_DEMO_PATH:-${DATA_ROOT}/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.filtered.with_dinov3_vits16_224.h5}"
STACKCUBE_MAP4D_DIT_SIDECAR_PATH="${STACKCUBE_MAP4D_DIT_SIDECAR_PATH:-${DATA_ROOT}/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.filtered.with_dinov3_vits16_224.map4d_dit_h${FUTURE_HORIZON}.h5}"
STACKCUBE_RGB_FEATURE_DIM="${STACKCUBE_RGB_FEATURE_DIM:-768}"

# PlugCharger inputs.
#   Update these if the demo file, sidecar, or vision feature dimension changes.
PLUGCHARGER_DEMO_PATH="${PLUGCHARGER_DEMO_PATH:-${DATA_ROOT}/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5}"
PLUGCHARGER_MAP4D_DIT_SIDECAR_PATH="${PLUGCHARGER_MAP4D_DIT_SIDECAR_PATH:-${PLUGCHARGER_DEMO_PATH%.h5}.map4d_dit_h${FUTURE_HORIZON}.h5}"
PLUGCHARGER_RGB_FEATURE_DIM="${PLUGCHARGER_RGB_FEATURE_DIM:-288}"
# ========================= END EDIT: dataset paths ===============================

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
      DEMO_PATH="${STACKCUBE_DEMO_PATH}"
      SIDECAR_PATH="${STACKCUBE_MAP4D_DIT_SIDECAR_PATH}"
      RGB_FEATURE_DIM="${STACKCUBE_RGB_FEATURE_DIM}"
      ;;
    plugcharger)
      TASK_OVERRIDE="task=plugcharger_map4d_dit"
      TASK_NAME="PlugCharger-v1"
      DEMO_PATH="${PLUGCHARGER_DEMO_PATH}"
      SIDECAR_PATH="${PLUGCHARGER_MAP4D_DIT_SIDECAR_PATH}"
      RGB_FEATURE_DIM="${PLUGCHARGER_RGB_FEATURE_DIM}"
      ;;
    *)
      echo "Internal error: unknown normalized task ${task}" >&2
      return 1
      ;;
  esac
}

split_csv "$TASK_LIST"
RAW_TASKS=("${SPLIT_RESULT[@]}")
split_csv "$NUM_DEMOS_LIST"
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
run_idx=0

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
    THIS_MASTER_PORT="$((MASTER_PORT + run_idx))"
    run_idx="$((run_idx + 1))"

    echo "[$(date +%H:%M:%S)] running ${RUN_NAME}"
    echo "  task: ${TASK_NAME}"
    echo "  num_demo: ${num_demo}"
    echo "  demo: ${DEMO_PATH}"
    echo "  sidecar: ${SIDECAR_PATH}"
    echo "  rgb_feature_dim: ${RGB_FEATURE_DIM}"
    echo "  batch_size: ${BATCH_SIZE}"
    echo "  master_port: ${THIS_MASTER_PORT}"
    echo "  log: ${THIS_LOG_FILE}"

    cmd=(
      torchrun --master_port="${THIS_MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}"
      map4d/backbone/train_map4d_dit.py
      --config-name map4d_dit
      "${TASK_OVERRIDE}"
      "seed=${SEED:-0}"
      "addition_info=${task}_${num_demo}demos"
      "task.dataset.num_traj=${num_demo}"
      "policy.model_cfg.rgb_feature_dim=${RGB_FEATURE_DIM}"
      "dataloader.batch_size=${BATCH_SIZE}"
      "val_dataloader.batch_size=${BATCH_SIZE}"
      "training.debug=${DEBUG:-false}"
      "logging.mode=${WANDB_MODE}"
    )

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'MAP4D_DEMO_PATH=%q MAP4D_KEYFRAME_SIDECAR_PATH=%q MAP4D_NUM_TRAJ=%q CUDA_VISIBLE_DEVICES=%q ' \
        "$DEMO_PATH" "$SIDECAR_PATH" "$num_demo" "$CUDA_VISIBLE_DEVICES"
      printf '%q ' "${cmd[@]}"
      printf '\n'
      continue
    fi

    mkdir -p "$(dirname "$THIS_LOG_FILE")"
    MAP4D_DEMO_PATH="$DEMO_PATH" \
    MAP4D_KEYFRAME_SIDECAR_PATH="$SIDECAR_PATH" \
    MAP4D_NUM_TRAJ="$num_demo" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "${cmd[@]}" 2>&1 | tee "$THIS_LOG_FILE"
  done
done
