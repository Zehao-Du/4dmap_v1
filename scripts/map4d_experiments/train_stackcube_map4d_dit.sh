#!/usr/bin/env bash
# Train Map4DDiT on an already-built StackCube dataset.
#
# Usage:
#   bash scripts/map4d_experiments/train_stackcube_map4d_dit.sh <num_demos>
#
# Example:
#   bash scripts/map4d_experiments/train_stackcube_map4d_dit.sh 1000
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <num_demos>" >&2
  exit 1
fi

NUM_DEMOS="$1"
if [[ ! "${NUM_DEMOS}" =~ ^[0-9]+$ || "${NUM_DEMOS}" -le 0 ]]; then
  echo "Invalid num_demos=${NUM_DEMOS}. Use a positive integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP_STACKCUBE_DIT_TRAIN:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP_STACKCUBE_DIT_TRAIN=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

TASK_KEY="stackcube"
RESOLUTION_TAG="${RESOLUTION_TAG:-224}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-map4d_dit}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
LOGGING_MODE="${LOGGING_MODE:-${WANDB_MODE:-disabled}}"
DEBUG="${DEBUG:-false}"
ONLINE_DINO="${ONLINE_DINO:-0}"
TRAIN_SEMANTIC_FEATURE_MODE="${TRAIN_SEMANTIC_FEATURE_MODE:-precomputed}"
if [[ "${ONLINE_DINO}" == "1" || "${ONLINE_DINO}" == "true" ]]; then
  TRAIN_SEMANTIC_FEATURE_MODE="online_dinov3"
fi
ADDITION_INFO="${ADDITION_INFO:-${TASK_KEY}_${NUM_DEMOS}demos_${RESOLUTION_TAG}}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(
    python - 29570 <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, start + 100):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit(f"no free port found in [{start}, {start + 100})")
PY
  )"
fi
if [[ -z "${BATCH_SIZE:-}" ]]; then
  BATCH_SIZE="32"
fi
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DINOV3_MODEL="${DINOV3_MODEL:-dinov3_vits16}"
DINOV3_WEIGHTS_PATH="${DINOV3_WEIGHTS_PATH:-/data2/zehao/models/DINOv3/DINOv3_ViT_LVD_1689M/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
DINOV3_THIRD_PARTY_DIR="${DINOV3_THIRD_PARTY_DIR:-${ROOT_DIR}/map4d/backbone/model/vision/dinov3}"
DINOV3_IMAGE_SIZE="${DINOV3_IMAGE_SIZE:-${RESOLUTION_TAG}}"
DINOV3_INPUT_MULTIPLE="${DINOV3_INPUT_MULTIPLE:-16}"
DINOV3_AMP="${DINOV3_AMP:-true}"

MANIFEST_DIR="${MANIFEST_DIR:-${ROOT_DIR}/outputs/map4d_dit_manifests}"
if [[ -z "${MANIFEST_PATH:-}" ]]; then
  MANIFEST_PATH="${MANIFEST_DIR}/${TASK_KEY}_${NUM_DEMOS}demos_${RESOLUTION_TAG}_h${FUTURE_HORIZON}.env"
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    BEST_MANIFEST=""
    BEST_NUM_DEMOS=""
    for candidate in "${MANIFEST_DIR}/${TASK_KEY}_"*demos_"${RESOLUTION_TAG}"_h"${FUTURE_HORIZON}".env; do
      [[ -f "${candidate}" ]] || continue
      filename="$(basename "${candidate}")"
      candidate_num="${filename#${TASK_KEY}_}"
      candidate_num="${candidate_num%%demos_*}"
      [[ "${candidate_num}" =~ ^[0-9]+$ ]] || continue
      if (( candidate_num >= NUM_DEMOS )); then
        if [[ -z "${BEST_NUM_DEMOS}" ]] || (( candidate_num < BEST_NUM_DEMOS )); then
          BEST_NUM_DEMOS="${candidate_num}"
          BEST_MANIFEST="${candidate}"
        fi
      fi
    done
    if [[ -n "${BEST_MANIFEST}" ]]; then
      MANIFEST_PATH="${BEST_MANIFEST}"
    fi
  fi
fi
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/map4d_dit_train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ADDITION_INFO}_seed${SEED}.log}"

cd "${ROOT_DIR}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  echo "Build the dataset first, or set MANIFEST_PATH explicitly." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${MANIFEST_PATH}"

required_vars=(
  TASK_OVERRIDE
  MAP4D_DEMO_PATH
  MAP4D_KEYFRAME_SIDECAR_PATH
  MAP4D_NUM_TRAJ
  SEMANTIC_FEATURE_DIM
  MAP_FEATURE_DIM
  NUM_MAP_NODES
)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Manifest missing required variable: ${var_name}" >&2
    exit 1
  fi
done

MANIFEST_NUM_TRAJ="${MAP4D_NUM_TRAJ}"
if (( MANIFEST_NUM_TRAJ < NUM_DEMOS )); then
  echo "Manifest MAP4D_NUM_TRAJ=${MAP4D_NUM_TRAJ} is smaller than requested num_demos=${NUM_DEMOS}." >&2
  exit 1
fi
if [[ ! -f "${MAP4D_DEMO_PATH}" ]]; then
  echo "Demo file not found: ${MAP4D_DEMO_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MAP4D_KEYFRAME_SIDECAR_PATH}" ]]; then
  echo "Sidecar file not found: ${MAP4D_KEYFRAME_SIDECAR_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

echo "[$(date +%H:%M:%S)] Training StackCube Map4DDiT"
echo "  num_demos: ${NUM_DEMOS}"
echo "  manifest_num_demos: ${MANIFEST_NUM_TRAJ}"
echo "  manifest: ${MANIFEST_PATH}"
echo "  demo: ${MAP4D_DEMO_PATH}"
echo "  sidecar: ${MAP4D_KEYFRAME_SIDECAR_PATH}"
echo "  semantic_feature_mode: ${TRAIN_SEMANTIC_FEATURE_MODE}"
echo "  semantic_feature_dim: ${SEMANTIC_FEATURE_DIM}"
echo "  map_feature_dim: ${MAP_FEATURE_DIM}"
echo "  num_map_nodes: ${NUM_MAP_NODES}"
if [[ "${TRAIN_SEMANTIC_FEATURE_MODE}" == "online_dinov3" ]]; then
  echo "  dinov3_model: ${DINOV3_MODEL}"
  echo "  dinov3_weights_path: ${DINOV3_WEIGHTS_PATH}"
  echo "  dinov3_third_party_dir: ${DINOV3_THIRD_PARTY_DIR}"
  echo "  dinov3_image_size: ${DINOV3_IMAGE_SIZE}"
fi
echo "  cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
echo "  master_port: ${MASTER_PORT}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  log: ${LOG_FILE}"

export CONFIG_NAME NPROC_PER_NODE
export MAP4D_DEMO_PATH MAP4D_KEYFRAME_SIDECAR_PATH
export MAP4D_NUM_TRAJ="${NUM_DEMOS}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF

overrides=(
  "${TASK_OVERRIDE}"
  "seed=${SEED}"
  "addition_info=${ADDITION_INFO}"
  "task.dataset.num_traj=${NUM_DEMOS}"
  "policy.model_cfg.semantic_feature_dim=${SEMANTIC_FEATURE_DIM}"
  "policy.model_cfg.semantic_feature_mode=${TRAIN_SEMANTIC_FEATURE_MODE}"
  "policy.model_cfg.map_feature_dim=${MAP_FEATURE_DIM}"
  "policy.model_cfg.num_map_nodes=${NUM_MAP_NODES}"
  "dataloader.batch_size=${BATCH_SIZE}"
  "val_dataloader.batch_size=${BATCH_SIZE}"
  "training.debug=${DEBUG}"
  "logging.mode=${LOGGING_MODE}"
)

if [[ "${TRAIN_SEMANTIC_FEATURE_MODE}" == "online_dinov3" ]]; then
  overrides+=(
    "policy.model_cfg.dinov3_model=${DINOV3_MODEL}"
    "policy.model_cfg.dinov3_weights_path=${DINOV3_WEIGHTS_PATH}"
    "policy.model_cfg.dinov3_third_party_dir=${DINOV3_THIRD_PARTY_DIR}"
    "policy.model_cfg.dinov3_image_size=${DINOV3_IMAGE_SIZE}"
    "policy.model_cfg.dinov3_input_multiple=${DINOV3_INPUT_MULTIPLE}"
    "policy.model_cfg.dinov3_amp=${DINOV3_AMP}"
  )
fi

if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  overrides+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
fi
if [[ -n "${MAX_VAL_STEPS:-}" ]]; then
  overrides+=("training.max_val_steps=${MAX_VAL_STEPS}")
fi
if [[ -n "${NUM_EPOCHS:-}" ]]; then
  overrides+=("training.num_epochs=${NUM_EPOCHS}")
fi
if [[ -n "${USE_EMA:-}" ]]; then
  overrides+=("training.use_ema=${USE_EMA}")
fi
if [[ -n "${ROLLOUT_ENABLED:-}" ]]; then
  overrides+=("rollout.enabled=${ROLLOUT_ENABLED}")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  overrides+=("dataloader.num_workers=${NUM_WORKERS}" "val_dataloader.num_workers=${NUM_WORKERS}")
fi

torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" \
  map4d/backbone/train_map4d_dit.py \
  --config-name "${CONFIG_NAME}" \
  "${overrides[@]}" \
  2>&1 | tee "${LOG_FILE}"
