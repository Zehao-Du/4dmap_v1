#!/usr/bin/env bash
# Build a Map4DDiT dataset from scratch, then train with the generated manifest.
#
# Usage:
#   bash scripts/build_and_train_map4d_dit.sh <task> <demos> [resolution] [-- hydra_overrides...]
#
# Examples:
#   bash scripts/build_and_train_map4d_dit.sh stackcube 1 224 -- training.max_train_steps=100
#   SKIP_COLLECT=1 DEMO_PATH=/path/to/demo_with_dino.h5 \
#     bash scripts/build_and_train_map4d_dit.sh plugcharger 100 native -- seed=1
#
# The dataset builder generates real per-point DINO features by default.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <task> <demos> [resolution] [-- hydra_overrides...]" >&2
  echo "  task: stackcube|plugcharger|StackCube-v1|PlugCharger-v1" >&2
  echo "  demos: positive integer" >&2
  echo "  resolution: native|SIZE|WIDTHxHEIGHT, default 224" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
POLICY_DIR="${POLICY_DIR:-${ROOT_DIR}}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP_BUILD_TRAIN:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP_BUILD_TRAIN=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

TASK_ARG="$1"
DEMOS="$2"
shift 2

RESOLUTION="224"
if [[ $# -gt 0 && "$1" != "--" ]]; then
  RESOLUTION="$1"
  shift
fi

TRAIN_OVERRIDES=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" != "--" ]]; then
    echo "Unexpected argument: $1" >&2
    echo "Put training Hydra overrides after --." >&2
    exit 1
  fi
  shift
  TRAIN_OVERRIDES=("$@")
fi

if [[ ! "${DEMOS}" =~ ^[0-9]+$ || "${DEMOS}" -le 0 ]]; then
  echo "Invalid demos=${DEMOS}. Use a positive integer." >&2
  exit 1
fi

case "${TASK_ARG,,}" in
  stackcube|stackcube-v1)
    TASK_KEY="stackcube"
    TASK_OVERRIDE="task=stackcube_map4d_dit"
    ;;
  plugcharger|plugcharger-v1)
    TASK_KEY="plugcharger"
    TASK_OVERRIDE="task=plugcharger_map4d_dit"
    ;;
  *)
    echo "Unsupported task=${TASK_ARG}. Use stackcube or plugcharger." >&2
    exit 1
    ;;
esac

RESOLUTION_TAG="${RESOLUTION,,}"
case "${RESOLUTION_TAG}" in
  native|original|none)
    RESOLUTION_TAG="native"
    ;;
  [0-9]*)
    if [[ "${RESOLUTION_TAG}" =~ ^[0-9]+$ ]]; then
      :
    elif [[ "${RESOLUTION_TAG}" =~ ^[0-9]+x[0-9]+$ ]]; then
      RESOLUTION_TAG="${RESOLUTION_TAG//x/-}"
    else
      echo "Invalid resolution=${RESOLUTION}. Use native, SIZE, or WIDTHxHEIGHT." >&2
      exit 1
    fi
    ;;
  *)
    echo "Invalid resolution=${RESOLUTION}. Use native, SIZE, or WIDTHxHEIGHT." >&2
    exit 1
    ;;
esac

FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-map4d_dit}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOGGING_MODE="${LOGGING_MODE:-${WANDB_MODE:-disabled}}"
DEBUG="${DEBUG:-false}"
ADDITION_INFO="${ADDITION_INFO:-${TASK_KEY}_${DEMOS}demos_${RESOLUTION_TAG}}"
MANIFEST_DIR="${MANIFEST_DIR:-${ROOT_DIR}/outputs/map4d_dit_manifests}"
MANIFEST_PATH="${MANIFEST_PATH:-${MANIFEST_DIR}/${TASK_KEY}_${DEMOS}demos_${RESOLUTION_TAG}_h${FUTURE_HORIZON}.env}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/map4d_dit_train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ADDITION_INFO}_seed${SEED}.log}"

mkdir -p "$(dirname "${MANIFEST_PATH}")" "$(dirname "${LOG_FILE}")"

cd "${ROOT_DIR}"

echo "[$(date +%H:%M:%S)] Building Map4DDiT dataset"
echo "  task: ${TASK_ARG}"
echo "  demos: ${DEMOS}"
echo "  resolution: ${RESOLUTION}"
echo "  manifest: ${MANIFEST_PATH}"

MANIFEST_PATH="${MANIFEST_PATH}" \
bash "${ROOT_DIR}/scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh" \
  "${TASK_ARG}" "${DEMOS}" "${RESOLUTION}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Dataset manifest was not created: ${MANIFEST_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${MANIFEST_PATH}"

required_vars=(
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

echo "[$(date +%H:%M:%S)] Training Map4DDiT"
echo "  demo: ${MAP4D_DEMO_PATH}"
echo "  sidecar: ${MAP4D_KEYFRAME_SIDECAR_PATH}"
echo "  num_traj: ${MAP4D_NUM_TRAJ}"
echo "  semantic_feature_dim: ${SEMANTIC_FEATURE_DIM}"
echo "  map_feature_dim: ${MAP_FEATURE_DIM}"
echo "  num_map_nodes: ${NUM_MAP_NODES}"
echo "  log: ${LOG_FILE}"

export CONFIG_NAME NPROC_PER_NODE
export MAP4D_DEMO_PATH MAP4D_KEYFRAME_SIDECAR_PATH MAP4D_NUM_TRAJ

bash "${ROOT_DIR}/scripts/map4d_backbone/run_map4d_dit_train.sh" \
  "${TASK_OVERRIDE}" \
  "seed=${SEED}" \
  "addition_info=${ADDITION_INFO}" \
  "task.dataset.num_traj=${MAP4D_NUM_TRAJ}" \
  "policy.model_cfg.semantic_feature_dim=${SEMANTIC_FEATURE_DIM}" \
  "policy.model_cfg.map_feature_dim=${MAP_FEATURE_DIM}" \
  "policy.model_cfg.num_map_nodes=${NUM_MAP_NODES}" \
  "training.debug=${DEBUG}" \
  "logging.mode=${LOGGING_MODE}" \
  "${TRAIN_OVERRIDES[@]}" \
  2>&1 | tee "${LOG_FILE}"
