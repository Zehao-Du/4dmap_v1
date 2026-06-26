#!/usr/bin/env bash
# Train Map4DDiT on an already-built StackCube dataset.
#
# Usage:
#   bash scripts/map4d_experiments/train_stackcube_map4d_dit.sh <num_demos> [hydra_overrides...]
#
# Example:
#   bash scripts/map4d_experiments/train_stackcube_map4d_dit.sh 1000
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <num_demos> [hydra_overrides...]" >&2
  exit 1
fi

NUM_DEMOS="$1"
shift
EXTRA_OVERRIDES=("$@")
if [[ ! "${NUM_DEMOS}" =~ ^[0-9]+$ || "${NUM_DEMOS}" -le 0 ]]; then
  echo "Invalid num_demos=${NUM_DEMOS}. Use a positive integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Machine-specific path: update this if the 4dmap_policy checkout is moved.
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP_STACKCUBE_DIT_TRAIN:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP_STACKCUBE_DIT_TRAIN=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "${NUM_DEMOS}" "${EXTRA_OVERRIDES[@]}"
fi

TASK_KEY="${TASK_KEY:-stackcube}"
RESOLUTION_TAG="${RESOLUTION_TAG:-224}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-map4d_dit}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Machine-specific GPU selection: update this for the target machine.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOGGING_MODE="${LOGGING_MODE:-${WANDB_MODE:-disabled}}"
DEBUG="${DEBUG:-false}"
TRAIN_SEMANTIC_FEATURE_MODE="${TRAIN_SEMANTIC_FEATURE_MODE:-precomputed}"
ADDITION_INFO="${ADDITION_INFO:-${TASK_KEY}_${NUM_DEMOS}demos_${RESOLUTION_TAG}_4d_action}"
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
DINOV3_WEIGHTS_PATH="${DINOV3_WEIGHTS_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/foundation_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
DINOV3_THIRD_PARTY_DIR="${DINOV3_THIRD_PARTY_DIR:-$(cd "${ROOT_DIR}/.." && pwd)/mappolicy/models/DINOv3/dinov3}"
DINOV3_IMAGE_SIZE="${DINOV3_IMAGE_SIZE:-${RESOLUTION_TAG}}"

# Machine-specific path: where reusable dataset manifests are stored.
MANIFEST_DIR="${MANIFEST_DIR:-${ROOT_DIR}/outputs/map4d_dit_manifests}"

# Machine-specific path: default location of generated ManiSkill datasets.
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset/maniskill}"
if [[ -z "${DATASET_DIR:-}" ]]; then
  case "${TASK_KEY}" in
    stackcube)
      DATASET_DIR="${DATA_ROOT}/ManiSkill/StackCube-v1/motionplanning"
      ;;
    plugcharger)
      DATASET_DIR="${DATA_ROOT}/ManiSkill/PlugCharger-v1/motionplanning"
      ;;
    *)
      echo "Unknown TASK_KEY=${TASK_KEY}. Set DATASET_DIR and MANIFEST_PATH explicitly." >&2
      exit 1
      ;;
  esac
fi
if [[ -z "${DATASET_MANIFEST:-}" ]]; then
  case "${TASK_KEY}" in
    stackcube)
      DATASET_MANIFEST="${DATASET_DIR}/StackCube.rgb+depth.pd_ee_delta_pos.physx_cpu.filtered.map4d_dit_h${FUTURE_HORIZON}.context.env"
      ;;
    plugcharger)
      DATASET_MANIFEST="${DATASET_DIR}/PlugCharger.rgb+depth.pd_ee_delta_pose.physx_cpu.filtered.map4d_dit_h${FUTURE_HORIZON}.context.env"
      ;;
    *)
      DATASET_MANIFEST=""
      ;;
  esac
fi

manifest_has_files() {
  local manifest="$1"
  [[ -f "${manifest}" ]] || return 1
  (
    set +u
    # shellcheck disable=SC1090
    source "${manifest}"
    [[ -f "${MAP4D_DEMO_PATH:-}" && -f "${MAP4D_KEYFRAME_SIDECAR_PATH:-}" ]]
  )
}

if [[ -z "${MANIFEST_PATH:-}" ]]; then
  MANIFEST_PATH="${DATASET_MANIFEST}"
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

if [[ "${TRAIN_SEMANTIC_FEATURE_MODE}" != "precomputed" ]]; then
  echo "Map4DDiT training now requires precomputed obs/dino_feature, got TRAIN_SEMANTIC_FEATURE_MODE=${TRAIN_SEMANTIC_FEATURE_MODE}." >&2
  exit 1
fi
POINTCLOUD_ENCODER_IN_CHANNELS="$((SEMANTIC_FEATURE_DIM + 3))"

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

echo "[$(date +%H:%M:%S)] Training ${TASK_KEY} Map4DDiT"
echo "  num_demos: ${NUM_DEMOS}"
echo "  manifest_num_demos: ${MANIFEST_NUM_TRAJ}"
echo "  manifest: ${MANIFEST_PATH}"
echo "  demo: ${MAP4D_DEMO_PATH}"
echo "  sidecar: ${MAP4D_KEYFRAME_SIDECAR_PATH}"
echo "  semantic_feature_mode: ${TRAIN_SEMANTIC_FEATURE_MODE}"
echo "  semantic_feature_dim: ${SEMANTIC_FEATURE_DIM}"
echo "  map_feature_dim: ${MAP_FEATURE_DIM}"
echo "  num_map_nodes: ${NUM_MAP_NODES}"
echo "  pointcloud_encoder_in_channels: ${POINTCLOUD_ENCODER_IN_CHANNELS}"
echo "  cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
echo "  data_root: ${DATA_ROOT}"
echo "  dataset_dir: ${DATASET_DIR}"
echo "  master_port: ${MASTER_PORT}"
echo "  batch_size: ${BATCH_SIZE}"
if [[ -n "${DDP_FIND_UNUSED_PARAMETERS:-}" ]]; then
  echo "  ddp_find_unused_parameters: ${DDP_FIND_UNUSED_PARAMETERS}"
fi
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
  "policy.model_cfg.rgb_feature_dim=${SEMANTIC_FEATURE_DIM}"
  "policy.model_cfg.semantic_feature_dim=${SEMANTIC_FEATURE_DIM}"
  "policy.model_cfg.semantic_feature_mode=${TRAIN_SEMANTIC_FEATURE_MODE}"
  "policy.model_cfg.pointcloud_encoder_cfg.in_channels=${POINTCLOUD_ENCODER_IN_CHANNELS}"
  "policy.model_cfg.map_feature_dim=${MAP_FEATURE_DIM}"
  "policy.model_cfg.num_map_nodes=${NUM_MAP_NODES}"
  "dataloader.batch_size=${BATCH_SIZE}"
  "val_dataloader.batch_size=${BATCH_SIZE}"
  "training.debug=${DEBUG}"
  "logging.mode=${LOGGING_MODE}"
  "rollout.dinov3_model=${DINOV3_MODEL}"
  "rollout.dinov3_weights_path=${DINOV3_WEIGHTS_PATH}"
  "rollout.dinov3_third_party_dir=${DINOV3_THIRD_PARTY_DIR}"
  "rollout.image_size=${DINOV3_IMAGE_SIZE}"
)

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
if [[ -n "${CHECKPOINT_SAVE_CKPT:-}" ]]; then
  overrides+=("checkpoint.save_ckpt=${CHECKPOINT_SAVE_CKPT}")
fi
if [[ -n "${ROLLOUT_ENABLED:-}" ]]; then
  overrides+=("rollout.enabled=${ROLLOUT_ENABLED}")
fi
if [[ -n "${ROLLOUT_EVERY:-}" ]]; then
  overrides+=("rollout.every=${ROLLOUT_EVERY}")
fi
if [[ -n "${ROLLOUT_NUM_EVAL_EPISODES:-}" ]]; then
  overrides+=("rollout.num_eval_episodes=${ROLLOUT_NUM_EVAL_EPISODES}")
fi
if [[ -n "${ROLLOUT_NUM_EVAL_ENVS:-}" ]]; then
  overrides+=("rollout.num_eval_envs=${ROLLOUT_NUM_EVAL_ENVS}")
fi
if [[ -n "${ROLLOUT_SIM_BACKEND:-}" ]]; then
  overrides+=("rollout.sim_backend=${ROLLOUT_SIM_BACKEND}")
fi
if [[ -n "${ROLLOUT_IMAGE_SIZE:-}" ]]; then
  overrides+=("rollout.image_size=${ROLLOUT_IMAGE_SIZE}")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  overrides+=("dataloader.num_workers=${NUM_WORKERS}" "val_dataloader.num_workers=${NUM_WORKERS}")
fi
if [[ -n "${DDP_FIND_UNUSED_PARAMETERS:-}" ]]; then
  overrides+=("training.ddp_find_unused_parameters=${DDP_FIND_UNUSED_PARAMETERS}")
fi
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  overrides+=("${EXTRA_OVERRIDES[@]}")
fi

VISIBLE_GPU_COUNT="$(
  python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if (( NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} but only ${VISIBLE_GPU_COUNT} CUDA device(s) are visible." >&2
  echo "Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}." >&2
  echo "Expose at least ${NPROC_PER_NODE} GPUs, or set NPROC_PER_NODE=${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" \
  map4d/backbone/train_map4d_dit.py \
  --config-name "${CONFIG_NAME}" \
  "${overrides[@]}" \
  2>&1 | tee "${LOG_FILE}"
