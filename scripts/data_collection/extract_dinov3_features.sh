#!/usr/bin/env bash
# Extract DINOv3 features from ManiSkill demo RGB images.
# Produces a sidecar HDF5 file with per-frame CLS-token features (384-dim for ViT-S/16).
#
# Usage:
#   bash scripts/data_collection/extract_dinov3_features.sh [TASK_NAME] [MODEL_SIZE]
#
# Examples:
#   bash scripts/data_collection/extract_dinov3_features.sh StackCube-v1
#   bash scripts/data_collection/extract_dinov3_features.sh PlugCharger-v1
#   MODEL_SIZE=base bash scripts/data_collection/extract_dinov3_features.sh StackCube-v1
#
# Environment variables:
#   TASK_NAME       - StackCube-v1 or PlugCharger-v1 (default: StackCube-v1)
#   MODEL_SIZE      - small, small_plus, base, large (default: small)
#   IMAGE_SIZE      - resize to this before extraction (default: 224)
#   BATCH_SIZE      - inference batch size (default: 64)
#   NUM_TRAJ        - limit number of trajectories (default: all)
#   CUDA_VISIBLE_DEVICES - GPU to use (default: 0)
set -euo pipefail

TASK_NAME="${1:-${TASK_NAME:-StackCube-v1}}"
MODEL_SIZE="${2:-${MODEL_SIZE:-small}}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAJ="${NUM_TRAJ:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OVERWRITE="${OVERWRITE:-0}"

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap}"
POLICY_DIR="${ROOT_DIR}/4dmap_policy"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

case "${MODEL_SIZE}" in
  small|s)   TIMM_MODEL="vit_small_patch16_dinov3"; SUFFIX="dinov3_s16" ;;
  small_plus|sp) TIMM_MODEL="vit_small_plus_patch16_dinov3"; SUFFIX="dinov3_sp16" ;;
  base|b)    TIMM_MODEL="vit_base_patch16_dinov3"; SUFFIX="dinov3_b16" ;;
  large|l)   TIMM_MODEL="vit_large_patch16_dinov3"; SUFFIX="dinov3_l16" ;;
  *)
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE}. Use small, small_plus, base, or large." >&2
    exit 1
    ;;
esac

case "${TASK_NAME}" in
  StackCube-v1)
    DEMO_PATH="${DEMO_PATH:-${ROOT_DIR}/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.h5}"
    ;;
  PlugCharger-v1)
    DEMO_PATH="${DEMO_PATH:-${ROOT_DIR}/dataset/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5}"
    ;;
  *)
    if [[ -z "${DEMO_PATH:-}" ]]; then
      echo "Unknown task ${TASK_NAME}. Set DEMO_PATH explicitly." >&2
      exit 1
    fi
    ;;
esac

OUTPUT_PATH="${OUTPUT_PATH:-${DEMO_PATH%.h5}.${SUFFIX}.h5}"

if [[ ! -f "${DEMO_PATH}" ]]; then
  echo "Demo file not found: ${DEMO_PATH}" >&2
  exit 1
fi

echo "[$(date +%H:%M:%S)] Extracting DINOv3 features"
echo "  task: ${TASK_NAME}"
echo "  model: ${TIMM_MODEL}"
echo "  image_size: ${IMAGE_SIZE}"
echo "  demo: ${DEMO_PATH}"
echo "  output: ${OUTPUT_PATH}"

cd "${POLICY_DIR}"

cmd=(
  python scripts/data_collection/extract_dinov3_features.py
  --demo-path "${DEMO_PATH}"
  --output-path "${OUTPUT_PATH}"
  --model-name "${TIMM_MODEL}"
  --image-size "${IMAGE_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --device "cuda:0"
)

if [[ -n "${NUM_TRAJ}" ]]; then
  cmd+=(--num-traj "${NUM_TRAJ}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
