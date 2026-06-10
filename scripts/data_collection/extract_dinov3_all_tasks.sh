#!/usr/bin/env bash
# Extract DINOv3 features for ALL tasks using the strongest available model.
# Then merge features into the keyframe sidecar files for Map4D DiT training.
#
# Usage:
#   bash scripts/data_collection/extract_dinov3_all_tasks.sh
#   MODEL_SIZE=base bash scripts/data_collection/extract_dinov3_all_tasks.sh
#
# Available models (strongest to fastest):
#   large   - ViT-L/16, 1024-dim, 1.2GB (default, best quality)
#   base    - ViT-B/16, 768-dim, 327MB
#   small   - ViT-S/16, 384-dim, 83MB (fastest)
set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-large}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OVERWRITE="${OVERWRITE:-1}"

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap}"
POLICY_DIR="${ROOT_DIR}/4dmap_policy"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

case "${MODEL_SIZE}" in
  small|s)       TIMM_MODEL="vit_small_patch16_dinov3"; SUFFIX="dinov3_s16"; FEAT_DIM=384 ;;
  small_plus|sp) TIMM_MODEL="vit_small_plus_patch16_dinov3"; SUFFIX="dinov3_sp16"; FEAT_DIM=384 ;;
  base|b)        TIMM_MODEL="vit_base_patch16_dinov3"; SUFFIX="dinov3_b16"; FEAT_DIM=768 ;;
  large|l)       TIMM_MODEL="vit_large_patch16_dinov3"; SUFFIX="dinov3_l16"; FEAT_DIM=1024 ;;
  *)
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE}. Use small, base, or large." >&2
    exit 1
    ;;
esac

echo "============================================"
echo " DINOv3 Feature Extraction - All Tasks"
echo "============================================"
echo "  Model: ${TIMM_MODEL} (${FEAT_DIM}-dim)"
echo "  Image size: ${IMAGE_SIZE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

cd "${POLICY_DIR}"

# Define all tasks and their demo paths
declare -A DEMO_PATHS
DEMO_PATHS=(
  ["StackCube-v1"]="${ROOT_DIR}/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.h5"
  ["PlugCharger-v1"]="${ROOT_DIR}/dataset/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5"
)

# Step 1: Extract features for each task
for TASK_NAME in "${!DEMO_PATHS[@]}"; do
  DEMO_PATH="${DEMO_PATHS[$TASK_NAME]}"
  OUTPUT_PATH="${DEMO_PATH%.h5}.${SUFFIX}.h5"

  if [[ ! -f "${DEMO_PATH}" ]]; then
    echo "[SKIP] ${TASK_NAME}: demo file not found: ${DEMO_PATH}"
    continue
  fi

  if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]; then
    echo "[SKIP] ${TASK_NAME}: output exists: ${OUTPUT_PATH}"
    continue
  fi

  echo "[$(date +%H:%M:%S)] Extracting ${TASK_NAME}..."
  echo "  Input: ${DEMO_PATH}"
  echo "  Output: ${OUTPUT_PATH}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python scripts/data_collection/extract_dinov3_features.py \
    --demo-path "${DEMO_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --model-name "${TIMM_MODEL}" \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "cuda:0" \
    --overwrite

  echo ""
done

# Step 2: Merge features into keyframe sidecar files
echo "[$(date +%H:%M:%S)] Merging features into sidecar files..."

python -c "
import h5py
import numpy as np
import os

root = '${ROOT_DIR}'
suffix = '${SUFFIX}'
tasks = {
    'StackCube-v1': {
        'dinov3': root + '/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.' + suffix + '.h5',
        'sidecar': root + '/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.map4d_dit_h4.h5',
    },
    'PlugCharger-v1': {
        'dinov3': root + '/dataset/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.' + suffix + '.h5',
        'sidecar': root + '/dataset/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.map4d_dit_h4.h5',
    },
}

for task_name, paths in tasks.items():
    dinov3_path = paths['dinov3']
    sidecar_path = paths['sidecar']

    if not os.path.exists(dinov3_path):
        print(f'  [SKIP] {task_name}: no DINOv3 file at {dinov3_path}')
        continue
    if not os.path.exists(sidecar_path):
        print(f'  [SKIP] {task_name}: no sidecar file at {sidecar_path}')
        continue

    with h5py.File(dinov3_path, 'r') as f_dino, h5py.File(sidecar_path, 'a') as f_side:
        traj_keys = [k for k in f_dino.keys() if k.startswith('traj_')]
        merged = 0
        for k in traj_keys:
            if k in f_side:
                if 'rgb_feature' in f_side[k]:
                    del f_side[k]['rgb_feature']
                feat = f_dino[k]['rgb_feature'][()]
                f_side[k].create_dataset('rgb_feature', data=feat, dtype=np.float32)
                merged += 1
        print(f'  {task_name}: merged {merged} trajectories ({feat.shape[-1]}-dim)')
"

echo ""
echo "============================================"
echo " Done! Feature dimension: ${FEAT_DIM}"
echo "============================================"
echo ""
echo "To train with these features, set:"
echo "  policy.model_cfg.use_rgb=true"
echo "  policy.model_cfg.rgb_feature_dim=${FEAT_DIM}"
