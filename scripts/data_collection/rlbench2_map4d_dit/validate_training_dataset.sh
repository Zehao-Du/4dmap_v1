#!/usr/bin/env bash
# Validate every training key and stream min/max/mean/std over all visual files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
DATASET_DIR="${POLICY_DIR}/dataset/rlbench2/map4d_dit/bimanual_push_box"
MANIFEST="${DATASET_DIR}/rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.env"
COPPELIASIM_ROOT="${MAP4D_COPPELIASIM_ROOT:-${PROJECT_ROOT}/codes/CoppeliaSim}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Dataset manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]]; then
  echo "CoppeliaSim library not found: ${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" >&2
  exit 1
fi

set -a
source "${MANIFEST}"
set +a

START="${VALIDATE_START:-${RLBENCH2_START}}"
END="${VALIDATE_END:-${RLBENCH2_END}}"
REPORT="${VALIDATE_REPORT:-${DATASET_DIR}/validation_stats_ep${START}-${END}.json}"
export PYTHONPATH="${POLICY_DIR}:${PROJECT_ROOT}/codes/rlbench:${PROJECT_ROOT}/codes/pyrep${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec conda run --no-capture-output -n 4dmap python "${SCRIPT_DIR}/validate_training_dataset.py" \
  --data-path "${RLBENCH2_DATA_PATH}" \
  --pcd-path "${RLBENCH2_PCD_PATH}" \
  --dino-path "${RLBENCH2_DINO_PATH}" \
  --lang-emb-path "${RLBENCH2_LANG_EMB_PATH}" \
  --pose-path "${RLBENCH2_POSE_PATH}" \
  --pcd-type "${RLBENCH2_PCD_TYPE}" \
  --prediction-type "${RLBENCH2_PREDICTION_TYPE}" \
  --start "${START}" \
  --end "${END}" \
  --report "${REPORT}"
