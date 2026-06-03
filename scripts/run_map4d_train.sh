#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy"
DP_DIR="${ROOT_DIR}/baselines/diffusion_policy"
MATRIX_SCRIPT="${ROOT_DIR}/smaller_than_4_matrix.py"
TRAIN_CONFIG="${TRAIN_CONFIG:-${ROOT_DIR}/baselines/diffusion_policy/configs/stackcube_map4d_train.conf}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

cuda_flag="${CUDA_FLAG:-}"
matrix_timeout_seconds="${MATRIX_TIMEOUT_SECONDS:-}"

source "${TRAIN_CONFIG}"

if [[ "${LOG_FILE}" != /* ]]; then
  LOG_FILE="${ROOT_DIR}/${LOG_FILE}"
fi

mkdir -p "$(dirname "${LOG_FILE}")"

cd "${DP_DIR}"

set +e
python train_rgbd.py "${TRAIN_ARGS[@]}" ${cuda_flag} "$@" 2>&1 | tee "${LOG_FILE}"
train_status=$?

echo "train_rgbd.py exited with status ${train_status}; starting ${MATRIX_SCRIPT}"

cd "${ROOT_DIR}"
if [[ -n "${matrix_timeout_seconds}" ]]; then
  timeout "${matrix_timeout_seconds}" python "${MATRIX_SCRIPT}"
else
  python "${MATRIX_SCRIPT}"
fi
matrix_status=$?

if [[ ${matrix_status} -ne 0 ]]; then
  exit "${matrix_status}"
fi
exit "${train_status}"
