#!/bin/bash
# DP baseline, 100 demos
set -o pipefail
ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"
if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$DP_DIR"
mkdir -p outputs/train_logs
source "$DP_DIR/configs/stackcube_pos_dp_baseline.conf"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$(date +%H:%M:%S)] running $RUN_NAME -> $LOG_FILE"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'cd %s\npython train_rgbd.py' "$DP_DIR"
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi
python train_rgbd.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
