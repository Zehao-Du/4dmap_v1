#!/bin/bash
# DP + raw concat + keyframe future loss + TCP pose on PlugCharger-v1, 1000 demos
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data2/zehao/MAP4D/4dmap_v1}"
DATA_ROOT="${DATA_ROOT:-/data2/zehao/MAP4D/dataset}"
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"
DEMO_PATH="${DEMO_PATH:-${DATA_ROOT}/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
KEYFRAME_AUX_PATH="${KEYFRAME_AUX_PATH:-${DEMO_PATH%.h5}.keyframe_aux_h${FUTURE_HORIZON}.h5}"
NUM_DEMOS="${NUM_DEMOS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TOTAL_ITERS="${TOTAL_ITERS:-100000}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

if [[ ! -f "${DEMO_PATH}" ]]; then
  echo "Demo file not found: ${DEMO_PATH}" >&2
  exit 1
fi
if [[ ! -f "${KEYFRAME_AUX_PATH}" ]]; then
  echo "Keyframe aux file not found: ${KEYFRAME_AUX_PATH}" >&2
  echo "Build it with: TASK_NAME=PlugCharger-v1 bash scripts/data_collection/build_keyframe_aux_dataset.sh" >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$DP_DIR"
mkdir -p outputs/train_logs

RUN_NAME="${RUN_NAME:-plugcharger_dp_map4d_keyframe_aux_1000demos_seed1}"
LOG_FILE="${LOG_FILE:-outputs/train_logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "$LOG_FILE")"

TRAIN_ARGS=(
  --exp-name "${RUN_NAME}"
  --seed 1
  --env-id PlugCharger-v1
  --demo-path "${DEMO_PATH}"
  --num-demos "${NUM_DEMOS}"
  --control-mode pd_ee_delta_pose
  --sim-backend physx_cpu
  --obs-mode rgb
  --max-episode-steps 400
  --total-iters "${TOTAL_ITERS}"
  --batch-size "${BATCH_SIZE}"
  --obs-horizon 2
  --act-horizon 8
  --pred-horizon 16
  --lr 0.0001
  --visual-encoder plain_conv
  --use-map4d
  --map4d-raw-concat
  --map4d-keyframe-aux-loss
  --map4d-keyframe-aux-path "${KEYFRAME_AUX_PATH}"
  --map4d-aux-weight 1.0
  --map4d-source maniskill_gt
  --map4d-task-name PlugCharger-v1
  --map4d-pre-horizon 6
  --map4d-future-horizon "${FUTURE_HORIZON}"
  --map4d-num-objects 2
  --log-freq 100
  --eval-freq 10000
  --num-eval-episodes 100
  --num-eval-envs 10
  --num-dataload-workers 0
  --no-track
  --no-capture-video
)

if [[ "${NO_CUDA:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-cuda)
fi
if [[ "${NO_EVAL:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-eval)
fi

echo "[$(date +%H:%M:%S)] running ${RUN_NAME} -> ${LOG_FILE}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'cd %s\npython train_rgbd.py' "$DP_DIR"
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi
python train_rgbd.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
