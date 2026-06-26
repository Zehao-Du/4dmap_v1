#!/bin/bash
# ACT + raw concat + keyframe future loss + TCP pos, 990 demos
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data2/zehao/MAP4D/4dmap_v1}"
DATA_ROOT="${DATA_ROOT:-/data2/zehao/MAP4D/dataset}"
ACT_DIR="$ROOT_DIR/baselines/act"
DEMO_PATH="${DEMO_PATH:-${DATA_ROOT}/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pos.physx_cpu.filtered.h5}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
TCP_TARGET="${TCP_TARGET:-pos}"
case "${TCP_TARGET}" in
  pose)
    TCP_DIM=7
    DEFAULT_KEYFRAME_AUX_PATH="${DEMO_PATH%.h5}.keyframe_aux_h${FUTURE_HORIZON}.h5"
    ;;
  pos)
    TCP_DIM=3
    DEFAULT_KEYFRAME_AUX_PATH="${DEMO_PATH%.h5}.keyframe_aux_tcp_pos_h${FUTURE_HORIZON}.h5"
    ;;
  *)
    echo "Unsupported TCP_TARGET=${TCP_TARGET}. Use pose or pos." >&2
    exit 1
    ;;
esac
KEYFRAME_AUX_PATH="${KEYFRAME_AUX_PATH:-${DEFAULT_KEYFRAME_AUX_PATH}}"
NUM_DEMOS="${NUM_DEMOS:-990}"
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
  echo "Build it with: TASK_NAME=StackCube-v1 TCP_TARGET=${TCP_TARGET} bash scripts/data_collection/act_dataset/build_keyframe_aux_dataset.sh" >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ACT_DIR"
mkdir -p outputs/train_logs

RUN_NAME="${RUN_NAME:-stackcube_pos_act_map4d_keyframe_tcp_${TCP_TARGET}_aux_990demos_seed1}"
LOG_FILE="${LOG_FILE:-outputs/train_logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "$LOG_FILE")"

TRAIN_ARGS=(
  --exp-name "${RUN_NAME}"
  --seed 1
  --env-id StackCube-v1
  --demo-path "${DEMO_PATH}"
  --num-demos "${NUM_DEMOS}"
  --control-mode pd_ee_delta_pos
  --sim-backend physx_cpu
  --no-include-depth
  --max-episode-steps 1000
  --total_iters "${TOTAL_ITERS}"
  --batch-size "${BATCH_SIZE}"
  --num-queries 30
  --kl-weight 10
  --lr 1e-4
  --lr-backbone 1e-5
  --log_freq 100
  --eval_freq 10000
  --num-eval-episodes 100
  --num-eval-envs 10
  --num-dataload-workers 0
  --no-track
  --no-capture-video
  --use-map4d
  --map4d-raw-concat
  --map4d-keyframe-aux-loss
  --map4d-keyframe-aux-path "${KEYFRAME_AUX_PATH}"
  --map4d-aux-weight 1.0
  --map4d-source maniskill_gt
  --map4d-task-name StackCube-v1
  --map4d-pre-horizon 30
  --map4d-future-horizon "${FUTURE_HORIZON}"
  --map4d-num-objects 3
  --map4d-tcp-dim "${TCP_DIM}"
)

if [[ "${NO_CUDA:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-cuda)
fi
if [[ "${NO_EVAL:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-eval)
fi
if [[ "${NO_BACKBONE_PRETRAINED:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-backbone-pretrained)
fi

echo "[$(date +%H:%M:%S)] running ${RUN_NAME} -> ${LOG_FILE}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'cd %s\npython train_rgbd_map4d.py' "$ACT_DIR"
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi
python train_rgbd_map4d.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
