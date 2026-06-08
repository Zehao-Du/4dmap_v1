#!/bin/bash
# Build sidecar data for 4.4 Raw Concat + Keyframe Future Loss + TCP targets.
# Input format matches collect_plugcharger.sh / collect_stackcube.sh outputs.
set -e
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data2/zehao/MAP4D}"
POLICY_DIR="${POLICY_DIR:-${ROOT_DIR}/4dmap_v1}"
TASK_NAME="${TASK_NAME:-PlugCharger-v1}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
NUM_TRAJ="${NUM_TRAJ:-}"
CONTROL_MODE="${CONTROL_MODE:-auto}"
TCP_TARGET="${TCP_TARGET:-pose}"
GRIPPER_SOURCE="${GRIPPER_SOURCE:-auto}"
STOPPING_DELTA="${STOPPING_DELTA:-0.1}"
MIN_SEPARATION="${MIN_SEPARATION:-1}"
OVERWRITE="${OVERWRITE:-1}"
NO_MATERIALIZE_TARGETS="${NO_MATERIALIZE_TARGETS:-0}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

case "${TASK_NAME}" in
  PlugCharger-v1)
    TRAJ_NAME="PlugCharger"
    ;;
  StackCube-v1)
    TRAJ_NAME="StackCube"
    ;;
  *)
    echo "Unsupported TASK_NAME=${TASK_NAME}. Set TRAJ_NAME and DEMO_PATH explicitly." >&2
    TRAJ_NAME="${TRAJ_NAME:-${TASK_NAME%-v1}}"
    ;;
esac

DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/dataset/ManiSkill/${TASK_NAME}/motionplanning}"
DEFAULT_CONTROL_MODES=()
case "${CONTROL_MODE}" in
  auto)
    DEFAULT_CONTROL_MODES=(pd_ee_delta_pose pd_ee_delta_pos)
    ;;
  pd_ee_delta_pose|pd_ee_delta_pos)
    DEFAULT_CONTROL_MODES=("${CONTROL_MODE}")
    ;;
  *)
    echo "Unsupported CONTROL_MODE=${CONTROL_MODE}. Use auto, pd_ee_delta_pose, or pd_ee_delta_pos." >&2
    exit 1
    ;;
esac

case "${TCP_TARGET}" in
  pose|pos)
    ;;
  *)
    echo "Unsupported TCP_TARGET=${TCP_TARGET}. Use pose or pos." >&2
    exit 1
    ;;
esac

if [[ "${TCP_TARGET}" == "pose" ]]; then
  TARGET_FORMAT="object_delta_pos_rot6d_plus_tcp_pose"
else
  TARGET_FORMAT="object_delta_pos_rot6d_plus_tcp_pos"
fi

if [[ $# -ge 1 ]]; then
  DEMO_PATH="$1"
else
  DEMO_PATH=""
  for mode in "${DEFAULT_CONTROL_MODES[@]}"; do
    candidate="${DATASET_DIR}/${TRAJ_NAME}.rgb.${mode}.physx_cpu.filtered.h5"
    if [[ -f "${candidate}" ]]; then
      DEMO_PATH="${candidate}"
      break
    fi
  done
  if [[ -z "${DEMO_PATH}" ]]; then
    echo "Filtered demo file not found for CONTROL_MODE=${CONTROL_MODE} under ${DATASET_DIR}" >&2
    for mode in "${DEFAULT_CONTROL_MODES[@]}"; do
      echo "  tried: ${DATASET_DIR}/${TRAJ_NAME}.rgb.${mode}.physx_cpu.filtered.h5" >&2
    done
    echo "Run scripts/data_collection/collect_plugcharger.sh first, or pass an explicit DEMO_PATH as arg 1." >&2
    exit 1
  fi
fi

if [[ ! -f "${DEMO_PATH}" ]]; then
  echo "Demo file not found: ${DEMO_PATH}" >&2
  echo "Run scripts/data_collection/collect_plugcharger.sh first, or pass DEMO_PATH as arg 1." >&2
  exit 1
fi

if [[ $# -ge 2 ]]; then
  OUTPUT_PATH="$2"
elif [[ "${TCP_TARGET}" == "pose" ]]; then
  OUTPUT_PATH="${DEMO_PATH%.h5}.keyframe_aux_h${FUTURE_HORIZON}.h5"
else
  OUTPUT_PATH="${DEMO_PATH%.h5}.keyframe_aux_tcp_pos_h${FUTURE_HORIZON}.h5"
fi
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_PATH%.h5}.summary.json}"

cmd=(
  python "${POLICY_DIR}/helper/build_keyframe_aux_dataset.py"
  --demo-path "${DEMO_PATH}"
  --output-path "${OUTPUT_PATH}"
  --summary-json "${SUMMARY_JSON}"
  --task-name "${TASK_NAME}"
  --future-horizon "${FUTURE_HORIZON}"
  --tcp-target "${TCP_TARGET}"
  --target-format "${TARGET_FORMAT}"
  --gripper-source "${GRIPPER_SOURCE}"
  --stopping-delta "${STOPPING_DELTA}"
  --min-separation "${MIN_SEPARATION}"
)

if [[ -n "${NUM_TRAJ}" ]]; then
  cmd+=(--num-traj "${NUM_TRAJ}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${NO_MATERIALIZE_TARGETS}" == "1" ]]; then
  cmd+=(--no-materialize-targets)
fi

echo "[$(date +%H:%M:%S)] Building keyframe aux dataset"
echo "  task: ${TASK_NAME}"
echo "  input: ${DEMO_PATH}"
echo "  output: ${OUTPUT_PATH}"
echo "  summary: ${SUMMARY_JSON}"
echo "  future_horizon: ${FUTURE_HORIZON}"
echo "  control_mode: ${CONTROL_MODE}"
echo "  tcp_target: ${TCP_TARGET}"
echo "  target_format: ${TARGET_FORMAT}"
"${cmd[@]}"
