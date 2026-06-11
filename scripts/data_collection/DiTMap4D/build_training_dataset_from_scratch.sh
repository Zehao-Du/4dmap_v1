#!/usr/bin/env bash
# One-click builder for the current Map4DDiT training dataset.
#
# Usage:
#   bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh <task> <demos> [resolution]
#
# Examples:
#   bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh stackcube 100 224
#   SKIP_COLLECT=1 DEMO_PATH=/path/to/demo_with_per_point_dino.h5 \
#     bash scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh plugcharger 1000 native
#
# Outputs:
#   1) filtered ManiSkill RGB-D demo
#   2) in-place PPI-style fused point cloud: traj_*/obs/point_cloud/fused [T,P,6]
#   3) generate/validate per-point DINO Semantic Field feature: traj_*/obs/dino_feature [T,P,D_sem]
#   4) keyframe sidecar with node-pose and TCP targets
#   5) final sidecar copied from keyframe sidecar; GT map is kept as traj_*/map4d
#   6) .env manifest with paths and training overrides
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <task> <demos> [resolution]" >&2
  echo "  task: stackcube|plugcharger|StackCube-v1|PlugCharger-v1" >&2
  echo "  demos: positive integer, e.g. 100 or 1000" >&2
  echo "  resolution: native|SIZE|WIDTHxHEIGHT, default 224" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${POLICY_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(cd "${POLICY_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset}"

FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
NUM_PROCS="${NUM_PROCS:-10}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
OVERWRITE="${OVERWRITE:-1}"
GRIPPER_SOURCE="${GRIPPER_SOURCE:-auto}"
STOPPING_DELTA="${STOPPING_DELTA:-0.1}"
MIN_SEPARATION="${MIN_SEPARATION:-1}"
CAMERAS="${CAMERAS:-auto}"
OBS_MODE="${OBS_MODE:-rgb+depth}"
POINTCLOUD_NUM_POINTS="${POINTCLOUD_NUM_POINTS:-6144}"
POINTCLOUD_BBOX="${POINTCLOUD_BBOX:-auto}"
POINTCLOUD_SEED="${POINTCLOUD_SEED:-0}"
SEMANTIC_FEATURE_MODE="${SEMANTIC_FEATURE_MODE:-dinov3}"
SEMANTIC_FEATURE_PATH="${SEMANTIC_FEATURE_PATH:-obs/dino_feature}"
DINOV3_MODEL="${DINOV3_MODEL:-dinov3_vits16}"
DINOV3_WEIGHTS_PATH="${DINOV3_WEIGHTS_PATH:-}"
DINOV3_THIRD_PARTY_DIR="${DINOV3_THIRD_PARTY_DIR:-${POLICY_DIR}/map4d/backbone/model/vision/dinov3}"
DINOV3_BATCH_SIZE="${DINOV3_BATCH_SIZE:-64}"
DINOV3_DEVICE="${DINOV3_DEVICE:-auto}"
DINOV3_MULTIPLE="${DINOV3_MULTIPLE:-16}"
DINOV3_NO_AMP="${DINOV3_NO_AMP:-0}"
SEMANTIC_FEATURE_DTYPE="${SEMANTIC_FEATURE_DTYPE:-float32}"
SEMANTIC_DINO_IMAGE_SIZE="${SEMANTIC_DINO_IMAGE_SIZE:-}"
MAP_FEATURE_DIM="${MAP_FEATURE_DIM:-240}"
NO_MATERIALIZE_TARGETS="${NO_MATERIALIZE_TARGETS:-0}"
TARGET_FORMAT_POSE="map4d_dit_local_delta_relative_rotation_v1"
TARGET_FORMAT_POS_GRIPPER="map4d_dit_local_delta_relative_rotation_tcp_pos_gripper_v1"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

RAW_TASK="$1"
DEMOS="$2"
RESOLUTION="${3:-224}"
RESOLUTION="${RESOLUTION,,}"

if [[ ! "$DEMOS" =~ ^[0-9]+$ || "$DEMOS" -le 0 ]]; then
  echo "Invalid demos=${DEMOS}. Use a positive integer." >&2
  exit 1
fi

case "$RESOLUTION" in
  native|original|none)
    RESOLUTION="native"
    RESOLUTION_TAG="native"
    ;;
  [0-9]*)
    if [[ "$RESOLUTION" =~ ^[0-9]+$ ]]; then
      RESOLUTION_TAG="${RESOLUTION}"
    elif [[ "$RESOLUTION" =~ ^[0-9]+x[0-9]+$ ]]; then
      RESOLUTION_TAG="${RESOLUTION//x/-}"
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

case "${SEMANTIC_FEATURE_MODE,,}" in
  dinov3|existing_dino)
    SEMANTIC_FEATURE_MODE="${SEMANTIC_FEATURE_MODE,,}"
    ;;
  *)
    echo "Unsupported SEMANTIC_FEATURE_MODE=${SEMANTIC_FEATURE_MODE}. Use dinov3 or existing_dino." >&2
    exit 1
    ;;
esac
if [[ -z "${SEMANTIC_DINO_IMAGE_SIZE}" && "${RESOLUTION}" != "native" ]]; then
  SEMANTIC_DINO_IMAGE_SIZE="${RESOLUTION}"
fi

case "${RAW_TASK,,}" in
  stackcube|stackcube-v1)
    TASK_KEY="stackcube"
    TASK_NAME="StackCube-v1"
    TRAJ_NAME="StackCube"
    DEFAULT_CONTROL_MODE="pd_ee_delta_pos"
    COLLECT_SCRIPT="${POLICY_DIR}/scripts/data_collection/collect_stackcube.sh"
    TASK_OVERRIDE="task=stackcube_map4d_dit"
    DIT_TCP_TARGET="pos_gripper"
    DIT_TARGET_FORMAT="${TARGET_FORMAT_POS_GRIPPER}"
    ;;
  plugcharger|plugcharger-v1)
    TASK_KEY="plugcharger"
    TASK_NAME="PlugCharger-v1"
    TRAJ_NAME="PlugCharger"
    DEFAULT_CONTROL_MODE="pd_ee_delta_pose"
    COLLECT_SCRIPT="${POLICY_DIR}/scripts/data_collection/collect_plugcharger.sh"
    TASK_OVERRIDE="task=plugcharger_map4d_dit"
    DIT_TCP_TARGET="pose"
    DIT_TARGET_FORMAT="${TARGET_FORMAT_POSE}"
    ;;
  *)
    echo "Unsupported task=${RAW_TASK}. Use stackcube or plugcharger." >&2
    exit 1
    ;;
esac

CONTROL_MODE="${CONTROL_MODE:-${DEFAULT_CONTROL_MODE}}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/ManiSkill/${TASK_NAME}/motionplanning}"
RECORD_DIR="${RECORD_DIR:-${DATA_ROOT}/ManiSkill}"
FILTERED_DEMO_PATH="${DEMO_PATH:-${DATASET_DIR}/${TRAJ_NAME}.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.filtered.h5}"
TRAIN_DEMO_PATH="${OUTPUT_DEMO_PATH:-${FILTERED_DEMO_PATH}}"
export ROOT_DIR POLICY_DIR DATA_ROOT DATASET_DIR RECORD_DIR OBS_MODE

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  if [[ "${CONTROL_MODE}" != "${DEFAULT_CONTROL_MODE}" ]]; then
    echo "CONTROL_MODE=${CONTROL_MODE} does not match ${TASK_NAME} collector output (${DEFAULT_CONTROL_MODE})." >&2
    echo "Use SKIP_COLLECT=1 with an explicit DEMO_PATH for custom control modes." >&2
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] Collecting ${DEMOS} ${TASK_NAME} demos"
  echo "  collector: ${COLLECT_SCRIPT}"
  echo "  obs_mode: ${OBS_MODE}"
  echo "  resolution: ${RESOLUTION}"
  bash "${COLLECT_SCRIPT}" "${DEMOS}" "${NUM_PROCS}" "${RESOLUTION}"
else
  echo "[$(date +%H:%M:%S)] SKIP_COLLECT=1; reusing demo"
fi

if [[ ! -f "${FILTERED_DEMO_PATH}" ]]; then
  echo "Filtered demo not found: ${FILTERED_DEMO_PATH}" >&2
  echo "Pass DEMO_PATH explicitly, or run without SKIP_COLLECT." >&2
  exit 1
fi

if [[ "${TRAIN_DEMO_PATH}" != "${FILTERED_DEMO_PATH}" ]]; then
  if [[ -e "${TRAIN_DEMO_PATH}" && "${OVERWRITE}" != "1" ]]; then
    echo "OUTPUT_DEMO_PATH exists: ${TRAIN_DEMO_PATH}. Set OVERWRITE=1 to replace it." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${TRAIN_DEMO_PATH}")"
  cp "${FILTERED_DEMO_PATH}" "${TRAIN_DEMO_PATH}"
  JSON_SRC="${FILTERED_DEMO_PATH%.h5}.json"
  JSON_DST="${TRAIN_DEMO_PATH%.h5}.json"
  if [[ -f "${JSON_SRC}" ]]; then
    cp "${JSON_SRC}" "${JSON_DST}"
  fi
fi

TRAJ_COUNT="$(
  python -c 'import h5py,sys; f=h5py.File(sys.argv[1],"r"); print(sum(k.startswith("traj_") for k in f.keys())); f.close()' \
    "${TRAIN_DEMO_PATH}"
)"
if [[ "${TRAJ_COUNT}" -lt "${DEMOS}" ]]; then
  echo "Demo only has ${TRAJ_COUNT} trajectories, but demos=${DEMOS}." >&2
  exit 1
fi

POINTCLOUD_SUMMARY_JSON="${POINTCLOUD_SUMMARY_JSON:-${TRAIN_DEMO_PATH%.h5}.pointcloud.summary.json}"
pointcloud_cmd=(
  python "${POLICY_DIR}/scripts/data_collection/build_pointcloud_dataset.py"
  --demo-path "${TRAIN_DEMO_PATH}"
  --summary-json "${POINTCLOUD_SUMMARY_JSON}"
  --task-name "${TASK_NAME}"
  --cameras "${CAMERAS}"
  --num-points "${POINTCLOUD_NUM_POINTS}"
  --bbox "${POINTCLOUD_BBOX}"
  --num-traj "${DEMOS}"
  --seed "${POINTCLOUD_SEED}"
  --in-place
)
if [[ "${OVERWRITE}" == "1" ]]; then
  pointcloud_cmd+=(--overwrite)
fi

echo "[$(date +%H:%M:%S)] Building fused point clouds"
echo "  demo: ${TRAIN_DEMO_PATH}"
echo "  points: ${POINTCLOUD_NUM_POINTS}"
"${pointcloud_cmd[@]}"

SEMANTIC_SUMMARY_JSON="${SEMANTIC_SUMMARY_JSON:-${TRAIN_DEMO_PATH%.h5}.semantic_field_${SEMANTIC_FEATURE_MODE}.summary.json}"
semantic_cmd=(
  python "${POLICY_DIR}/scripts/data_collection/DiTMap4D/build_point_semantic_features.py"
  --demo-path "${TRAIN_DEMO_PATH}"
  --mode "${SEMANTIC_FEATURE_MODE}"
  --pointcloud-path "obs/point_cloud/fused"
  --pointcloud-source-path "obs/point_cloud_source/fused"
  --output-path "${SEMANTIC_FEATURE_PATH}"
  --summary-json "${SEMANTIC_SUMMARY_JSON}"
  --num-traj "${DEMOS}"
)
if [[ "${SEMANTIC_FEATURE_MODE}" == "dinov3" ]]; then
  semantic_cmd+=(
    --model "${DINOV3_MODEL}"
    --third-party-dir "${DINOV3_THIRD_PARTY_DIR}"
    --batch-size "${DINOV3_BATCH_SIZE}"
    --device "${DINOV3_DEVICE}"
    --multiple "${DINOV3_MULTIPLE}"
    --output-dtype "${SEMANTIC_FEATURE_DTYPE}"
  )
  if [[ -n "${DINOV3_WEIGHTS_PATH}" ]]; then
    semantic_cmd+=(--weights-path "${DINOV3_WEIGHTS_PATH}")
  fi
  if [[ -n "${SEMANTIC_DINO_IMAGE_SIZE}" ]]; then
    semantic_cmd+=(--image-size "${SEMANTIC_DINO_IMAGE_SIZE}")
  fi
  if [[ "${DINOV3_NO_AMP}" == "1" ]]; then
    semantic_cmd+=(--no-amp)
  fi
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  semantic_cmd+=(--overwrite)
fi

echo "[$(date +%H:%M:%S)] Preparing per-point DINO Semantic Field features"
echo "  mode: ${SEMANTIC_FEATURE_MODE}"
echo "  path: ${SEMANTIC_FEATURE_PATH}"
if [[ "${SEMANTIC_FEATURE_MODE}" == "dinov3" ]]; then
  echo "  model: ${DINOV3_MODEL}"
  echo "  image_size: ${SEMANTIC_DINO_IMAGE_SIZE:-auto_multiple}"
fi
"${semantic_cmd[@]}"

SEMANTIC_FEATURE_DIM="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["semantic_feature_dim"])' \
    "${SEMANTIC_SUMMARY_JSON}"
)"

KEYFRAME_SIDECAR_PATH="${KEYFRAME_SIDECAR_PATH:-${TRAIN_DEMO_PATH%.h5}.map4d_dit_h${FUTURE_HORIZON}.keyframe.h5}"
KEYFRAME_SUMMARY_JSON="${KEYFRAME_SUMMARY_JSON:-${KEYFRAME_SIDECAR_PATH%.h5}.summary.json}"
keyframe_cmd=(
  python "${POLICY_DIR}/helper/build_keyframe_aux_dataset.py"
  --demo-path "${TRAIN_DEMO_PATH}"
  --output-path "${KEYFRAME_SIDECAR_PATH}"
  --summary-json "${KEYFRAME_SUMMARY_JSON}"
  --task-name "${TASK_NAME}"
  --future-horizon "${FUTURE_HORIZON}"
  --tcp-target "${DIT_TCP_TARGET}"
  --target-format "${DIT_TARGET_FORMAT}"
  --gripper-source "${GRIPPER_SOURCE}"
  --stopping-delta "${STOPPING_DELTA}"
  --min-separation "${MIN_SEPARATION}"
  --num-traj "${DEMOS}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  keyframe_cmd+=(--overwrite)
fi
if [[ "${NO_MATERIALIZE_TARGETS}" == "1" ]]; then
  keyframe_cmd+=(--no-materialize-targets)
fi

echo "[$(date +%H:%M:%S)] Building keyframe target sidecar"
echo "  sidecar: ${KEYFRAME_SIDECAR_PATH}"
"${keyframe_cmd[@]}"

FINAL_SIDECAR_PATH="${FINAL_SIDECAR_PATH:-${TRAIN_DEMO_PATH%.h5}.map4d_dit_h${FUTURE_HORIZON}.context.h5}"
CONTEXT_SUMMARY_JSON="${CONTEXT_SUMMARY_JSON:-${FINAL_SIDECAR_PATH%.h5}.summary.json}"
context_cmd=(
  python "${POLICY_DIR}/scripts/data_collection/DiTMap4D/build_map4d_context_dataset.py"
  --demo-path "${TRAIN_DEMO_PATH}"
  --input-sidecar-path "${KEYFRAME_SIDECAR_PATH}"
  --output-path "${FINAL_SIDECAR_PATH}"
  --summary-json "${CONTEXT_SUMMARY_JSON}"
  --task-name "${TASK_NAME}"
  --num-traj "${DEMOS}"
  --pointcloud-path "obs/point_cloud/fused"
  --dino-feature-path "${SEMANTIC_FEATURE_PATH}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  context_cmd+=(--overwrite)
fi

echo "[$(date +%H:%M:%S)] Building Map4D context sidecar"
echo "  final_sidecar: ${FINAL_SIDECAR_PATH}"
"${context_cmd[@]}"

NUM_MAP_NODES="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_map_nodes"])' \
    "${CONTEXT_SUMMARY_JSON}"
)"
MANIFEST_PATH="${MANIFEST_PATH:-${FINAL_SIDECAR_PATH%.h5}.env}"
mkdir -p "$(dirname "${MANIFEST_PATH}")"
{
  echo "# Generated by scripts/data_collection/DiTMap4D/build_training_dataset_from_scratch.sh"
  printf 'TASK_NAME=%q\n' "${TASK_NAME}"
  printf 'TASK_KEY=%q\n' "${TASK_KEY}"
  printf 'TASK_OVERRIDE=%q\n' "${TASK_OVERRIDE}"
  printf 'MAP4D_DEMO_PATH=%q\n' "${TRAIN_DEMO_PATH}"
  printf 'MAP4D_KEYFRAME_SIDECAR_PATH=%q\n' "${FINAL_SIDECAR_PATH}"
  printf 'MAP4D_RAW_KEYFRAME_SIDECAR_PATH=%q\n' "${KEYFRAME_SIDECAR_PATH}"
  printf 'MAP4D_NUM_TRAJ=%q\n' "${DEMOS}"
  printf 'FUTURE_HORIZON=%q\n' "${FUTURE_HORIZON}"
  printf 'TCP_TARGET=%q\n' "${DIT_TCP_TARGET}"
  printf 'TARGET_FORMAT=%q\n' "${DIT_TARGET_FORMAT}"
  printf 'SEMANTIC_FEATURE_MODE=%q\n' "${SEMANTIC_FEATURE_MODE}"
  printf 'SEMANTIC_FEATURE_PATH=%q\n' "${SEMANTIC_FEATURE_PATH}"
  printf 'SEMANTIC_FEATURE_DIM=%q\n' "${SEMANTIC_FEATURE_DIM}"
  printf 'SEMANTIC_FEATURE_DTYPE=%q\n' "${SEMANTIC_FEATURE_DTYPE}"
  printf 'DINOV3_MODEL=%q\n' "${DINOV3_MODEL}"
  printf 'POINTCLOUD_NUM_POINTS=%q\n' "${POINTCLOUD_NUM_POINTS}"
  printf 'POINTCLOUD_BBOX=%q\n' "${POINTCLOUD_BBOX}"
  printf 'MAP_FEATURE_DIM=%q\n' "${MAP_FEATURE_DIM}"
  printf 'NUM_MAP_NODES=%q\n' "${NUM_MAP_NODES}"
  printf 'MANISKILL_OBS_MODE=%q\n' "${OBS_MODE}"
  printf 'MANISKILL_RGB_RESOLUTION=%q\n' "${RESOLUTION_TAG}"
  printf 'POINTCLOUD_SUMMARY_JSON=%q\n' "${POINTCLOUD_SUMMARY_JSON}"
  printf 'SEMANTIC_SUMMARY_JSON=%q\n' "${SEMANTIC_SUMMARY_JSON}"
  printf 'KEYFRAME_SUMMARY_JSON=%q\n' "${KEYFRAME_SUMMARY_JSON}"
  printf 'CONTEXT_SUMMARY_JSON=%q\n' "${CONTEXT_SUMMARY_JSON}"
} > "${MANIFEST_PATH}"

echo "[$(date +%H:%M:%S)] Done"
echo "  demo: ${TRAIN_DEMO_PATH}"
echo "  final_sidecar: ${FINAL_SIDECAR_PATH}"
echo "  manifest: ${MANIFEST_PATH}"
echo
echo "Training example:"
echo "  source ${MANIFEST_PATH}"
echo "  MAP4D_DEMO_PATH=\"\$MAP4D_DEMO_PATH\" MAP4D_KEYFRAME_SIDECAR_PATH=\"\$MAP4D_KEYFRAME_SIDECAR_PATH\" MAP4D_NUM_TRAJ=\"\$MAP4D_NUM_TRAJ\" \\"
echo "    bash scripts/map4d_backbone/run_map4d_dit_train.sh --config-name map4d_dit \"\$TASK_OVERRIDE\" \\"
echo "    policy.model_cfg.semantic_feature_dim=${SEMANTIC_FEATURE_DIM} policy.model_cfg.map_feature_dim=${MAP_FEATURE_DIM} policy.model_cfg.num_map_nodes=${NUM_MAP_NODES}"
