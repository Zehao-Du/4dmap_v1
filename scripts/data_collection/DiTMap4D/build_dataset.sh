#!/usr/bin/env bash
# One-click dataset builder for standalone DiTMap/Map4D DiT training.
#
# Usage:
#   bash scripts/data_collection/DiTMap4D/build_dataset.sh <task> <demos> <vision_model> [resolution]
#
# Examples:
#   bash scripts/data_collection/DiTMap4D/build_dataset.sh stackcube 100 dinov3_vits16 224
#   SKIP_COLLECT=1 DINOV3_WEIGHTS_PATH=/path/to/dinov3_vits16.pth \
#     bash scripts/data_collection/DiTMap4D/build_dataset.sh plugcharger 1000 dinov3_vits16 224
#
# Outputs:
#   1) filtered ManiSkill demo, collected by scripts/data_collection/collect_*.sh
#   2) train demo with per-camera obs/dino_feature/<camera> when vision_model is dinov3_*
#      resolution controls ManiSkill replay camera size when collecting, and
#      DINO input resize when generating features
#   3) Map4D DiT keyframe sidecar with target_format=map4d_dit_local_delta_relative_rotation_v1
#   4) a .env manifest with paths and suggested Hydra overrides
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <task> <demos> <vision_model> [resolution]" >&2
  echo "  task: stackcube|plugcharger|StackCube-v1|PlugCharger-v1" >&2
  echo "  demos: positive integer, e.g. 100 or 1000" >&2
  echo "  vision_model: dinov3_vits16|dinov3_vits16plus|dinov3_vitb16|dinov3_vitl16|dinov3_vith16plus|none" >&2
  echo "  resolution: native|SIZE|HEIGHTxWIDTH, default 224" >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap}"
POLICY_DIR="${POLICY_DIR:-${ROOT_DIR}/4dmap_policy}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
NUM_PROCS="${NUM_PROCS:-10}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
OVERWRITE="${OVERWRITE:-1}"
MAX_STEPS="${MAX_STEPS:-400}"
GRIPPER_SOURCE="${GRIPPER_SOURCE:-auto}"
STOPPING_DELTA="${STOPPING_DELTA:-0.1}"
MIN_SEPARATION="${MIN_SEPARATION:-1}"
CAMERAS="${CAMERAS:-auto}"
OBS_MODE="${OBS_MODE:-rgb+depth}"
BUILD_POINTCLOUD="${BUILD_POINTCLOUD:-1}"
POINTCLOUD_NUM_POINTS="${POINTCLOUD_NUM_POINTS:-6144}"
POINTCLOUD_BBOX="${POINTCLOUD_BBOX:-auto}"
POINTCLOUD_SEED="${POINTCLOUD_SEED:-0}"
DINOV3_BATCH_SIZE="${DINOV3_BATCH_SIZE:-64}"
DINOV3_DEVICE="${DINOV3_DEVICE:-auto}"
DINOV3_POOL="${DINOV3_POOL:-patch_mean}"
DINOV3_THIRD_PARTY_DIR="${DINOV3_THIRD_PARTY_DIR:-${POLICY_DIR}/third_party/dinov3}"
TARGET_FORMAT="map4d_dit_local_delta_relative_rotation_v1"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

RAW_TASK="$1"
DEMOS="$2"
VISION_MODEL="${3,,}"
RESOLUTION="${4:-224}"
RESOLUTION="${RESOLUTION,,}"

if [[ ! "$DEMOS" =~ ^[0-9]+$ || "$DEMOS" -le 0 ]]; then
  echo "Invalid demos=${DEMOS}. Use a positive integer." >&2
  exit 1
fi

case "$RESOLUTION" in
  native|original|none)
    RESOLUTION="native"
    RESOLUTION_TAG="native"
    DINOV3_IMAGE_SIZE_ARG=()
    ;;
  [0-9]*)
    if [[ "$RESOLUTION" =~ ^[0-9]+$ ]]; then
      RESOLUTION_TAG="${RESOLUTION}"
      DINOV3_IMAGE_SIZE_ARG=(--image-size "$RESOLUTION")
    elif [[ "$RESOLUTION" =~ ^[0-9]+x[0-9]+$ ]]; then
      RESOLUTION_TAG="${RESOLUTION//x/-}"
      DINOV3_IMAGE_SIZE_ARG=(--image-size "$RESOLUTION")
    else
      echo "Invalid resolution=${RESOLUTION}. Use native, SIZE, or HEIGHTxWIDTH." >&2
      exit 1
    fi
    ;;
  *)
    echo "Invalid resolution=${RESOLUTION}. Use native, SIZE, or HEIGHTxWIDTH." >&2
    exit 1
    ;;
esac

case "${RAW_TASK,,}" in
  stackcube|stackcube-v1)
    TASK_KEY="stackcube"
    TASK_NAME="StackCube-v1"
    TRAJ_NAME="StackCube"
    DEFAULT_CONTROL_MODE="pd_ee_delta_pos"
    COLLECT_SCRIPT="${POLICY_DIR}/scripts/data_collection/collect_stackcube.sh"
    TASK_OVERRIDE="task=stackcube_map4d_dit"
    DIT_TCP_TARGET="pos_gripper"
    DIT_TARGET_FORMAT="map4d_dit_local_delta_relative_rotation_tcp_pos_gripper_v1"
    ;;
  plugcharger|plugcharger-v1)
    TASK_KEY="plugcharger"
    TASK_NAME="PlugCharger-v1"
    TRAJ_NAME="PlugCharger"
    DEFAULT_CONTROL_MODE="pd_ee_delta_pose"
    COLLECT_SCRIPT="${POLICY_DIR}/scripts/data_collection/collect_plugcharger.sh"
    TASK_OVERRIDE="task=plugcharger_map4d_dit"
    DIT_TCP_TARGET="pose"
    DIT_TARGET_FORMAT="${TARGET_FORMAT}"
    ;;
  *)
    echo "Unsupported task=${RAW_TASK}. Use stackcube or plugcharger." >&2
    exit 1
    ;;
esac

CONTROL_MODE="${CONTROL_MODE:-${DEFAULT_CONTROL_MODE}}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/ManiSkill/${TASK_NAME}/motionplanning}"
FILTERED_DEMO_PATH="${DEMO_PATH:-${DATASET_DIR}/${TRAJ_NAME}.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.filtered.h5}"

case "$VISION_MODEL" in
  dino|dinov3)
    VISION_MODEL="dinov3_vits16"
    ;;
esac

case "$VISION_MODEL" in
  dinov3_vits16|dinov3_vits16plus|dinov3_vitb16|dinov3_vitl16|dinov3_vith16plus)
    USE_RGB="true"
    ;;
  none|no_vision)
    VISION_MODEL="none"
    USE_RGB="false"
    ;;
  *)
    echo "Unsupported vision_model=${VISION_MODEL}." >&2
    echo "Use a dinov3_* model or none." >&2
    exit 1
    ;;
esac

if [[ "$SKIP_COLLECT" != "1" ]]; then
  if [[ "$CONTROL_MODE" != "$DEFAULT_CONTROL_MODE" ]]; then
    echo "CONTROL_MODE=${CONTROL_MODE} does not match ${TASK_NAME} collector output (${DEFAULT_CONTROL_MODE})." >&2
    echo "Use SKIP_COLLECT=1 with an explicit DEMO_PATH for custom control modes." >&2
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] Collecting ${DEMOS} ${TASK_NAME} demos"
  echo "  collector: ${COLLECT_SCRIPT}"
  echo "  max_steps: ${MAX_STEPS}"
  echo "  obs_mode: ${OBS_MODE}"
  echo "  stored_image_resolution: ${RESOLUTION}"
  bash "$COLLECT_SCRIPT" "$DEMOS" "$NUM_PROCS" "$RESOLUTION"
else
  echo "[$(date +%H:%M:%S)] SKIP_COLLECT=1; reusing filtered demo"
  if [[ "$RESOLUTION" != "native" ]]; then
    echo "  note: existing demo image resolution is not changed; ${RESOLUTION} is applied during DINO feature extraction"
  fi
fi

if [[ ! -f "$FILTERED_DEMO_PATH" ]]; then
  echo "Filtered demo not found: ${FILTERED_DEMO_PATH}" >&2
  echo "Pass DEMO_PATH explicitly, or run without SKIP_COLLECT." >&2
  exit 1
fi

TRAJ_COUNT="$(
  python -c 'import h5py,sys; f=h5py.File(sys.argv[1],"r"); print(sum(k.startswith("traj_") for k in f.keys())); f.close()' \
    "$FILTERED_DEMO_PATH"
)"
if [[ "$TRAJ_COUNT" -lt "$DEMOS" ]]; then
  echo "Filtered demo only has ${TRAJ_COUNT} trajectories, but demos=${DEMOS}." >&2
  exit 1
fi

TRAIN_DEMO_PATH="$FILTERED_DEMO_PATH"
FEATURE_SUMMARY_JSON=""
RGB_FEATURE_DIM="0"

if [[ "$VISION_MODEL" == dinov3_* ]]; then
  TRAIN_DEMO_PATH="${OUTPUT_DEMO_PATH:-${FILTERED_DEMO_PATH%.h5}.with_${VISION_MODEL}_${RESOLUTION_TAG}.h5}"
  FEATURE_SUMMARY_JSON="${FEATURE_SUMMARY_JSON:-${TRAIN_DEMO_PATH%.h5}.summary.json}"
  dinov3_cmd=(
    python "${POLICY_DIR}/scripts/data_collection/build_dinov3_features.py"
    --demo-path "${FILTERED_DEMO_PATH}"
    --output-path "${TRAIN_DEMO_PATH}"
    --summary-json "${FEATURE_SUMMARY_JSON}"
    --model "${VISION_MODEL}"
    --third-party-dir "${DINOV3_THIRD_PARTY_DIR}"
    --cameras "${CAMERAS}"
    --batch-size "${DINOV3_BATCH_SIZE}"
    --num-traj "${DEMOS}"
    --device "${DINOV3_DEVICE}"
    --pool "${DINOV3_POOL}"
    --embed-in-output-demo
  )
  if [[ "${#DINOV3_IMAGE_SIZE_ARG[@]}" -gt 0 ]]; then
    dinov3_cmd+=("${DINOV3_IMAGE_SIZE_ARG[@]}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    dinov3_cmd+=(--overwrite)
  fi
  if [[ -n "${DINOV3_WEIGHTS_PATH:-}" ]]; then
    dinov3_cmd+=(--weights-path "${DINOV3_WEIGHTS_PATH}")
  fi

  echo "[$(date +%H:%M:%S)] Generating ${VISION_MODEL} features"
  echo "  input_demo: ${FILTERED_DEMO_PATH}"
  echo "  train_demo: ${TRAIN_DEMO_PATH}"
  echo "  dinov3_input_resolution: ${RESOLUTION}"
  "${dinov3_cmd[@]}"

  RGB_FEATURE_DIM="$(
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["feature_dim"] if "feature_dim" in d else d["trajectories"][0]["concat_shape"][-1])' \
      "$FEATURE_SUMMARY_JSON"
  )"
else
  echo "[$(date +%H:%M:%S)] vision_model=none; no obs/dino_feature will be generated"
fi

POINTCLOUD_SUMMARY_JSON=""
if [[ "${BUILD_POINTCLOUD}" == "1" ]]; then
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

  echo "[$(date +%H:%M:%S)] Building PPI-style point clouds"
  echo "  train_demo: ${TRAIN_DEMO_PATH}"
  echo "  cameras: ${CAMERAS}"
  echo "  points: ${POINTCLOUD_NUM_POINTS}"
  "${pointcloud_cmd[@]}"
else
  echo "[$(date +%H:%M:%S)] BUILD_POINTCLOUD=0; no obs/point_cloud will be generated"
fi

SIDECAR_PATH="${SIDECAR_PATH:-${TRAIN_DEMO_PATH%.h5}.map4d_dit_h${FUTURE_HORIZON}.h5}"
SIDECAR_SUMMARY_JSON="${SIDECAR_SUMMARY_JSON:-${SIDECAR_PATH%.h5}.summary.json}"
sidecar_cmd=(
  python "${POLICY_DIR}/helper/build_keyframe_aux_dataset.py"
  --demo-path "${TRAIN_DEMO_PATH}"
  --output-path "${SIDECAR_PATH}"
  --summary-json "${SIDECAR_SUMMARY_JSON}"
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
  sidecar_cmd+=(--overwrite)
fi

echo "[$(date +%H:%M:%S)] Building DiTMap keyframe sidecar"
echo "  train_demo: ${TRAIN_DEMO_PATH}"
echo "  sidecar: ${SIDECAR_PATH}"
"${sidecar_cmd[@]}"

MANIFEST_PATH="${MANIFEST_PATH:-${SIDECAR_PATH%.h5}.env}"
cat > "$MANIFEST_PATH" <<EOF
# Generated by scripts/data_collection/DiTMap4D/build_dataset.sh
TASK_NAME=${TASK_NAME}
TASK_OVERRIDE=${TASK_OVERRIDE}
VISION_MODEL=${VISION_MODEL}
DINOV3_INPUT_RESOLUTION=${RESOLUTION}
MANISKILL_RGB_RESOLUTION=${RESOLUTION}
MANISKILL_OBS_MODE=${OBS_MODE}
MAP4D_DEMO_PATH=${TRAIN_DEMO_PATH}
MAP4D_KEYFRAME_SIDECAR_PATH=${SIDECAR_PATH}
MAP4D_NUM_TRAJ=${DEMOS}
FUTURE_HORIZON=${FUTURE_HORIZON}
TCP_TARGET=${DIT_TCP_TARGET}
TARGET_FORMAT=${DIT_TARGET_FORMAT}
USE_RGB=${USE_RGB}
RGB_FEATURE_DIM=${RGB_FEATURE_DIM}
FEATURE_SUMMARY_JSON=${FEATURE_SUMMARY_JSON}
BUILD_POINTCLOUD=${BUILD_POINTCLOUD}
POINTCLOUD_NUM_POINTS=${POINTCLOUD_NUM_POINTS}
POINTCLOUD_BBOX=${POINTCLOUD_BBOX}
POINTCLOUD_SUMMARY_JSON=${POINTCLOUD_SUMMARY_JSON}
SIDECAR_SUMMARY_JSON=${SIDECAR_SUMMARY_JSON}
EOF

echo "[$(date +%H:%M:%S)] Done"
echo "  train_demo: ${TRAIN_DEMO_PATH}"
echo "  sidecar: ${SIDECAR_PATH}"
echo "  manifest: ${MANIFEST_PATH}"
echo
echo "Training example:"
echo "  source ${MANIFEST_PATH}"
echo "  MAP4D_DEMO_PATH=\"\$MAP4D_DEMO_PATH\" MAP4D_KEYFRAME_SIDECAR_PATH=\"\$MAP4D_KEYFRAME_SIDECAR_PATH\" MAP4D_NUM_TRAJ=\"\$MAP4D_NUM_TRAJ\" \\"
echo "    bash scripts/map4d_backbone/run_map4d_dit_train.sh --config-name map4d_dit \"\$TASK_OVERRIDE\" \\"
echo "    policy.model_cfg.use_rgb=${USE_RGB} policy.model_cfg.rgb_feature_dim=${RGB_FEATURE_DIM}"
