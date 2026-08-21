#!/usr/bin/env bash
# Prepare an RLBench2 dataset directory/manifest for Map4DDiT training.
#
# Given an RLBench2 squashfs, this builder can extract the needed raw files,
# build point-cloud and feature npy files, and write an .env manifest. For
# bimanual_push_box it also builds the Map4D pose sidecar used by the current
# RLBench2Map4DDataset.
#
# Usage:
#   Edit TASKS below, then run:
#   bash scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh
#   bash scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh --overwrite all
#   bash scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh --overwrite dino
#
# Examples:
#   bash scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh
#
# Important env vars:
#   RLBENCH2_OUTPUT_ROOT=/path/to/output/root
#   RLBENCH2_EPISODES=100
#   RLBENCH2_PCD_TYPE=rgb_pcd_rps6144
#   RLBENCH2_DINO_PATH=/path/to/real/dino_feature/root
#   RLBENCH2_LANG_EMB_PATH=/path/to/instruction_embeddings.pkl
#   DINOV2_REPO_PATH=/path/to/PPI/repos/dinov2
#   DINOV2_WEIGHTS_PATH=/path/to/dinov2_vits14_pretrain.pth
#   DINO_DEVICE=cuda:0


# =============== Tasks =============== #
TASKS=(
  bimanual_push_box
  # bimanual_pick_plate
)

# =============== Target Directory =============== #
TARGET_DIR="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/dataset/rlbench2/map4d_dit"

# =============== Source Directory =============== #
SOURCE_DIR="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/dataset/rlbench2"

# =============== DINOv2 Checkpoint =============== #
DINOV2_REPO_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/PPI/repos/dinov2"
DINOV2_WEIGHTS_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/foundation_models/DINOv2/dinov2_vits14_pretrain.pth"
DINO_DEVICE="${DINO_DEVICE:-cuda:0}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-36}"


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${POLICY_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(cd "${POLICY_DIR}/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)}"
COPPELIASIM_ROOT="${MAP4D_COPPELIASIM_ROOT:-${PROJECT_ROOT}/codes/CoppeliaSim}"
RLBENCH_ROOT="${MAP4D_RLBENCH_ROOT:-${PROJECT_ROOT}/codes/rlbench}"
PYREP_ROOT="${MAP4D_PYREP_ROOT:-${PROJECT_ROOT}/codes/pyrep}"

OVERWRITE_TARGETS="${OVERWRITE_TARGETS:-}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --overwrite)
      if [[ "$#" -lt 2 || "${2}" == --* ]]; then
        echo "--overwrite requires a target: all, dino, raw, point_cloud, pose, or manifest" >&2
        exit 2
      fi
      OVERWRITE_TARGETS="${2}"
      shift 2
      ;;
    --overwrite=*)
      OVERWRITE_TARGETS="${1#--overwrite=}"
      shift
      ;;
    *)
      echo "This script no longer accepts positional task arguments." >&2
      echo "Edit TASKS under '# =============== Tasks =============== #' instead." >&2
      exit 2
      ;;
  esac
done

normalize_overwrite_targets() {
  local value="$1"
  local normalized=""
  local item
  value="${value// /}"
  IFS=',' read -r -a items <<< "${value}"
  for item in "${items[@]}"; do
    [[ -z "${item}" ]] && continue
    case "${item}" in
      all|dino|raw|point_cloud|pcd|pose|manifest)
        if [[ -n "${normalized}" ]]; then
          normalized+=","
        fi
        normalized+="${item}"
        ;;
      *)
        echo "Unsupported --overwrite target: ${item}" >&2
        echo "Supported targets: all, dino, raw, point_cloud, pcd, pose, manifest" >&2
        exit 2
        ;;
    esac
  done
  echo "${normalized}"
}

overwrite_requested() {
  local target="$1"
  local item
  IFS=',' read -r -a items <<< "${OVERWRITE_TARGETS}"
  for item in "${items[@]}"; do
    if [[ "${item}" == "all" || "${item}" == "${target}" ]]; then
      return 0
    fi
    if [[ "${target}" == "point_cloud" && "${item}" == "pcd" ]]; then
      return 0
    fi
  done
  return 1
}

OVERWRITE_TARGETS="$(normalize_overwrite_targets "${OVERWRITE_TARGETS}")"
export OVERWRITE_TARGETS

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP_RLBENCH2_DIT:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP_RLBENCH2_DIT=1
  exec conda run --no-capture-output -n 4dmap bash "$0"
fi

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No RLBench2 tasks configured. Edit TASKS in this script." >&2
  exit 2
fi

if [[ ! -f "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]]; then
  echo "CoppeliaSim library not found: ${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" >&2
  exit 1
fi
if [[ ! -d "${RLBENCH_ROOT}/rlbench" ]]; then
  echo "RLBench Python package not found: ${RLBENCH_ROOT}/rlbench" >&2
  exit 1
fi
if [[ ! -d "${PYREP_ROOT}/pyrep" ]]; then
  echo "PyRep Python package not found: ${PYREP_ROOT}/pyrep" >&2
  exit 1
fi
export PYTHONPATH="${RLBENCH_ROOT}:${PYREP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

PCD_TYPE="${RLBENCH2_PCD_TYPE:-rgb_pcd_rps6144}"
EPISODES="${RLBENCH2_EPISODES:-100}"
if [[ ! "${EPISODES}" =~ ^[0-9]+$ || "${EPISODES}" -le 0 ]]; then
  echo "Invalid episodes=${EPISODES}. Use a positive integer." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-${SOURCE_DIR}}"
OUTPUT_ROOT="${RLBENCH2_OUTPUT_ROOT:-${TARGET_DIR}}"
OUTPUT_ROOT="$(mkdir -p "${OUTPUT_ROOT}" && cd "${OUTPUT_ROOT}" && pwd)"
PPI_PROCESSED_ROOT="${PPI_PROCESSED_ROOT:-${ROOT_DIR}/PPI/data/data/Open-PPI/training_processed}"
LANG_EMB_PATH="${RLBENCH2_LANG_EMB_PATH:-${PPI_PROCESSED_ROOT}/instruction_embeddings.pkl}"

FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
PREDICTION_TYPE="${RLBENCH2_PREDICTION_TYPE:-continuous}"
VAL_RATIO="${RLBENCH2_VAL_RATIO:-0.05}"
START_EP="${RLBENCH2_START:-0}"
END_EP="${RLBENCH2_END:-$((EPISODES - 1))}"
MAX_TRAIN_EPISODES="${RLBENCH2_MAX_TRAIN_EPISODES:-}"
NUM_MAP_NODES="1"
MAP_FEATURE_DIM="${MAP_FEATURE_DIM:-240}"
SEMANTIC_FEATURE_DIM="${SEMANTIC_FEATURE_DIM:-384}"
POINTCLOUD_NUM_POINTS="${POINTCLOUD_NUM_POINTS:-}"
STRICT_VISUAL_CHECK="${STRICT_VISUAL_CHECK:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_USE_RGB="${SMOKE_USE_RGB:-1}"
OVERWRITE_POSE="${OVERWRITE_POSE:-0}"
BUILD_VISUAL_FEATURES="${BUILD_VISUAL_FEATURES:-1}"
VISUAL_MAX_EPISODES="${VISUAL_MAX_EPISODES:-${EPISODES}}"
VISUAL_MAX_FRAMES="${VISUAL_MAX_FRAMES:-}"
VISUAL_FEATURE_MODE="${VISUAL_FEATURE_MODE:-dinov2}"
VISUAL_CAMERAS="${VISUAL_CAMERAS:-front,over_shoulder_left,over_shoulder_right,overhead,wrist_left,wrist_right}"

remove_output_path() {
  local path="$1"
  if [[ -z "${path}" || "${path}" == "/" ]]; then
    echo "Refusing to remove unsafe path: ${path}" >&2
    exit 2
  fi
  case "${path}" in
    "${OUTPUT_ROOT}"/*)
      rm -rf "${path}"
      ;;
    *)
      echo "Refusing to overwrite path outside target root: ${path}" >&2
      echo "Target root: ${OUTPUT_ROOT}" >&2
      exit 2
      ;;
  esac
}

build_one_task() {
local TASK="$1"
local SQUASHFS="${DATA_ROOT}/${TASK}.train.squashfs"
local OUTPUT_DIR
OUTPUT_DIR="$(mkdir -p "${OUTPUT_ROOT}/${TASK}" && cd "${OUTPUT_ROOT}/${TASK}" && pwd)"
local RAW_DIR="${OUTPUT_DIR}/raw/all_variations/episodes"
local POSE_PATH="${OUTPUT_DIR}/${TASK}_train_poses.npz"
local POSE_CSV_PATH="${OUTPUT_DIR}/${TASK}_train_poses.csv"
local PCD_PATH="${OUTPUT_DIR}/point_cloud"
local DINO_PATH="${RLBENCH2_DINO_PATH:-${OUTPUT_DIR}/dino_feature}"
local POINT_FLOW_PATH="${OUTPUT_DIR}/point_flow"
local TASK_OVERRIDE="task=rlbench2_push_box_map4d_dit"
local MAP_NAME
if [[ "${TASK}" == "bimanual_push_box" ]]; then
  MAP_NAME="rlbench2_push_box"
else
  MAP_NAME="rlbench2_${TASK#bimanual_}"
  TASK_OVERRIDE=""
fi
local MANIFEST_DIR="${OUTPUT_DIR}"
local MANIFEST_PATH="${MANIFEST_DIR}/${MAP_NAME}_${EPISODES}eps_${PCD_TYPE}_h${FUTURE_HORIZON}.env"
local SUMMARY_PATH="${MANIFEST_PATH%.env}.summary.txt"

if overwrite_requested "all"; then
  echo "[$(date +%H:%M:%S)] Overwriting all task outputs: ${OUTPUT_DIR}"
  remove_output_path "${OUTPUT_DIR}"
elif [[ -n "${OVERWRITE_TARGETS}" ]]; then
  if overwrite_requested "raw"; then
    echo "[$(date +%H:%M:%S)] Overwriting raw data: ${OUTPUT_DIR}/raw"
    remove_output_path "${OUTPUT_DIR}/raw"
  fi
  if overwrite_requested "point_cloud"; then
    echo "[$(date +%H:%M:%S)] Overwriting point cloud data: ${PCD_PATH}"
    remove_output_path "${PCD_PATH}"
  fi
  if overwrite_requested "dino"; then
    echo "[$(date +%H:%M:%S)] Overwriting DINO feature data: ${DINO_PATH}"
    remove_output_path "${DINO_PATH}"
  fi
  if overwrite_requested "pose"; then
    echo "[$(date +%H:%M:%S)] Overwriting pose sidecar: ${POSE_PATH}"
    remove_output_path "${POSE_PATH}"
    remove_output_path "${POSE_CSV_PATH}"
  fi
  if overwrite_requested "manifest"; then
    echo "[$(date +%H:%M:%S)] Overwriting manifest files: ${MANIFEST_PATH}"
    remove_output_path "${MANIFEST_PATH}"
    remove_output_path "${SUMMARY_PATH}"
  fi
fi

mkdir -p "$(dirname "${MANIFEST_PATH}")" "$(dirname "${SUMMARY_PATH}")"

echo "[$(date +%H:%M:%S)] Preparing RLBench2 Map4DDiT dataset manifest"
echo "  task: ${TASK}"
echo "  episodes: ${EPISODES}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  raw_dir: ${RAW_DIR}"
echo "  squashfs: ${SQUASHFS}"
echo "  pcd_path: ${PCD_PATH}"
echo "  dino_path: ${DINO_PATH}"
echo "  pcd_type: ${PCD_TYPE}"
echo "  pose_path: ${POSE_PATH}"
echo "  dinov2_repo_path: ${DINOV2_REPO_PATH}"
echo "  dinov2_weights_path: ${DINOV2_WEIGHTS_PATH}"
echo "  dino_device: ${DINO_DEVICE}"
echo "  dino_batch_size: ${DINO_BATCH_SIZE}"

if [[ ! -f "${LANG_EMB_PATH}" ]]; then
  echo "Instruction embedding file not found: ${LANG_EMB_PATH}" >&2
  echo "Set RLBENCH2_LANG_EMB_PATH." >&2
  exit 1
fi

if [[ "${BUILD_VISUAL_FEATURES}" == "1" ]]; then
  visual_cmd=(
    python "${POLICY_DIR}/scripts/data_collection/rlbench2_map4d_dit/build_visual_features.py"
    --input "${SQUASHFS}"
    --output-dir "${OUTPUT_DIR}"
    --pcd-type "${PCD_TYPE}"
    --num-points "${POINTCLOUD_NUM_POINTS:-6144}"
    --feature-mode "${VISUAL_FEATURE_MODE}"
    --dinov2-repo-path "${DINOV2_REPO_PATH}"
    --dinov2-weights-path "${DINOV2_WEIGHTS_PATH}"
    --dino-device "${DINO_DEVICE}"
    --dino-batch-size "${DINO_BATCH_SIZE}"
    --cameras "${VISUAL_CAMERAS}"
    --max-episodes "${VISUAL_MAX_EPISODES:-${EPISODES}}"
  )
  if [[ -n "${VISUAL_MAX_FRAMES}" ]]; then
    visual_cmd+=(--max-frames "${VISUAL_MAX_FRAMES}")
  fi
  echo "[$(date +%H:%M:%S)] Building visual point-cloud/features"
  "${visual_cmd[@]}"
fi

if [[ ! -d "${RAW_DIR}" ]]; then
  echo "Raw RLBench2 episode dir not found: ${RAW_DIR}" >&2
  echo "Set RLBENCH2_RAW_DIR, or enable BUILD_VISUAL_FEATURES=1 for ${SQUASHFS}." >&2
  exit 1
fi

if [[ "${TASK}" == "bimanual_push_box" && ("${OVERWRITE_POSE}" == "1" || ! -f "${POSE_PATH}") ]]; then
  POSE_INPUT="${SQUASHFS}"
  if [[ ! -f "${POSE_INPUT}" ]]; then
    POSE_INPUT="${DATA_ROOT}"
  fi
  echo "[$(date +%H:%M:%S)] Extracting push-box Map4D poses"
  python "${POLICY_DIR}/scripts/data_collection/rlbench2/extract_push_box_poses.py" \
    --input "${POSE_INPUT}" \
    --output "${POSE_PATH}" \
    --csv "${POSE_CSV_PATH}"
else
  if [[ "${TASK}" == "bimanual_push_box" ]]; then
    echo "[$(date +%H:%M:%S)] Reusing pose sidecar: ${POSE_PATH}"
  else
    echo "[$(date +%H:%M:%S)] Skipping Map4D pose sidecar for ${TASK}; only bimanual_push_box is wired today."
  fi
fi

if [[ "${TASK}" == "bimanual_push_box" && ! -f "${POSE_PATH}" ]]; then
  echo "Pose sidecar was not created: ${POSE_PATH}" >&2
  exit 1
fi

if [[ -f "${POSE_PATH}" ]]; then
  POSE_INFO="$(
  python - "${POSE_PATH}" <<'PY'
import numpy as np
import sys
path = sys.argv[1]
data = np.load(path)
episodes = len(np.unique(data["episode"]))
frames = len(data["frame"])
size = data["size_xyz"].tolist() if "size_xyz" in data.files else []
print(f"episodes={episodes} frames={frames} size_xyz={size}")
PY
)"
else
  POSE_INFO="missing"
fi
echo "  pose: ${POSE_INFO}"

if [[ -d "${PCD_PATH}" ]]; then
  FIRST_PCD="$(find "${PCD_PATH}" -path "*/${PCD_TYPE}/step*.npy" -type f -print -quit)"
else
  FIRST_PCD=""
fi
if [[ -d "${DINO_PATH}" ]]; then
  FIRST_DINO="$(find "${DINO_PATH}" -path "*/${PCD_TYPE}/step*.npy" -type f -print -quit)"
else
  FIRST_DINO=""
fi

if [[ -n "${FIRST_PCD}" ]]; then
  POINTCLOUD_NUM_POINTS="$(
    python - "${FIRST_PCD}" <<'PY'
import numpy as np
import sys
arr = np.load(sys.argv[1], mmap_mode="r")
print(arr.shape[-2] if arr.ndim >= 2 else arr.shape[0])
PY
  )"
  echo "  first point cloud: ${FIRST_PCD}"
  echo "  pointcloud_num_points: ${POINTCLOUD_NUM_POINTS}"
else
  echo "No point cloud npy found under ${PCD_PATH}/*/${PCD_TYPE}/step*.npy" >&2
  echo "Point clouds are required; no fallback is allowed." >&2
  exit 1
fi

if [[ -n "${FIRST_DINO}" ]]; then
  SEMANTIC_FEATURE_DIM="$(
    python - "${FIRST_DINO}" <<'PY'
import numpy as np
import sys
arr = np.load(sys.argv[1], mmap_mode="r")
print(arr.shape[-1])
PY
  )"
  python - "${FIRST_PCD}" "${FIRST_DINO}" <<'PY'
import numpy as np
import sys

pcd_path, dino_path = sys.argv[1:3]
pcd = np.load(pcd_path, mmap_mode="r")
dino = np.load(dino_path, mmap_mode="r")
if dino.ndim != 2:
    raise ValueError(f"DINO feature must have shape [P,D], got {dino.shape} from {dino_path}")
if dino.shape[0] != pcd.shape[0]:
    raise ValueError(
        f"Point cloud and DINO feature must share point count, got {pcd.shape} and {dino.shape}"
    )
if dino.shape[-1] <= 3:
    raise ValueError(f"DINO feature dim must be >3, got {dino.shape[-1]} from {dino_path}")
if np.allclose(dino, 0.0):
    raise ValueError(f"{dino_path} is all zeros; fake semantic fallback is not allowed")
if pcd.ndim == 2 and pcd.shape[1] >= 6:
    rgb = np.asarray(pcd[:, 3:6], dtype=np.float32)
    reps = int(np.ceil(dino.shape[1] / 3))
    rgb_tile = np.tile(rgb, (1, reps))[:, : dino.shape[1]]
    if np.allclose(np.asarray(dino, dtype=np.float32), rgb_tile, atol=1e-6):
        raise ValueError(
            f"{dino_path} is RGB tiled to {dino.shape[1]} dims, not DINO. "
            "Generate real DINO features and set RLBENCH2_DINO_PATH."
        )
print("DINO feature provenance check passed")
PY
  echo "  first DINO feature: ${FIRST_DINO}"
  echo "  semantic_feature_dim: ${SEMANTIC_FEATURE_DIM}"
else
  echo "No DINO npy found under ${DINO_PATH}/*/${PCD_TYPE}/step*.npy" >&2
  echo "Real DINO features are required; no RGB/zero fallback is allowed." >&2
  echo "Set RLBENCH2_DINO_PATH to a directory containing episode*/${PCD_TYPE}/step*.npy." >&2
  exit 1
fi

{
  echo "# Generated by scripts/data_collection/rlbench2_map4d_dit/build_training_dataset.sh"
  printf 'TASK_NAME=%q\n' "${MAP_NAME}"
  printf 'TASK_KEY=%q\n' "rlbench2_push_box"
  printf 'TASK_OVERRIDE=%q\n' "${TASK_OVERRIDE}"
  printf 'RLBENCH2_TASK=%q\n' "${TASK}"
  printf 'RLBENCH2_DATA_PATH=%q\n' "${RAW_DIR}"
  printf 'RLBENCH2_SQUASHFS=%q\n' "${SQUASHFS}"
  printf 'RLBENCH2_POSE_PATH=%q\n' "${POSE_PATH}"
  printf 'RLBENCH2_POSE_CSV_PATH=%q\n' "${POSE_CSV_PATH}"
  printf 'RLBENCH2_PCD_PATH=%q\n' "${PCD_PATH}"
  printf 'RLBENCH2_DINO_PATH=%q\n' "${DINO_PATH}"
  printf 'RLBENCH2_LANG_EMB_PATH=%q\n' "${LANG_EMB_PATH}"
  printf 'RLBENCH2_POINT_FLOW_PATH=%q\n' "${POINT_FLOW_PATH}"
  printf 'RLBENCH2_PCD_TYPE=%q\n' "${PCD_TYPE}"
  printf 'RLBENCH2_PREDICTION_TYPE=%q\n' "${PREDICTION_TYPE}"
  printf 'RLBENCH2_START=%q\n' "${START_EP}"
  printf 'RLBENCH2_END=%q\n' "${END_EP}"
  printf 'RLBENCH2_VAL_RATIO=%q\n' "${VAL_RATIO}"
  if [[ -n "${MAX_TRAIN_EPISODES}" ]]; then
    printf 'RLBENCH2_MAX_TRAIN_EPISODES=%q\n' "${MAX_TRAIN_EPISODES}"
  fi
  printf 'MAP4D_NUM_TRAJ=%q\n' "${EPISODES}"
  printf 'FUTURE_HORIZON=%q\n' "${FUTURE_HORIZON}"
  printf 'SEMANTIC_FEATURE_MODE=%q\n' "precomputed"
  printf 'TRAIN_SEMANTIC_FEATURE_MODE=%q\n' "precomputed"
  printf 'SEMANTIC_FEATURE_DIM=%q\n' "${SEMANTIC_FEATURE_DIM}"
  printf 'MAP_FEATURE_DIM=%q\n' "${MAP_FEATURE_DIM}"
  printf 'NUM_MAP_NODES=%q\n' "${NUM_MAP_NODES}"
  printf 'POINTCLOUD_NUM_POINTS=%q\n' "${POINTCLOUD_NUM_POINTS}"
  printf 'POSE_INFO=%q\n' "${POSE_INFO}"
} > "${MANIFEST_PATH}"

{
  echo "manifest=${MANIFEST_PATH}"
  echo "task_override=${TASK_OVERRIDE}"
  echo "raw_dir=${RAW_DIR}"
  echo "pose_path=${POSE_PATH}"
  echo "pose_info=${POSE_INFO}"
  echo "pcd_path=${PCD_PATH}"
  echo "dino_path=${DINO_PATH}"
  echo "pcd_type=${PCD_TYPE}"
  echo "semantic_feature_dim=${SEMANTIC_FEATURE_DIM}"
  echo "pointcloud_num_points=${POINTCLOUD_NUM_POINTS}"
} > "${SUMMARY_PATH}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  if [[ "${TASK}" != "bimanual_push_box" ]]; then
    echo "[$(date +%H:%M:%S)] Running visual-file smoke check for ${TASK}"
    python - "${RAW_DIR}" "${PCD_PATH}" "${DINO_PATH}" "${PCD_TYPE}" <<'PY'
import sys
from pathlib import Path
import numpy as np
raw_dir, pcd_path, dino_path, pcd_type = map(Path, sys.argv[1:])
first_ep = sorted(raw_dir.glob("episode*"))[0]
ep = first_ep.name
pcd = sorted((pcd_path / ep / str(pcd_type)).glob("step*.npy"))[0]
dino = sorted((dino_path / ep / str(pcd_type)).glob("step*.npy"))[0]
pcd_arr = np.load(pcd, mmap_mode="r")
dino_arr = np.load(dino, mmap_mode="r")
print(f"raw_episode={first_ep}")
print(f"point_cloud={pcd} shape={pcd_arr.shape} dtype={pcd_arr.dtype}")
print(f"dino_feature={dino} shape={dino_arr.shape} dtype={dino_arr.dtype}")
PY
  else
    echo "[$(date +%H:%M:%S)] Running RLBench2Map4DDataset smoke check"
  python - "${RAW_DIR}" "${PCD_PATH}" "${DINO_PATH}" "${LANG_EMB_PATH}" "${POSE_PATH}" "${PCD_TYPE}" "${START_EP}" "${END_EP}" "${PREDICTION_TYPE}" "${SMOKE_USE_RGB}" <<'PY'
import sys
import numpy as np
from map4d.backbone.dataset.rlbench2_map4d_dataset import RLBench2Map4DDataset

raw_dir, pcd_path, dino_path, lang_path, pose_path, pcd_type, start, end, prediction_type, smoke_use_rgb = sys.argv[1:]
ds = RLBench2Map4DDataset(
    data_path=raw_dir,
    pcd_path=pcd_path,
    dino_path=dino_path,
    lang_emb_path=lang_path,
    pose_path=pose_path,
    start=int(start),
    end=min(int(end), int(start)),
    val_ratio=0.0,
    max_train_episodes=1,
    prediction_type=prediction_type,
    pcd_type=pcd_type,
    horizon_action=4,
    horizon_keyframe=2,
    n_obs_steps=2,
    action_type="bimanual_ee_pose",
    use_rgb=smoke_use_rgb == "1",
)
raw_state = np.asarray(ds.replay_buffer["state"], dtype=np.float32)
raw_action = np.asarray(ds.replay_buffer["action"], dtype=np.float32)
ds._validate_ppi_bimanual_layout(raw_state, field_name="smoke robot_state")
ds._validate_ppi_bimanual_layout(raw_action, field_name="smoke action")

trajectory = ds.trajectories[0]
if not np.array_equal(trajectory["robot_state"][0], raw_state[0]):
    raise AssertionError("Map4D changed PPI robot_state values or ordering")
formatted_action = trajectory["actions"][0]
if formatted_action.shape != (2, 8):
    raise AssertionError(f"Expected Map4D action shape (2,8), got {formatted_action.shape}")
if not np.allclose(formatted_action[0, 0:3], raw_action[0, 0:3]):
    raise AssertionError("Map4D left-arm action does not match PPI action[0:3]")
if not np.allclose(formatted_action[1, 0:3], raw_action[0, 7:10]):
    raise AssertionError("Map4D right-arm action does not match PPI action[7:10]")
if not np.allclose(formatted_action[:, 7], raw_action[0, 14:16]):
    raise AssertionError("Map4D gripper actions do not match PPI action[14:16]")

sample = ds[0]
print("ppi_layout=left_pose(7),right_pose(7),left_open,right_open verified")
print(f"smoke_len={len(ds)} trajectories={len(ds.trajectories)}")
for group_name, group in sample.items():
    if isinstance(group, dict):
        for key, value in group.items():
            print(f"{group_name}.{key}={tuple(value.shape)}")
PY
  fi
fi

echo "[$(date +%H:%M:%S)] Done"
echo "  manifest: ${MANIFEST_PATH}"
echo "  summary: ${SUMMARY_PATH}"
echo
echo "Training example:"
if [[ "${TASK}" == "bimanual_push_box" ]]; then
  echo "  cd ${POLICY_DIR}"
  echo "  source ${MANIFEST_PATH}"
  echo "  python map4d/backbone/train_map4d_dit.py --config-name map4d_dit \"\${TASK_OVERRIDE}\" \\"
  echo "    policy.model_cfg.semantic_feature_dim=\${SEMANTIC_FEATURE_DIM} \\"
  echo "    policy.model_cfg.map_feature_dim=\${MAP_FEATURE_DIM} \\"
  echo "    policy.model_cfg.num_map_nodes=\${NUM_MAP_NODES}"
else
  echo "  ${TASK} visual dataset is ready, but Map4D pose/representation config is not wired yet."
  echo "  Add a task config and pose sidecar, then source ${MANIFEST_PATH} for paths."
fi
}

for TASK in "${TASKS[@]}"; do
  if [[ -z "${TASK}" ]]; then
    echo "Empty task entry in TASKS is not allowed." >&2
    exit 2
  fi
  build_one_task "${TASK}"
done
