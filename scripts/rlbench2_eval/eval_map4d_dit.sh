#!/usr/bin/env bash
# Evaluate a Map4D DiT checkpoint on an RLBench2 task.
#
# Usage:
#   Edit TASKS below, then run:
#   bash scripts/rlbench2_eval/eval_map4d_dit.sh
#
# Optional env overrides:
#   MAP4D_DIT_CKPT=/path/to/checkpoint.pth.tar
#   RLBENCH2_TEST_DEMO_PATH=/path/to/task/test/or/dataset/root
#   EVAL_OUTPUT_DIR=/path/to/eval/output
#   GPU=0 EVAL_EPISODES=1 EVAL_SEED=0
#   EPISODE_LENGTH=300 NUM_INFERENCE_STEPS=1000
#   SAVE_VIDEO=true MAP4D_VIDEO_CAMERA=front MAP4D_VIDEO_FPS=10
#   MAP4D_VIDEO_PATH=/path/to/map4d_overlay.mp4 EVAL_SAVE_METRICS=true


# =============== Tasks =============== #
TASKS=(
  bimanual_push_box
  # bimanual_pick_plate
)

# =============== Experiment / Checkpoint =============== #
EXPERIMENT_DIR="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/exp_logs/map4d_dit/bimanual_push_box/train_map4d_dit_rlbench2_push_box_debug_seed0"
CHECKPOINT_NAME="epoch=0400-val_loss=1.0467659.pth.tar"
CHECKPOINT_PATH="${MAP4D_DIT_CKPT:-${EXPERIMENT_DIR}/checkpoints/${CHECKPOINT_NAME}}"

# =============== Test Dataset =============== #
TEST_DEMO_PATH="${RLBENCH2_TEST_DEMO_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/dataset/rlbench2/squashfs-root}"

# =============== Foundation Model =============== #
DINOV2_REPO_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/PPI/repos/dinov2"
DINOV2_WEIGHTS_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/foundation_models/DINOv2/dinov2_vits14_pretrain.pth"

# =============== Evaluation =============== #
GPU="${GPU:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
EPISODE_LENGTH="${EPISODE_LENGTH:-300}"
QUERY_FREQ="${QUERY_FREQ:-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1000}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
MAP4D_VIDEO_CAMERA="${MAP4D_VIDEO_CAMERA:-front}"
MAP4D_VIDEO_FPS="${MAP4D_VIDEO_FPS:-10}"
EVAL_SAVE_METRICS="${EVAL_SAVE_METRICS:-true}"


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${POLICY_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(cd "${POLICY_DIR}/.." && pwd)}"
RLBENCH2_DIR="${POLICY_DIR}/third_party/rlbench2"
COPPELIASIM_ROOT="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/codes/CoppeliaSim"

if [[ "$#" -ne 0 ]]; then
  echo "This script does not accept task arguments." >&2
  echo "Edit TASKS under '# =============== Tasks =============== #' instead." >&2
  exit 2
fi
if [[ "${#TASKS[@]}" -ne 1 || -z "${TASKS[0]}" ]]; then
  echo "Configure exactly one non-empty RLBench2 task in TASKS." >&2
  echo "Each checkpoint evaluation must use its matching task." >&2
  exit 2
fi
TASK="${TASKS[0]}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${POLICY_DIR}/outputs/rlbench2_map4d_dit_eval/${TASK}/seed${EVAL_SEED}}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP_RLBENCH2_EVAL:-0}" != "1" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found; the 4dmap environment is required." >&2
    exit 1
  fi
  export RUNNING_IN_4DMAP_RLBENCH2_EVAL=1
  exec conda run --no-capture-output -n 4dmap bash "$0"
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" ]]; then
  echo "Failed to enter the required 4dmap conda environment." >&2
  exit 1
fi

require_integer() {
  local name="$1"
  local value="$2"
  local allow_zero="$3"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Invalid ${name}=${value}; expected an integer." >&2
    exit 2
  fi
  if [[ "${allow_zero}" != "1" && "${value}" -le 0 ]]; then
    echo "Invalid ${name}=${value}; expected a positive integer." >&2
    exit 2
  fi
}

require_boolean() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "Invalid ${name}=${value}; expected true or false." >&2
    exit 2
  fi
}

require_integer "GPU" "${GPU}" 1
require_integer "EVAL_EPISODES" "${EVAL_EPISODES}" 0
require_integer "EVAL_SEED" "${EVAL_SEED}" 1
require_integer "EPISODE_LENGTH" "${EPISODE_LENGTH}" 0
require_integer "QUERY_FREQ" "${QUERY_FREQ}" 0
require_integer "NUM_INFERENCE_STEPS" "${NUM_INFERENCE_STEPS}" 0
require_integer "MAP4D_VIDEO_FPS" "${MAP4D_VIDEO_FPS}" 0
require_boolean "SAVE_VIDEO" "${SAVE_VIDEO}"
require_boolean "EVAL_SAVE_METRICS" "${EVAL_SAVE_METRICS}"
case "${MAP4D_VIDEO_CAMERA}" in
  front|overhead|over_shoulder_left|over_shoulder_right|wrist_left|wrist_right) ;;
  *)
    echo "Unsupported MAP4D_VIDEO_CAMERA=${MAP4D_VIDEO_CAMERA}." >&2
    exit 2
    ;;
esac

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Map4D DiT checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${TEST_DEMO_PATH}/all_variations/episodes" && ! -d "${TEST_DEMO_PATH}/${TASK}/all_variations/episodes" ]]; then
  echo "RLBench2 test episodes for ${TASK} were not found under: ${TEST_DEMO_PATH}" >&2
  echo "Set RLBENCH2_TEST_DEMO_PATH to an RLBench2 dataset root or extracted task directory." >&2
  exit 1
fi
if [[ ! -f "${DINOV2_REPO_PATH}/hubconf.py" ]]; then
  echo "DINOv2 repository hubconf.py not found: ${DINOV2_REPO_PATH}/hubconf.py" >&2
  exit 1
fi
if [[ ! -f "${DINOV2_WEIGHTS_PATH}" ]]; then
  echo "DINOv2 checkpoint not found: ${DINOV2_WEIGHTS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]]; then
  echo "CoppeliaSim library not found: ${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" >&2
  exit 1
fi
if [[ ! -f "${COPPELIASIM_ROOT}/platforms/libqxcb.so" ]]; then
  echo "CoppeliaSim Qt plugin not found: ${COPPELIASIM_ROOT}/platforms/libqxcb.so" >&2
  exit 1
fi
if [[ ! -f "${RLBENCH2_DIR}/eval_map4d_dit.py" ]]; then
  echo "RLBench2 evaluator not found: ${RLBENCH2_DIR}/eval_map4d_dit.py" >&2
  exit 1
fi
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run was not found. Install the system packages xvfb and xauth." >&2
  exit 1
fi

OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
VIDEO_DIR="${OUTPUT_DIR}/videos"
MAP4D_VIDEO_PATH="${MAP4D_VIDEO_PATH:-${VIDEO_DIR}/map4d_overlay.mp4}"
if [[ "${SAVE_VIDEO}" == "true" ]]; then
  mkdir -p "$(dirname "${MAP4D_VIDEO_PATH}")"
fi
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-${UID}}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

echo "[$(date +%H:%M:%S)] Evaluating RLBench2 Map4D DiT checkpoint"
echo "  task: ${TASK}"
echo "  checkpoint: ${CHECKPOINT_PATH}"
echo "  test_demo_path: ${TEST_DEMO_PATH}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  gpu: ${GPU}"
echo "  eval_episodes: ${EVAL_EPISODES}"
echo "  eval_seed: ${EVAL_SEED}"
echo "  episode_length: ${EPISODE_LENGTH}"
echo "  num_inference_steps: ${NUM_INFERENCE_STEPS}"
echo "  save_map4d_video: ${SAVE_VIDEO}"
if [[ "${SAVE_VIDEO}" == "true" ]]; then
  echo "  map4d_video_path: ${MAP4D_VIDEO_PATH}"
  echo "  map4d_video_camera: ${MAP4D_VIDEO_CAMERA}"
fi

eval_args=(
  "framework.eval_from_eps_number=${EVAL_SEED}"
  "framework.start_seed=${EVAL_SEED}"
  "framework.eval_episodes=${EVAL_EPISODES}"
  "framework.eval_type=0"
  "framework.eval_envs=1"
  "framework.gpu=${GPU}"
  "framework.logdir=${OUTPUT_DIR}"
  "framework.eval_save_metrics=${EVAL_SAVE_METRICS}"
  "rlbench.headless=true"
  "rlbench.episode_length=${EPISODE_LENGTH}"
  "rlbench.task_name=${TASK}"
  "rlbench.tasks=[${TASK}]"
  "rlbench.demo_path=${TEST_DEMO_PATH}"
  "rlbench.query_freq=${QUERY_FREQ}"
  "method.policy.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "method.dinov2_repo_path=${DINOV2_REPO_PATH}"
  "method.dinov2_weights_path=${DINOV2_WEIGHTS_PATH}"
  "method.semantic_feature_source=fusion"
  "method.map_pose_source=simulator"
  "method.map_object_name=cube"
  "cinematic_recorder.enabled=false"
)
if [[ "${SAVE_VIDEO}" == "true" ]]; then
  eval_args+=(
    "method.debug_map_video_path=${MAP4D_VIDEO_PATH}"
    "method.debug_map_video_camera=${MAP4D_VIDEO_CAMERA}"
    "method.debug_map_video_fps=${MAP4D_VIDEO_FPS}"
  )
fi

cd "${RLBENCH2_DIR}"
BASE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
EVAL_LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${CONDA_PREFIX}/lib"
if [[ -n "${BASE_LD_LIBRARY_PATH}" ]]; then
  EVAL_LD_LIBRARY_PATH+=":${BASE_LD_LIBRARY_PATH}"
fi

printf '[%s] Command:' "$(date +%H:%M:%S)"
printf ' %q' python eval_map4d_dit.py "${eval_args[@]}"
printf '\n'

env -u LD_LIBRARY_PATH \
  xvfb-run -a -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" \
  env \
    COPPELIASIM_ROOT="${COPPELIASIM_ROOT}" \
    QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}" \
    LD_LIBRARY_PATH="${EVAL_LD_LIBRARY_PATH}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}" \
    MAP4D_DIT_CKPT="${CHECKPOINT_PATH}" \
    RLBENCH2_TEST_DEMO_PATH="${TEST_DEMO_PATH}" \
    HYDRA_FULL_ERROR=1 \
    python eval_map4d_dit.py "${eval_args[@]}"

echo "[$(date +%H:%M:%S)] Evaluation finished"
echo "  output_dir: ${OUTPUT_DIR}"
