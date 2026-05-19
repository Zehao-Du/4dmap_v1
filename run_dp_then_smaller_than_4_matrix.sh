#!/usr/bin/env bash
set -u

ROOT_DIR="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy"
DP_DIR="${ROOT_DIR}/baselines/diffusion_policy"
MATRIX_SCRIPT="${ROOT_DIR}/smaller_than_4_matrix.py"

if [[ "${CONDA_DEFAULT_ENV:-}" != "4dmap" && "${RUNNING_IN_4DMAP:-0}" != "1" ]]; then
  export RUNNING_IN_4DMAP=1
  exec conda run --no-capture-output -n 4dmap bash "$0" "$@"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

seed="${SEED:-1}"
demos="${DEMOS:-100}"
total_iters="${TOTAL_ITERS:-400000}"
max_episode_steps="${MAX_EPISODE_STEPS:-1000}"
num_eval_episodes="${NUM_EVAL_EPISODES:-100}"
num_eval_envs="${NUM_EVAL_ENVS:-10}"
matrix_timeout_seconds="${MATRIX_TIMEOUT_SECONDS:-}"

cd "${DP_DIR}"

set +e
python train_rgbd.py --env-id StackCube-v1 \
  --demo-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pose.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos "${demos}" --max_episode_steps "${max_episode_steps}" \
  --total_iters "${total_iters}" --obs-mode "rgb" \
  --num-eval-episodes "${num_eval_episodes}" --num-eval-envs "${num_eval_envs}" \
  --exp-name "diffusion_policy-StackCube-v1-rgb-${demos}_motionplanning_demos-${seed}" \
  --track
train_status=$?

echo "train_rgbd.py exited with status ${train_status}; starting ${MATRIX_SCRIPT}"

cd "${ROOT_DIR}"
if [[ -n "${matrix_timeout_seconds}" ]]; then
  timeout "${matrix_timeout_seconds}" python "${MATRIX_SCRIPT}" "$@"
else
  python "${MATRIX_SCRIPT}" "$@"
fi
matrix_status=$?

if [[ ${matrix_status} -ne 0 ]]; then
  exit "${matrix_status}"
fi
exit "${train_status}"
