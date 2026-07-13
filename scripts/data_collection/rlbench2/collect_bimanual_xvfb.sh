#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash collect_bimanual_xvfb.sh
#   TASKS="bimanual_push_box another_task" EPISODES_PER_TASK=10 bash collect_bimanual_xvfb.sh
#
# Extra arguments are forwarded to dataset_generator_bimanual.py.

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao}
CONDA_ENV_ROOT=${CONDA_ENV_ROOT:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/miniconda3/envs/4dmap}

export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-${PROJECT_ROOT}/codes/CoppeliaSim}
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${CONDA_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# Keep this pointed at COPPELIASIM_ROOT, not COPPELIASIM_ROOT/platforms:
# CoppeliaSim also needs sibling Qt plugin dirs such as xcbglintegrations/.
export QT_QPA_PLATFORM_PLUGIN_PATH=${QT_QPA_PLATFORM_PLUGIN_PATH:-${COPPELIASIM_ROOT}}

RLBENCH_TOOLS=${RLBENCH_TOOLS:-${PROJECT_ROOT}/codes/rlbench/tools}
SAVE_PATH=${SAVE_PATH:-${PROJECT_ROOT}/4dmap/dataset/rlbench2}
# Drawer Laptop Dustpan
TASKS=${TASKS:-"
bimanual_push_box
bimanual_lift_ball
bimanual_put_item_in_drawer
bimanual_pick_laptop
bimanual_sweep_to_dustpan
bimanual_lift_tray
bimanual_handover_item_easy
bimanual_dual_push_buttons
bimanual_pick_plate
bimanual_put_bottle_in_fridge
bimanual_set_the_table
bimanual_straighten_rope
bimanual_take_tray_out_of_oven
"}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-100}
IMAGE_SIZE=${IMAGE_SIZE:-256x256}
export RLBENCH_NUM_WORKERS=${RLBENCH_NUM_WORKERS:-1}

task_args=()
for task in ${TASKS}; do
    task_args+=(--tasks "${task}")
done

cd "${RLBENCH_TOOLS}"

xvfb-run -a -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" \
    python dataset_generator_bimanual.py \
    --save_path "${SAVE_PATH}" \
    "${task_args[@]}" \
    --episodes_per_task "${EPISODES_PER_TASK}" \
    --headless \
    --image-size "${IMAGE_SIZE}" \
    "$@"
