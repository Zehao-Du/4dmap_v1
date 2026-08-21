#!/usr/bin/env bash
# Train Map4D DiT with the validated PPI-compatible RLBench2 dataset layout.

set -eo pipefail

POLICY_DIR="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy"
CONDA_ROOT="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/miniconda3"
COPPELIASIM_ROOT="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/codes/CoppeliaSim"
MANIFEST="${POLICY_DIR}/dataset/rlbench2/map4d_dit/bimanual_push_box/rlbench2_push_box_100eps_rgb_pcd_rps6144_h4.env"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate 4dmap
set -u
cd "${POLICY_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Dataset manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]]; then
  echo "CoppeliaSim library not found: ${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" >&2
  exit 1
fi

VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
NPROC_PER_NODE="${NPROC_PER_NODE:-${VISIBLE_GPUS}}"
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ || "${NPROC_PER_NODE}" -gt "${VISIBLE_GPUS}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}, but only ${VISIBLE_GPUS} GPU(s) are visible." >&2
  exit 1
fi

export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export NPROC_PER_NODE
export MASTER_PORT="${MASTER_PORT:-29501}"
set -a
source "${MANIFEST}"
set +a

echo "visible_gpus=${VISIBLE_GPUS} nproc_per_node=${NPROC_PER_NODE}"
echo "run=ppi_layout_fixed"

exec torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  map4d/backbone/train_map4d_dit.py \
  --config-name map4d_dit "${TASK_OVERRIDE}" \
  addition_info=ppi_layout_fixed \
  policy.model_cfg.semantic_feature_dim="${SEMANTIC_FEATURE_DIM}" \
  policy.model_cfg.map_feature_dim="${MAP_FEATURE_DIM}" \
  policy.model_cfg.num_map_nodes="${NUM_MAP_NODES}" \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32
