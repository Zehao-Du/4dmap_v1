#!/usr/bin/env bash
# Train StackCube Map4DDiT with the read-heavy HDF5 dataset staged in shared memory.
#
# Usage:
#   bash scripts/map4d_experiments/train_stackcube_map4d_dit_shm.sh <num_demos> [hydra_overrides...]
#
# Optional environment variables:
#   SHM_ROOT=/dev/shm/4dmap_stackcube_dit   Shared-memory staging directory.
#   FORCE_SHM_COPY=1                        Re-copy files even if staged copies exist.
#   CLEAN_SHM_ON_EXIT=1                     Remove staged files when training exits.
#   MANIFEST_PATH=/path/to/context.env      Source manifest to stage; defaults to the normal StackCube manifest.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <num_demos> [hydra_overrides...]" >&2
  exit 1
fi

NUM_DEMOS="$1"
shift
EXTRA_OVERRIDES=("$@")
if [[ ! "${NUM_DEMOS}" =~ ^[0-9]+$ || "${NUM_DEMOS}" -le 0 ]]; then
  echo "Invalid num_demos=${NUM_DEMOS}. Use a positive integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset/maniskill}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/ManiSkill/StackCube-v1/motionplanning}"
FUTURE_HORIZON="${FUTURE_HORIZON:-4}"
SOURCE_MANIFEST="${MANIFEST_PATH:-${DATASET_DIR}/StackCube.rgb+depth.pd_ee_delta_pos.physx_cpu.filtered.map4d_dit_h${FUTURE_HORIZON}.context.env}"
SHM_ROOT="${SHM_ROOT:-/dev/shm/4dmap_stackcube_dit}"
FORCE_SHM_COPY="${FORCE_SHM_COPY:-0}"
CLEAN_SHM_ON_EXIT="${CLEAN_SHM_ON_EXIT:-0}"

if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "Source manifest not found: ${SOURCE_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${SHM_ROOT}"

# shellcheck disable=SC1090
source "${SOURCE_MANIFEST}"

required_vars=(
  TASK_OVERRIDE
  MAP4D_DEMO_PATH
  MAP4D_KEYFRAME_SIDECAR_PATH
  MAP4D_NUM_TRAJ
  SEMANTIC_FEATURE_DIM
  MAP_FEATURE_DIM
  NUM_MAP_NODES
)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Manifest missing required variable: ${var_name}" >&2
    exit 1
  fi
done

if (( MAP4D_NUM_TRAJ < NUM_DEMOS )); then
  echo "Manifest MAP4D_NUM_TRAJ=${MAP4D_NUM_TRAJ} is smaller than requested num_demos=${NUM_DEMOS}." >&2
  exit 1
fi
if [[ ! -f "${MAP4D_DEMO_PATH}" ]]; then
  echo "Demo file not found: ${MAP4D_DEMO_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MAP4D_KEYFRAME_SIDECAR_PATH}" ]]; then
  echo "Sidecar file not found: ${MAP4D_KEYFRAME_SIDECAR_PATH}" >&2
  exit 1
fi

DEMO_BASENAME="$(basename "${MAP4D_DEMO_PATH}")"
SIDECAR_BASENAME="$(basename "${MAP4D_KEYFRAME_SIDECAR_PATH}")"
MANIFEST_BASENAME="$(basename "${SOURCE_MANIFEST}")"
SHM_DEMO_PATH="${SHM_ROOT}/${DEMO_BASENAME}"
SHM_SIDECAR_PATH="${SHM_ROOT}/${SIDECAR_BASENAME}"
SHM_MANIFEST_PATH="${SHM_ROOT}/${MANIFEST_BASENAME%.env}.shm.env"

copy_if_needed() {
  local src="$1"
  local dst="$2"
  if [[ "${FORCE_SHM_COPY}" == "1" || "${FORCE_SHM_COPY}" == "true" || ! -f "${dst}" ]]; then
    echo "Staging $(basename "${src}") -> ${dst}"
    cp -f "${src}" "${dst}.tmp"
    mv -f "${dst}.tmp" "${dst}"
  else
    echo "Using existing staged file: ${dst}"
  fi
}

cleanup() {
  if [[ "${CLEAN_SHM_ON_EXIT}" == "1" || "${CLEAN_SHM_ON_EXIT}" == "true" ]]; then
    rm -f "${SHM_DEMO_PATH}" "${SHM_SIDECAR_PATH}" "${SHM_MANIFEST_PATH}"
  fi
}
trap cleanup EXIT

echo "Shared-memory staging directory: ${SHM_ROOT}"
copy_if_needed "${MAP4D_DEMO_PATH}" "${SHM_DEMO_PATH}"
copy_if_needed "${MAP4D_KEYFRAME_SIDECAR_PATH}" "${SHM_SIDECAR_PATH}"

cp -f "${SOURCE_MANIFEST}" "${SHM_MANIFEST_PATH}.tmp"
python - "${SHM_MANIFEST_PATH}.tmp" "${SHM_MANIFEST_PATH}" "${SHM_DEMO_PATH}" "${SHM_SIDECAR_PATH}" <<'PY'
from pathlib import Path
import sys

src, dst, demo_path, sidecar_path = map(Path, sys.argv[1:5])
lines = src.read_text(encoding="utf-8").splitlines()
out = []
for line in lines:
    if line.startswith("MAP4D_DEMO_PATH="):
        out.append(f"MAP4D_DEMO_PATH={str(demo_path)!r}")
    elif line.startswith("MAP4D_KEYFRAME_SIDECAR_PATH="):
        out.append(f"MAP4D_KEYFRAME_SIDECAR_PATH={str(sidecar_path)!r}")
    else:
        out.append(line)
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
src.unlink()
PY

echo "Staged manifest: ${SHM_MANIFEST_PATH}"
echo "Training will read demo from: ${SHM_DEMO_PATH}"
echo "Training will read sidecar from: ${SHM_SIDECAR_PATH}"

export ROOT_DIR DATA_ROOT DATASET_DIR
export MANIFEST_PATH="${SHM_MANIFEST_PATH}"

exec bash "${SCRIPT_DIR}/train_stackcube_map4d_dit.sh" "${NUM_DEMOS}" "${EXTRA_OVERRIDES[@]}"
