#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CONFIG_NAME="${CONFIG_NAME:-map4d_dit}"

torchrun --nproc_per_node="$NPROC_PER_NODE" \
  map4d/backbone/train_map4d_dit.py \
  --config-name "$CONFIG_NAME" \
  "$@"
