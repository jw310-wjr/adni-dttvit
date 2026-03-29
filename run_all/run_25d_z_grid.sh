#!/usr/bin/env bash
# Train a grid over candidate axial centers × 2.5D modes (same recipe; extra args appended).
# Example:
#   bash run_all/run_25d_z_grid.sh --manifest_dir manifests --train_aug
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
Z_LIST=(77 115 127 144)
MODES=(single stack3 stack5)
for z in "${Z_LIST[@]}"; do
  for mode in "${MODES[@]}"; do
    out="${ROOT}/runs/25d/z${z}_${mode}"
    python "${ROOT}/run_all/train.py" \
      --out_dir "$out" \
      --z_index "$z" \
      --slice_stack_mode "$mode" \
      "$@"
  done
done
