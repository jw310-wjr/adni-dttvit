#!/bin/bash
#SBATCH -J slice_sel_fine
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/slice_selection_fine_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/slice_selection_fine_%j.err

# Fine search between coarse grid winners (e.g. z=115..144 after first sweep).
# No train_aug: keep consistent with coarse slice_selection (z chosen without aug).
# Edit --z_list if you narrow further.

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== Slice selection (fine band 115-144, step 3) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

python run_all/run_slice_selection.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --z_list 115 118 121 124 127 130 133 136 139 142 144 \
  --out_base runs/slice_selection_fine \
  --results_csv results/z_results_fine.csv \
  --epochs 30 \
  --batch_size 8 \
  --train_extra "--amp --lr 1e-4 --weight_decay 0.1 --num_workers 4 --ce_class_weights none --label_smoothing 0 --best_metric acc"

echo "=== DONE ==="
date
