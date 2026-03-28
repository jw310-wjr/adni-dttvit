#!/bin/bash
#SBATCH -J ablation
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/compare_ablation_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/compare_ablation_%j.err

# Ablation: Baseline, Only EE, +DTT, +DTT+EE

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== Ablation (Baseline / +DTT / +DTT+EE) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

# Same train protocol as baseline (HPC_QUICKSTART.md).
# Optional: after slice_selection + results/z_results.csv, set e.g. Z_INDEX=48 sbatch ...
Z_INDEX="${Z_INDEX:-77}"
python run_all/run_compare_ablation.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --out_base runs/compare_ablation \
  --results_dir results \
  --epochs 30 \
  --batch_size 8 \
  --z_index "${Z_INDEX}" \
  --thin_method attn \
  --train_extra "--amp --lr 1e-4 --weight_decay 0.1 --num_workers 4"

echo "=== DONE ==="
date
