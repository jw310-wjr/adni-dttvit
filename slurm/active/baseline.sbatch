#!/bin/bash
#SBATCH -J vit2d_baseline
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/baseline_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/baseline_%j.err

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== ViT2D Baseline (fixed z=77) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

python run_all/train.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --slice_selector fixed \
  --z_index 77 \
  --epochs 50 \
  --batch_size 8 \
  --num_workers 4 \
  --out_dir runs/vit2d_baseline \
  --amp

echo "=== DONE ==="
date
