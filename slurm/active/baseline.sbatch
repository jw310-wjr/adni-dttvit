#!/bin/bash
#SBATCH -J vit2d_baseline
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/baseline_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/baseline_%j.err

# Protocol: mild aug + light wd, no label smoothing (prior stack caused ~chance CE/acc — underfit).
# If val >> train again, re-add --label_smoothing 0.03–0.05 and/or raise --weight_decay.

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

Z_INDEX="${Z_INDEX:-144}"
echo "=== ViT2D Baseline (aug + light wd, z=${Z_INDEX}) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

# Training protocol: see HPC_QUICKSTART.md § "Fixed training protocol (anti-overfit baseline)"
# After slice_selection: Z_INDEX=<best from results/z_results.csv> sbatch slurm/active/baseline.sbatch
python run_all/train.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --slice_selector fixed \
  --z_index "${Z_INDEX}" \
  --epochs 50 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 3e-4 \
  --weight_decay 0.08 \
  --label_smoothing 0 \
  --train_aug \
  --aug_deg 8 \
  --aug_translate 0.025 \
  --aug_noise 0.02 \
  --aug_flip_prob 0.0 \
  --out_dir runs/vit2d_baseline \
  --amp

echo "=== DONE ==="
date
