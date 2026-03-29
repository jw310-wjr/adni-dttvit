#!/bin/bash
#SBATCH -J vit2d_baseline_aug
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/baseline_aug_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/baseline_aug_%j.err

# Baseline + train-time 2D augment + label smoothing + slightly stronger weight decay.
# Requires: pip install torchvision (see requirements.txt).

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

Z_INDEX="${Z_INDEX:-144}"
echo "=== ViT2D Baseline (augment + reg, z=${Z_INDEX}) ==="
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
  --z_index "${Z_INDEX}" \
  --epochs 50 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 0.12 \
  --label_smoothing 0.05 \
  --train_aug \
  --aug_deg 12 \
  --aug_translate 0.04 \
  --aug_noise 0.03 \
  --aug_flip_prob 0.0 \
  --out_dir runs/vit2d_baseline_aug \
  --amp

echo "=== DONE ==="
date
