#!/bin/bash
#SBATCH -J vit2d_baseline_aug
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/baseline_aug_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/baseline_aug_%j.err

# Same balanced protocol as baseline.sbatch; alternate out_dir. Requires torchvision.

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

Z_INDEX="${Z_INDEX:-144}"
echo "=== ViT2D Baseline (balanced, z=${Z_INDEX}) -> vit2d_baseline_aug ==="
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
  --lr 2.5e-4 \
  --weight_decay 0.09 \
  --label_smoothing 0.035 \
  --ce_class_weights auto \
  --best_metric balanced_acc \
  --early_stopping_patience 15 \
  --train_aug \
  --aug_deg 10 \
  --aug_translate 0.03 \
  --aug_noise 0.025 \
  --aug_flip_prob 0.0 \
  --out_dir runs/vit2d_baseline_aug \
  --amp

echo "=== DONE ==="
date
