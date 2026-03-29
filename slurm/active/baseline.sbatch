#!/bin/bash
#SBATCH -J vit2d_baseline
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/baseline_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/baseline_%j.err

# Pretrained ViT-2D, single fixed z. Conservative: lr 1e-4, WD 0.01, CE class weights + shuffle only
# (NOT WeightedRandomSampler + auto weights). Light aug, early stopping on val_balanced.

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

Z_INDEX="${Z_INDEX:-144}"
echo "=== ViT2D Baseline (pretrained, low WD, shuffle+class weights, light aug, z=${Z_INDEX}) ==="
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
  --weight_decay 0.01 \
  --label_smoothing 0.02 \
  --ce_class_weights auto \
  --train_sampler shuffle \
  --best_metric balanced_acc \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 0.005 \
  --lr_scheduler cosine \
  --warmup_epochs 3 \
  --lr_min_ratio 0.05 \
  --drop_path_rate 0.05 \
  --train_aug \
  --aug_noise 0.015 \
  --out_dir runs/vit2d_baseline \
  --amp

echo "=== DONE ==="
date
