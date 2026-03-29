#!/bin/bash
#SBATCH -J thin_compare
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/compare_thin_methods_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/compare_thin_methods_%j.err

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== Baseline + DTT+EE (l2, attn, random) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

Z_INDEX="${Z_INDEX:-144}"
python run_all/run_compare_thin_methods.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --out_base runs/compare_thin_methods \
  --results_dir results \
  --epochs 50 \
  --batch_size 8 \
  --z_index "${Z_INDEX}" \
  --methods baseline l2 attn random \
  --train_extra "--amp --lr 2.5e-4 --weight_decay 0.09 --num_workers 4 --label_smoothing 0.035 --ce_class_weights auto --best_metric balanced_acc --early_stopping_patience 15 --early_stopping_min_delta 0.005 --lr_scheduler cosine --drop_path_rate 0.1 --train_aug --aug_deg 10 --aug_translate 0.03 --aug_noise 0.025 --aug_flip_prob 0.0"

echo "=== DONE ==="
date
