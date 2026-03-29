#!/bin/bash
#SBATCH -J ablation_aug
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/compare_ablation_aug_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/compare_ablation_aug_%j.err

# Same ablation as compare_ablation.sbatch but matches baseline_aug.sbatch regularization.

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== Ablation (aug + reg, same protocol as baseline_aug) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

Z_INDEX="${Z_INDEX:-115}"
python run_all/run_compare_ablation.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --out_base runs/compare_ablation_aug \
  --results_dir results \
  --epochs 50 \
  --batch_size 8 \
  --z_index "${Z_INDEX}" \
  --thin_method attn \
  --train_extra "--amp --lr 1e-4 --weight_decay 0.01 --num_workers 4 --label_smoothing 0.02 --ce_class_weights auto --train_sampler shuffle --best_metric balanced_acc --early_stopping_patience 15 --early_stopping_min_delta 0.005 --lr_scheduler cosine --warmup_epochs 3 --lr_min_ratio 0.05 --drop_path_rate 0.05 --train_aug --aug_noise 0.015 --slice_stack_mode stack3"

echo "=== DONE ==="
date
