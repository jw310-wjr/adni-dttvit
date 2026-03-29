#!/bin/bash
#SBATCH -J vit2d_25d5_z
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --array=77,115,127,144
#  Five-slice 2.5D [z-2..z+2] per center z; ViT in_chans=5.
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/baseline_25d5_array_%A_%a.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/baseline_25d5_array_%A_%a.err

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

Z_INDEX="${SLURM_ARRAY_TASK_ID}"

echo "=== ViT2D 2.5D stack5 ARRAY z=${Z_INDEX} (5 slices) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

OUT_TAG="vit2d_baseline_stack5_z${Z_INDEX}"

python run_all/train.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --slice_selector fixed \
  --z_index "${Z_INDEX}" \
  --slice_stack_mode stack5 \
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
  --out_dir "runs/${OUT_TAG}" \
  --amp

echo "=== DONE stack5 z=${Z_INDEX} ==="
date
