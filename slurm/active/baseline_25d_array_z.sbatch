#!/bin/bash
#SBATCH -J vit2d_25d_z
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --array=77,115,127,144
#  One GPU job per center slice z (2.5D stack around that z). Logs: %A=jobid, %a=z task id.
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/baseline_25d_array_%A_%a.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/baseline_25d_array_%A_%a.err

# Same train recipe as baseline_25d.sbatch; z comes from array (77,115,127,144).
# Optional: SLICE_STACK_MODE=stack5 sbatch slurm/active/baseline_25d_array_z.sbatch

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

Z_INDEX="${SLURM_ARRAY_TASK_ID}"
SLICE_STACK_MODE="${SLICE_STACK_MODE:-stack3}"

echo "=== ViT2D 2.5D Baseline ARRAY z=${Z_INDEX} slice_stack_mode=${SLICE_STACK_MODE} job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

OUT_TAG="vit2d_baseline_${SLICE_STACK_MODE}_z${Z_INDEX}"

python run_all/train.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --slice_selector fixed \
  --z_index "${Z_INDEX}" \
  --slice_stack_mode "${SLICE_STACK_MODE}" \
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

echo "=== DONE z=${Z_INDEX} ==="
date
