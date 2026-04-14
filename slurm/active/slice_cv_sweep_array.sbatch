#!/bin/bash
# Slice CV sweep: one GPU job per z index, each runs 5-fold cross-validation.
#
# QUICK START
# -----------
# Coarse sweep (z=10,20,...,170 step 10, 17 jobs):
#   sbatch --array=10-170:10 slurm/active/slice_cv_sweep_array.sbatch
#
# Dense sweep (z=10,15,...,170 step 5, 33 jobs):
#   sbatch --array=10-170:5 slurm/active/slice_cv_sweep_array.sbatch
#
# Specific z list (e.g. refine top candidates):
#   sbatch --array=77,95,105,115,127,144 slurm/active/slice_cv_sweep_array.sbatch
#
# After all tasks finish, aggregate results:
#   python run_all/run_slice_cv_sweep.py --aggregate_only \
#          --out_base runs/slice_cv --results_csv results/slice_cv_results.csv --topk 5
#
# Override defaults via env vars:
#   EPOCHS=30 STACK=stack5 sbatch --array=10-170:10 slurm/active/slice_cv_sweep_array.sbatch

#SBATCH -J slice_cv
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
# --array is set on the sbatch command line (see examples above)
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/slice_cv_%A_%a.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/slice_cv_%A_%a.err

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

Z_INDEX="${SLURM_ARRAY_TASK_ID}"
KFOLD="${KFOLD:-5}"
EPOCHS="${EPOCHS:-20}"          # use fewer epochs for initial sweep; raise to 50 for final
STACK="${STACK:-stack3}"        # single | stack3 | stack5

echo "=== slice_cv_sweep  z=${Z_INDEX}  kfold=${KFOLD}  epochs=${EPOCHS}  stack=${STACK} ==="
echo "=== job=${SLURM_ARRAY_JOB_ID}  task=${SLURM_ARRAY_TASK_ID} ==="
hostname
date

source "${VENV}"
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

mkdir -p runs/logs

# Each z writes to runs/slice_cv/z_{z}/  (fold_1/ … fold_K/ + cv_results.json)
python run_all/run_slice_cv_sweep.py \
  --single_z      "${Z_INDEX}" \
  --kfold         "${KFOLD}" \
  --epochs        "${EPOCHS}" \
  --slice_stack_mode "${STACK}" \
  --manifest_dir  manifests \
  --data_root     "${PROJECT}" \
  --out_base      runs/slice_cv \
  --batch_size    8 \
  --train_extra   "--num_workers 4 --lr 1e-4 --weight_decay 0.01 --label_smoothing 0.02 \
                   --ce_class_weights auto --train_sampler shuffle \
                   --best_metric balanced_acc \
                   --early_stopping_patience 10 --early_stopping_min_delta 0.005 \
                   --lr_scheduler cosine --warmup_epochs 3 --lr_min_ratio 0.05 \
                   --drop_path_rate 0.05 --train_aug --aug_noise 0.015 --amp"

echo "=== DONE z=${Z_INDEX} ==="
date
