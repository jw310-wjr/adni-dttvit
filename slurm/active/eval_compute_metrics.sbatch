#!/bin/bash
#SBATCH -J eval_metrics
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/eval_compute_metrics_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/eval_compute_metrics_%j.err

# Evaluate accuracy + inference time + speedup + GFLOPs for all configs.
# Run after: baseline, compare_ablation, compare_dtt_only, compare_thin_methods, teacher_student

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== Evaluate Compute Metrics ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

python run_all/run_eval_compute_metrics.py \
  --manifest_dir manifests \
  --manifest test.csv \
  --data_root ${PROJECT} \
  --tau 0.8 \
  --out_csv results/compute_metrics.csv

echo "=== DONE ==="
date
