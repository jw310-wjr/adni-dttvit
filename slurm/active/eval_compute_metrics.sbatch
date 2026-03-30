#!/bin/bash
#SBATCH -J eval_metrics
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/eval_compute_metrics_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/eval_compute_metrics_%j.err

# Inference **compute** table: test acc, AUROC, ms/sample, speedup (vs first model in list),
# avg_exit (early-exit), params, peak GPU MB, GFLOPs -> results/compute_metrics.csv + .md
# Training does NOT write these; run this after you have best.pt under each run dir.
# Default dirs in run_eval_compute_metrics.py are ordered for stack3 z115 mainline first.

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

Z_INDEX="${Z_INDEX:-115}"
python run_all/run_eval_compute_metrics.py \
  --manifest_dir manifests \
  --manifest test.csv \
  --data_root ${PROJECT} \
  --z_index "${Z_INDEX}" \
  --tau 0.8 \
  --out_csv results/compute_metrics.csv

echo "=== DONE ==="
date
