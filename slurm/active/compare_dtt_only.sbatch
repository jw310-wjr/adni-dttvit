#!/bin/bash
#SBATCH -J dtt_only
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/compare_dtt_only_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/compare_dtt_only_%j.err

# Stage 1: Compare DTT methods (DTT only, no EE)
# Pick best DTT method for DTT+EE and Teacher-Student

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== Compare DTT Only (L2, Attn, Random) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

python run_all/run_compare_dtt_only.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --out_base runs/compare_dtt_only \
  --results_dir results

echo "=== DONE ==="
date
