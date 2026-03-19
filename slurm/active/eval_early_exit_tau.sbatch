#!/bin/bash
#SBATCH -J eval_tau
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/eval_tau_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/eval_tau_%j.err

# Evaluate early-exit at different tau. Set CKPT before submitting.
# sbatch --export=CKPT=runs/vit2d_dtt_ee_attn/best.pt slurm/active/eval_early_exit_tau.sbatch

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate
CKPT=${CKPT:-runs/vit2d_dtt_ee_attn/best.pt}

echo "=== Early Exit Tau Evaluation ==="
hostname
date
echo "CKPT=${CKPT}"
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

python run_all/run_eval_early_exit_tau.py \
  --ckpt ${CKPT} \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --tau_list 0.5 0.6 0.7 0.8 0.9 \
  --out_csv results/early_exit_tau_results.csv

echo "=== DONE ==="
date
