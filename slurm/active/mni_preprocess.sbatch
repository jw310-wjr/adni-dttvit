#!/bin/bash
#SBATCH --job-name=mni_pre_431_cpu
#SBATCH --output=/scratch/jw310/adni-dttvit/runs/active/mni_pre_431_cpu_%j.out
#SBATCH --error=/scratch/jw310/adni-dttvit/runs/active/mni_pre_431_cpu_%j.err
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

echo "===== MNI Preprocessing (CPU) Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

cd /scratch/jw310/adni-dttvit
source /scratch/jw310/venvs/torch_fix/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}

export TORCH_HOME=/scratch/jw310/torch_cache
export HF_HOME=/scratch/jw310/hf_cache

python scripts/01_MNI_preprocess.py

echo "===== Job Finished ====="
echo "End time: $(date)"
