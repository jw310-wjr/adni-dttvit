#!/bin/bash
#SBATCH -J teacher_stu
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/teacher_student_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/teacher_student_%j.err

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

echo "=== Teacher-Student (Knowledge Distillation) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

python run_all/run_teacher_student.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --epochs 30 \
  --batch_size 8 \
  --teacher_out runs/teacher \
  --student_out runs/student_distill \
  --distill_temp 4.0 \
  --distill_alpha 0.5 \
  --results_dir results \
  --amp \
  --train_extra "--lr 1e-4 --weight_decay 0.1 --num_workers 4"

echo "=== DONE ==="
date
