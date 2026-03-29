#!/bin/bash
#SBATCH -J slice_sel
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jw310/adni-dttvit/runs/logs/slice_selection_%j.out
#SBATCH -e /scratch/jw310/adni-dttvit/runs/logs/slice_selection_%j.err

set -euo pipefail

PROJECT=/scratch/jw310/adni-dttvit
VENV=/scratch/jw310/venvs/torch_fix/bin/activate

# No train_aug here: z is chosen under the same regime as historical sweeps (wd=0.1, no aug).
echo "=== Slice Selection (find best z) ==="
hostname
date
echo

source ${VENV}
export HF_HOME=/scratch/jw310/hf_cache
export TORCH_HOME=/scratch/jw310/torch_cache
export TIMM_CACHE_DIR=/scratch/jw310/torch_cache/timm

cd ${PROJECT}
export PYTHONPATH=${PROJECT}:${PYTHONPATH:-}

python run_all/run_slice_selection.py \
  --manifest_dir manifests \
  --data_root ${PROJECT} \
  --z_list 19 48 77 115 144 173 \
  --out_base runs/slice_selection \
  --results_csv results/z_results.csv \
  --epochs 30 \
  --batch_size 8 \
  --train_extra "--amp --lr 1e-4 --weight_decay 0.1 --num_workers 4 --ce_class_weights none --label_smoothing 0 --best_metric acc --lr_scheduler none --drop_path_rate 0"

echo "=== DONE ==="
date
