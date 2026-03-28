# HPC Quick Start

Run the project on HPC (e.g. SLURM cluster). Assumes you clone to `/scratch/jw310/adni-dttvit`.

---

## 1. Clone and Setup

```bash
cd /scratch/jw310
git clone git@github.com:jw310-wjr/adni-dttvit.git
cd adni-dttvit

# Create dirs
mkdir -p runs/logs results

# Activate your venv (or create one)
source /scratch/jw310/venvs/torch_fix/bin/activate
pip install -r requirements.txt
```

**If your paths differ** (e.g. project at `/home/user/adni-dttvit`), edit `PROJECT` and `VENV` at the top of each file in `slurm/active/*.sbatch`.

### Fixed training protocol (anti-overfit baseline, same for +DTT / ablation)

All `slurm/active/*.sbatch` jobs that call `train.py` (or wrappers that call it) use **the same optimization settings** so **baseline vs. proposal methods** are compared fairly:

| Setting | Value |
|--------|--------|
| Epochs | **30** |
| Learning rate | **1e-4** |
| Weight decay | **0.1** |
| Batch size | **8** |
| Workers | **4** |
| Slice | fixed **z_index=77** |
| AMP | **on** (`--amp` where applicable) |

**Why:** Shorter training + stronger regularization reduces ViT overfitting on small ADNI 2D slice data; `best.pt` is still chosen by **best validation accuracy**. Your manual `baseline_antioverfit.sbatch` matched this protocol; the repo defaults are now aligned.

---

## 2. Data Preparation (if not done)

Ensure `data/` has the expected structure. From project root (uses preprocessed/MNI-registered data):

```bash
# 1. 02: Merge metadata (raw paths, for preprocessing)
python scripts/02_build_manifest.py

# 2. 01: MNI preprocessing (N4 → SynthStrip → Affine → SyN)
sbatch slurm/active/mni_preprocess.sbatch

# 3. 02 --preprocessed: Build manifest pointing to syn/
python scripts/02_build_manifest.py --preprocessed

# 4. 03: Split into train/val/test (default: data/merged431_syn.csv)
python scripts/03_make_splits.py --out_dir manifests
```

**If data is already prepared**: Skip this step. Manifests must exist and point to valid `nifti_path` (data/syn/ for preprocessed).

---

## 3. Run Experiments (SLURM)

Submit from project root: `cd /scratch/jw310/adni-dttvit`

### Recommended order

```bash
# 1. Baseline
sbatch slurm/active/baseline.sbatch

# 2. Ablation (Baseline, Only EE, +DTT, +DTT+EE)
sbatch slurm/active/compare_ablation.sbatch

# 3. Stage 1: DTT only (L2, Attn, Random)
sbatch slurm/active/compare_dtt_only.sbatch

# 4. Stage 2: DTT+EE (l2, attn, random)
sbatch slurm/active/compare_thin_methods.sbatch

# 5. Stage 3: Teacher-Student
sbatch slurm/active/teacher_student.sbatch

# 6. Evaluate compute metrics
sbatch slurm/active/eval_compute_metrics.sbatch
```

### Optional jobs

```bash
# Slice selection (find best z)
sbatch slurm/active/slice_selection.sbatch

# Single DTT+EE run
sbatch slurm/active/dtt_ee.sbatch
THIN_METHOD=l2 sbatch slurm/active/dtt_ee.sbatch

# Teacher-Student (student only, if teacher already trained)
sbatch slurm/active/teacher_student_skip.sbatch

# Early-exit tau sweep
sbatch --export=CKPT=runs/compare_thin_methods/attn/best.pt slurm/active/eval_early_exit_tau.sbatch
```

---

## 4. Check Results

- **Logs**: `runs/logs/*.out`, `runs/logs/*.err`
- **Checkpoints**: `runs/vit2d_baseline/best.pt`, `runs/compare_ablation/*/best.pt`, etc.
- **Aggregated**: `results/*.csv`, `results/*.md`

```bash
squeue -u $USER          # Check job status
tail -f runs/logs/baseline_*.out   # Watch log
```

---

## 5. Paths to Customize

If your HPC paths differ, edit these in each `slurm/active/*.sbatch`:

| Variable | Default | Your path |
|----------|---------|-----------|
| `PROJECT` | `/scratch/jw310/adni-dttvit` | Where you cloned |
| `VENV` | `/scratch/jw310/venvs/torch_fix/bin/activate` | Your Python venv |
| `HF_HOME` | `/scratch/jw310/hf_cache` | HuggingFace cache |
| `TORCH_HOME` | `/scratch/jw310/torch_cache` | PyTorch/timm cache |

Output paths in `#SBATCH -o` and `-e` use `PROJECT/runs/logs/`, so they update if you change `PROJECT`.
