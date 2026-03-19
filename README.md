# ADNI-DTT-ViT

ADNI MRI classification with Vision Transformers, Dynamic Token Thinning (DTT), Early Exit, and Teacher-Student distillation.

## Setup

```bash
pip install -r requirements.txt
mkdir -p runs/logs results
```

## Data Preparation

Data root `BASE = /scratch/jw310/adni-dttvit/data`. Preprocessing output structure:

```
data/
├── adni_nifti/     # Raw NIfTI
├── adni431.csv     # Metadata (Subject, Group)
├── merged431.csv   # Manifest from 02 (merged)
├── n4/             # N4 bias correction
├── stripped/       # Skull-stripped
├── affine/         # Affine registration
├── syn/            # SyN registration
├── jacobian/       # Jacobian maps
└── DCM/            # Original DICOM
```

**Full pipeline (run from project root):**

```bash
# 0. Setup
mkdir -p runs/logs results

# 1. 02: Merge metadata with NIfTI paths
# Input: data/adni431.csv, data/adni_nifti/
# Output: data/merged431.csv
python scripts/02_build_manifest.py

# 2. 03: Split into train/val/test
# Input: data/merged431.csv
# Output: manifests/train.csv, val.csv, test.csv
python scripts/03_make_splits.py --in_csv data/merged431.csv --out_dir manifests

# 3. 01: MNI preprocessing (N4 → SynthStrip → Affine → SyN → Jacobian)
# Input: data/merged431.csv
# Output: data/n4/, stripped/, affine/, syn/, jacobian/
sbatch slurm/active/mni_preprocess.sbatch
# Or: python scripts/01_MNI_preprocess.py

# 4. Training (uses nifti_path from manifests, default: data/adni_nifti)
sbatch slurm/active/baseline.sbatch
# ... see Quick Start
```

- **Manifest columns**: `Subject`, `Group` (CN/MCI/AD), `nifti_path`
- **Preprocessing**: SynthStrip → ANTs (Affine + SyN) → MNI space

See `scripts/01_MNI_preprocess.py` and `src/data/readmd.md` for details.

## Quick Start

**Before running**: Complete data preparation (02 → 03 → 01) and ensure `runs/logs`, `results` exist.

```bash
# 1. Slice selection (find best z)
python run_all/run_slice_selection.py --epochs 30

# 2. Ablation: Baseline, Only EE, +DTT, +DTT+EE
python run_all/run_compare_ablation.py --epochs 30

# 3. Stage 1: DTT only (L2, Attn, Random), pick best
python run_all/run_compare_dtt_only.py --epochs 30

# 4. Stage 2: DTT+EE (l2, attn, random)
python run_all/run_compare_thin_methods.py --epochs 30

# 5. Stage 3: Teacher-Student
python run_all/run_teacher_student.py --epochs 30

# 6. Evaluate compute metrics
python run_all/run_eval_compute_metrics.py
```

## Experiments

**Pipeline**: Baseline → Only EE → DTT only (3 methods) → pick best → DTT+EE → Teacher-Student

| Script | Description |
|--------|-------------|
| `run_compare_ablation.py` | Baseline, Only EE, +DTT, +DTT+EE |
| `run_compare_dtt_only.py` | Stage 1: DTT(L2/Attn/Random) only, pick best method |
| `run_compare_thin_methods.py` | Stage 2: DTT(L2/Attn/Random)+EE |
| `run_teacher_student.py` | Stage 3: Teacher → Student (DTT+EE) |
| `run_slice_selection.py` | Find best fixed z slice |
| `run_eval_early_exit_tau.py` | Evaluate early-exit at different tau |
| `run_eval_compute_metrics.py` | Accuracy + time + speedup (all configs) |

## Outputs

- **runs/** — checkpoints, logs
- **results/** — aggregated CSV, figures, LaTeX tables

## HPC (SLURM)

```bash
sbatch slurm/active/baseline.sbatch
sbatch slurm/active/slice_selection.sbatch
sbatch slurm/active/compare_thin_methods.sbatch
sbatch slurm/active/teacher_student.sbatch
```

See `slurm/README.md` for details.

## Inference

```bash
python run_all/inference.py --ckpt runs/vit2d_baseline/best.pt \
  --manifest_dir manifests --data_root . --output predictions.csv
```

## Binary Classification (CN vs AD)

```bash
python run_all/train.py --label_map "CN=0,AD=1" --out_dir runs/binary
```
