# ADNI-DTT-ViT (JSD-ViT)

ADNI MRI classification with **Joint Spatial-Depth Adaptive ViT (JSD-ViT)**: Dynamic Token Thinning (DTT), Uncertainty-Guided Early Exit, and Teacher-Student distillation.

**Proposal features:**
- **Token thinning methods**: L2, Attn, Random (heuristic) + **Learnable** (ScoreHead MLP, §3.3)
- **Approximate differentiable Top-K** (§3.4): hard top-k + soft reweight (Gumbel-Softmax straight-through)
- **Anatomical regularization** (§3.3): L_anatomy = ||M_token - M_prior||². M_prior: spatial center (default) or atlas mask (e.g. hippocampus ROI)
- **Budget-aware training** (§3.8): L_sparse = Σ(N_ℓ/N_0 - r_ℓ)²
- **Feature distillation** (§3.7): L_feat = ||f_student - f_teacher||²

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

**Full pipeline (run from project root, uses preprocessed/MNI-registered data for training):**

```bash
# 0. Setup
mkdir -p runs/logs results

# 1. 02: Merge metadata with NIfTI paths (raw, for preprocessing)
python scripts/02_build_manifest.py
# Output: data/merged431.csv

# 2. 01: MNI preprocessing (N4 → SynthStrip → Affine → SyN)
# Input: data/merged431.csv
# Output: data/n4/, stripped/, affine/, syn/, jacobian/
sbatch slurm/active/mni_preprocess.sbatch

# 3. 02 --preprocessed: Build manifest pointing to syn/ (MNI-registered)
python scripts/02_build_manifest.py --preprocessed
# Output: data/merged431_syn.csv

# 4. 03: Split into train/val/test (default in_csv: data/merged431_syn.csv)
python scripts/03_make_splits.py --out_dir manifests
# Output: manifests/train.csv, val.csv, test.csv (nifti_path -> data/syn/)

# 5. Training (uses preprocessed NIfTI from syn/)
sbatch slurm/active/baseline.sbatch
```

- **Manifest columns**: `Subject`, `Group` (CN/AD for binary), `nifti_path`
- **Preprocessing**: SynthStrip → ANTs (Affine + SyN) → MNI space

See `docs/DATA_FLOW.md` for full pipeline details, and `scripts/01_MNI_preprocess.py`, `src/data/readmd.md`.

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

# 7. Interpretability: token heatmaps + anatomical alignment (§4.3)
python run_all/run_interpretability.py --ckpt runs/compare_dtt_only/learnable/best.pt --n_samples 5 --overlay_mri
# With anatomical ROI for alignment eval:
python run_all/run_interpretability.py --ckpt ... --anatomical_mask atlases/hippocampus_roi.nii.gz --eval_alignment
```

## Experiments

**Pipeline**: Baseline → Only EE → DTT only (4 methods) → pick best → DTT+EE → Teacher-Student

| Script | Description |
|--------|-------------|
| `run_compare_ablation.py` | Baseline, Only EE, +DTT, +DTT+EE |
| `run_compare_dtt_only.py` | Stage 1: DTT(L2/Attn/Random/Learnable) only, pick best method |
| `run_compare_thin_methods.py` | Stage 2: DTT(L2/Attn/Random/Learnable)+EE |
| `run_teacher_student.py` | Stage 3: Teacher → Student (DTT+EE) |
| `run_slice_selection.py` | Find best fixed z slice |
| `run_eval_early_exit_tau.py` | Evaluate early-exit at different tau |
| `run_eval_compute_metrics.py` | Accuracy, AUROC, time, speedup, Pareto plot (§4.2, §4.3) |
| `run_interpretability.py` | Token heatmaps, overlay on MRI, anatomical alignment eval (§4.3) |

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

## Classification

**Binary (CN vs AD)** — default `--label_map "CN=0,AD=1"`

```bash
python run_all/train.py --out_dir runs/binary
```

## Proposal (JSD-ViT) Options

```bash
# Learnable DTT + anatomical prior (spatial center) + uncertainty-guided EE
python run_all/train.py --thinning --thin_method learnable --early_exit \
  --use_anatomical_prior --lambda_anatomy 0.1 --lambda_sparse 0.01 \
  --tau 0.8 --tau_u 0.5

# Anatomically grounded prior (atlas mask, e.g. hippocampus ROI)
python run_all/train.py --thinning --thin_method learnable --use_anatomical_prior \
  --anatomical_prior_path atlases/hippocampus_roi.nii.gz --lambda_anatomy 0.1

# Teacher-Student with feature distillation
python run_all/run_teacher_student.py --thin_method learnable --lambda_feat 0.1

# Interpretability: token heatmaps, overlay on MRI, anatomical alignment (§4.3)
python run_all/run_interpretability.py --ckpt runs/compare_dtt_only/learnable/best.pt --n_samples 5 --overlay_mri
python run_all/run_interpretability.py --ckpt ... --anatomical_mask atlases/hippocampus_roi.nii.gz --eval_alignment
```
