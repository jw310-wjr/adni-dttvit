# Data Flow and Preprocessing Pipeline

## Overview

This document describes the data preparation pipeline and how it connects to training.

---

## 1. Pipeline Order (Preprocessed for Training)

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `02_build_manifest.py` | `data/adni431.csv`, `data/adni_nifti/` | `data/merged431.csv` (raw paths) |
| 2 | `01_MNI_preprocess.py` | `data/merged431.csv` | `data/n4/`, `stripped/`, `affine/`, `syn/`, `jacobian/` |
| 3 | `02_build_manifest.py --preprocessed` | same + `data/syn/` | `data/merged431_syn.csv` (syn paths) |
| 4 | `03_make_splits.py` | `data/merged431_syn.csv` (default) | `manifests/train.csv`, `val.csv`, `test.csv` |

---

## 2. What Each Script Does

### 2.1 `02_build_manifest.py`

- **Input**: `data/adni431.csv` (metadata: Subject, Group) and `data/adni_nifti/` (raw NIfTI files)
- **Output**: `data/merged431.csv` (raw) or `data/merged431_syn.csv` (preprocessed)
- **Logic**: Inner merge on Subject; keeps only subjects that have exactly one NIfTI in `adni_nifti/`
- **Without --preprocessed**: nifti_path -> raw, e.g. `data/adni_nifti/941_S_1203/xxx.nii.gz`
- **With --preprocessed**: nifti_path -> `data/syn/syn_registered_{Subject}_m12.nii.gz` (run 01 first)

### 2.2 `03_make_splits.py`

- **Input**: `--in_csv` (default `data/merged431_syn.csv` for preprocessed)
- **Output**: `manifests/train.csv`, `val.csv`, `test.csv` (80/10/10 split, stratified by Group)
- **Logic**: Splits subjects; manifest rows are copied as-is (including `nifti_path`)

### 2.3 `01_MNI_preprocess.py`

- **Input**: `data/merged431.csv` (reads `nifti_path` for each subject)
- **Output** (all under `BASE_DIR/data/`):

  | Dir | Files | Description |
  |-----|-------|-------------|
  | `n4/` | `{Subject}_m12_N4.nii.gz` | N4 bias correction |
  | `stripped/` | `{Subject}_m12_brain.nii.gz` | Skull stripping (SynthStrip) |
  | `affine/` | `affine_registered_{Subject}_m12.nii.gz` | Affine registration to MNI |
  | `syn/` | `syn_registered_{Subject}_m12.nii.gz` | **SyN registration to MNI (final aligned)** |
  | `jacobian/` | `jacobian_{Subject}_m12.nii.gz` | Jacobian determinant (deformation) |

- **Note**: Subject ID in CSV is `941_S_1203`; preprocessed files use `941_S_1203_m12` as base name.

---

## 3. What Training Uses

- **Dataset**: `Nii2DSliceDataset` reads `nifti_path` from manifests (train.csv, val.csv, test.csv)
- **Default pipeline**: Manifests point to **preprocessed** NIfTI in `data/syn/` (MNI-registered)

---

## 4. Two Options for Training

### Option A: Train on Preprocessed (MNI-Registered) NIfTI (default)

- Run 02 → 01 → 02 --preprocessed → 03
- Manifests point to `data/syn/syn_registered_{Subject}_m12.nii.gz`
- Better spatial consistency across subjects

### Option B: Train on Raw NIfTI

- Run 02 (no --preprocessed) → 03 --in_csv data/merged431.csv
- Manifests point to `data/adni_nifti/`
- No preprocessing required; faster setup

---

## 5. Script `04_build_manifest_preprocessed.py`

Creates manifests that point to preprocessed images so training uses MNI-registered data.

```bash
# After 01_MNI_preprocess completes:
python scripts/04_build_manifest_preprocessed.py --in_csv data/merged431.csv --out_csv data/merged431_syn.csv

# Then split for train/val/test:
python scripts/03_make_splits.py --in_csv data/merged431_syn.csv --out_dir manifests
```

Options:
- `--stage syn` (default): Use SyN-registered MNI images
- `--stage stripped`: Use skull-stripped only
- `--stage affine`: Use affine-registered only

---

## 6. Summary: What to Use for Training

| Pipeline | nifti_path | When to use |
|----------|------------|-------------|
| 02 --preprocessed + 03 | Preprocessed `data/syn/` | Default; MNI-aligned |
| 02 + 03 --in_csv data/merged431.csv | Raw `data/adni_nifti/` | Quick start, no preprocessing |
