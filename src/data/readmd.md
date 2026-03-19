# ADNI MRI Preprocessing and 2D Slice Selection

This repository implements a fully reproducible preprocessing pipeline for
Alzheimer’s Disease (AD) MRI classification using Vision Transformers (ViT).
The preprocessing stage defines the **only fixed data input** for all subsequent
experiments (baseline ViT, Dynamic Token Thinning, Early Exit, etc.).
All downstream experiments modify *only the model architecture*, not the data.

---

## 1. Dataset Definition and Task Setup

- **Dataset**: ADNI T1-weighted MPRAGE MRI
- **Task**: Binary classification (CN vs AD)
- **Deduplication**: Subject-level; only one scan retained per subject
- **Fixed subject-level splits** (used across all experiments):
  - Train: 472 subjects
  - Validation: 59 subjects
  - Test: 59 subjects

Fixing the data splits avoids information leakage and ensures fair comparison
across model variants.

---

## 2. Skull Stripping

Each raw MRI volume is skull-stripped using **SynthStrip**:

$$
I_{\text{brain}}(x,y,z)=\text{SynthStrip}(I_{\text{raw}}(x,y,z))
$$


We do not explicitly store a binary mask. Instead, we define a deterministic
brain mask as:

$$
M(x,y,z) = \mathbf{1}\{ I_{\text{brain}}(x,y,z) > 0 \}
$$


This approximation is fully reproducible and sufficient for downstream analysis.

---

## 3. Spatial Normalization to MNI Space

All skull-stripped volumes are registered to the **MNI152 T1 1mm template**
using ANTs:

1. Affine registration
2. Non-linear SyN registration

All deformation-based analyses are performed in the common MNI space.
Affine- and SyN-registered volumes are stored separately.

---

## 4. 2D Slice Selection

We select a single axial 2D slice per volume for ViT input. Supported methods:

- **Fixed z**: Use a fixed slice index (e.g. z=77, middle for ~155 slices) — anatomically meaningful on registered volumes.
- **Middle**: Deterministic middle slice.
- **Entropy**: Slice with highest intensity entropy (heuristic).

Use `run_slice_selection.py` to find the best fixed z by training baseline for multiple z values.

---

## 5. Outputs and Manifests

For each split (train / val / test), we store:

- Skull-stripped image
- Affine-registered image
- SyN-registered image

CSV manifest files record:
- Original MRI path (or registered path)

This design guarantees:
- Full reproducibility
- No data leakage
- Fair comparison across model variants

---

## 6. Key Design Principle

> **Data is fixed. Models change.**

All experiments use the same preprocessed data.
Performance differences arise solely from model architecture and inference design
(DTT, early exit, etc.), not from data selection.
