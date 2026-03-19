# Anatomical Prior (M_prior)

Proposal §3.3: L_anatomy = ||M_token - M_prior||² regularizes learned token importance toward anatomical relevance.

## Prior Types

| Type | Flag | Description |
|------|------|--------------|
| **Spatial center** | `--use_anatomical_prior` (default) | 2D Gaussian centered on image. Weak proxy; approximates that hippocampal/ventricular regions often lie near center in axial MRI. **Not** atlas-based. |
| **Atlas mask** | `--anatomical_prior_path <nii>` | Load ROI from NIfTI (e.g. hippocampus, ventricle). Anatomically grounded prior. |

## Usage

```bash
# Spatial center prior (default)
python run_all/train.py --thinning --thin_method learnable --use_anatomical_prior --lambda_anatomy 0.1

# Atlas-based anatomical prior
python run_all/train.py --thinning --thin_method learnable --use_anatomical_prior \
  --anatomical_prior_path atlases/hippocampus_roi.nii.gz \
  --anatomical_prior_slice 77 \
  --lambda_anatomy 0.1
```

## Paper Wording

- **Spatial center**: Describe as "spatial center prior" or "weak anatomical proxy"; avoid "anatomical prior" without qualification.
- **Atlas mask**: Can use "anatomically grounded prior" or "atlas-based anatomical prior."
