#!/usr/bin/env python3
"""
Proposal §4.3 Interpretability: Token importance maps and anatomical alignment evaluation.

- Visualize token importance heatmaps from ScoreHead (M_token)
- Overlay heatmap on MRI slice to verify disease-relevant regions
- Compare M_token vs M_prior (spatial center or atlas mask)
- Anatomical alignment: overlap/correlation with hippocampus/ventricle ROI (if --anatomical_mask)

Usage:
  python run_all/run_interpretability.py --ckpt runs/compare_dtt_only/learnable/best.pt
  python run_all/run_interpretability.py --ckpt runs/compare_dtt_only/learnable/best.pt --n_samples 5 --overlay_mri
  python run_all/run_interpretability.py --ckpt ... --anatomical_mask atlases/hippocampus_roi.nii.gz --eval_alignment
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_nii2d import Nii2DSliceDataset
from src.models.vit2d import build_vit2d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Checkpoint path (learnable model)")
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--manifest", default="test.csv")
    p.add_argument("--data_root", default=".")
    p.add_argument("--label_map", default="CN=0,AD=1")
    p.add_argument("--n_samples", type=int, default=5, help="Number of samples to visualize")
    p.add_argument("--out_dir", default="results/interpretability")
    p.add_argument("--grid_size", type=int, default=14)
    p.add_argument("--no_pretrained", action="store_true", help="Skip ImageNet weights (use when loading from checkpoint)")
    p.add_argument("--overlay_mri", action="store_true", help="Overlay token heatmap on MRI slice (Proposal §4.3)")
    p.add_argument("--anatomical_mask", default=None, help="Path to ROI NIfTI (e.g. hippocampus) for anatomical alignment eval")
    p.add_argument("--anatomical_slice", type=int, default=None, help="Slice index for anatomical mask (default: middle)")
    p.add_argument("--eval_alignment", action="store_true", help="Compute overlap/correlation of M_token with anatomical mask")
    return p.parse_args()


def parse_label_map(s):
    return {k: int(v) for k, v in (x.split("=") for x in s.split(","))}


@torch.no_grad()
def extract_token_scores(model, x):
    """
    Run forward and capture ScoreHead outputs at each thinning block.
    Returns list of (block_idx, scores) with scores [B, N].
    """
    if not hasattr(model, "score_heads") or not model.score_heads:
        raise ValueError("Model must have learnable ScoreHeads (thin_method=learnable)")

    vit = model.vit
    x = vit.patch_embed(x)
    cls = vit.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = x + vit.pos_embed[:, : x.shape[1], :]
    x = vit.pos_drop(x)

    scores_list = []
    for i, blk in enumerate(vit.blocks):
        x = blk(x)
        if str(i) in model.score_heads:
            toks = x[:, 1:, :]
            scores = model.score_heads[str(i)](toks)
            scores_list.append((i, scores.cpu().numpy()))

    return scores_list


def plot_importance_map(scores, out_path, title="", grid_size=14):
    """Plot token importance as 2D heatmap (Proposal §4.3)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    if scores.ndim == 2:
        scores = scores[0]
    scores = scores.reshape(grid_size, grid_size)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(scores, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Importance")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_overlay_mri(mri_slice, scores, out_path, title="", grid_size=14):
    """Overlay token importance heatmap on MRI slice (Proposal §4.3: verify disease-relevant regions)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if isinstance(mri_slice, torch.Tensor):
        mri_slice = mri_slice.cpu().numpy()
    if mri_slice.ndim == 3:
        mri_slice = mri_slice[0]  # (C,H,W) -> (H,W)
    if scores.ndim == 2:
        scores = scores[0]
    scores_2d = scores.reshape(grid_size, grid_size)
    scores_norm = (scores_2d - scores_2d.min()) / (scores_2d.max() - scores_2d.min() + 1e-8)
    # Upsample heatmap to match MRI size (224x224)
    t = torch.from_numpy(scores_norm).float().unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(mri_slice.shape[0], mri_slice.shape[1]),
        mode="bilinear", align_corners=False,
    )
    heatmap = t.squeeze().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mri_slice, cmap="gray")
    im = ax.imshow(heatmap, cmap="hot", alpha=0.5, interpolation="bilinear")
    plt.colorbar(im, label="Token importance")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_prior_vs_token(m_prior, scores, out_path, grid_size=14, prior_label="M_prior"):
    """Compare M_prior vs M_token (learned). prior_label: 'spatial center' or 'anatomical (atlas)'."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    prior = m_prior.reshape(grid_size, grid_size).numpy()
    if scores.ndim == 2:
        scores = scores[0]
    token = scores.reshape(grid_size, grid_size)
    token = (token - token.min()) / (token.max() - token.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(prior, cmap="hot", interpolation="nearest")
    axes[0].set_title(f"M_prior ({prior_label})")
    axes[0].axis("off")
    axes[1].imshow(token, cmap="hot", interpolation="nearest")
    axes[1].set_title("M_token (learned)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_anatomical_mask(path, grid_size=14, slice_idx=None, axis=2):
    """Load ROI mask from NIfTI, extract slice, downsample to grid. Returns [N] float."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required for anatomical mask")
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Mask must be 3D, got {vol.shape}")
    vol = np.moveaxis(vol, axis, -1)
    z = slice_idx if slice_idx is not None else vol.shape[-1] // 2
    slc = vol[..., z]
    t = torch.from_numpy(slc).float().unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(grid_size, grid_size), mode="bilinear", align_corners=False
    )
    mask = t.squeeze().numpy().flatten()
    mask = mask / (mask.max() + 1e-8)
    return mask


def eval_anatomical_alignment(m_token, mask, top_k_frac=0.3):
    """
    Evaluate alignment of M_token with anatomical ROI.
    Returns overlap (IoU of top-k tokens with mask), correlation.
    """
    m = m_token.flatten()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    k = max(1, int(len(m) * top_k_frac))
    top_idx = np.argsort(m)[-k:]
    token_bin = np.zeros_like(m)
    token_bin[top_idx] = 1
    mask_bin = (mask > 0.5).astype(np.float32)
    inter = (token_bin * mask_bin).sum()
    union = (token_bin + mask_bin > 0).sum()
    iou = inter / (union + 1e-8)
    corr = np.corrcoef(m, mask)[0, 1] if np.std(mask) > 1e-8 else 0.0
    return float(iou), float(corr) if not np.isnan(corr) else 0.0


def main():
    args = parse_args()
    label_map = parse_label_map(args.label_map)
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {}
    run_dir = os.path.dirname(args.ckpt)
    cfg_path = os.path.join(run_dir, "results.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    model = build_vit2d(
        num_classes=num_classes,
        thinning=True,
        thin_method="learnable",
        enable_early_exit=cfg.get("early_exit", False),
        use_anatomical_prior=cfg.get("use_anatomical_prior", False),
        anatomical_prior_path=cfg.get("anatomical_prior_path"),
        anatomical_prior_slice=cfg.get("anatomical_prior_slice"),
        pretrained=not args.no_pretrained,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    model.eval()

    csv_path = os.path.join(args.manifest_dir, args.manifest)
    slice_cfg = {"slice_selector": "fixed", "data_root": args.data_root, "z_index": 77}
    ds = Nii2DSliceDataset(csv_path, label_map=label_map, **slice_cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load anatomical mask for alignment eval (Proposal §4.3)
    anat_mask = None
    if args.anatomical_mask and os.path.exists(args.anatomical_mask):
        anat_mask = load_anatomical_mask(
            args.anatomical_mask, args.grid_size,
            slice_idx=args.anatomical_slice, axis=2,
        )
        print(f"Loaded anatomical mask: {args.anatomical_mask}")

    prior_label = "anatomical (atlas)" if cfg.get("anatomical_prior_path") else "spatial center"

    inv_label = {v: k for k, v in label_map.items()}
    alignment_rows = []
    for idx, (x, y) in enumerate(loader):
        if idx >= args.n_samples:
            break
        x = x.to(device)
        label_str = inv_label.get(int(y.item()), str(y.item()))

        scores_list = extract_token_scores(model, x)

        for blk_idx, scores in scores_list:
            out_path = os.path.join(
                args.out_dir, f"sample{idx}_{label_str}_block{blk_idx}.png"
            )
            plot_importance_map(
                scores, out_path,
                title=f"Sample {idx} ({label_str}) Block {blk_idx}",
                grid_size=args.grid_size,
            )
            print(f"Saved {out_path}")

            if args.overlay_mri:
                mri_slice = x[0]  # (3,224,224)
                out_path_overlay = os.path.join(
                    args.out_dir, f"sample{idx}_{label_str}_block{blk_idx}_overlay.png"
                )
                plot_overlay_mri(
                    mri_slice, scores, out_path_overlay,
                    title=f"Sample {idx} ({label_str}) Block {blk_idx} on MRI",
                    grid_size=args.grid_size,
                )
                print(f"Saved {out_path_overlay}")

        if hasattr(model, "m_prior") and model.m_prior is not None and scores_list:
            m_prior = model.m_prior.squeeze(0).cpu()
            _, last_scores = scores_list[-1]
            out_path = os.path.join(args.out_dir, f"sample{idx}_{label_str}_prior_vs_token.png")
            plot_prior_vs_token(m_prior, last_scores, out_path, args.grid_size, prior_label=prior_label)
            print(f"Saved {out_path}")

        if args.eval_alignment and anat_mask is not None and scores_list:
            _, last_scores = scores_list[-1]
            iou, corr = eval_anatomical_alignment(last_scores, anat_mask)
            alignment_rows.append({"sample": idx, "label": label_str, "IoU": iou, "corr": corr})
            print(f"Sample {idx} ({label_str}): IoU={iou:.4f} corr={corr:.4f}")

    if alignment_rows:
        import pandas as pd
        df = pd.DataFrame(alignment_rows)
        align_path = os.path.join(args.out_dir, "anatomical_alignment.csv")
        df.to_csv(align_path, index=False)
        print(f"\nAnatomical alignment: {align_path}")
        print(f"  Mean IoU: {df['IoU'].mean():.4f}  Mean corr: {df['corr'].mean():.4f}")

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
