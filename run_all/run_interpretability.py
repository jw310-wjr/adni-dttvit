#!/usr/bin/env python3
"""
Proposal §4.3 Interpretability: Visualize token importance maps from ScoreHead.
Verifies anatomical alignment with M_prior (e.g. hippocampal regions).

Usage:
  python run_all/run_interpretability.py --ckpt runs/compare_dtt_only/learnable/best.pt
  python run_all/run_interpretability.py --ckpt runs/compare_dtt_only/learnable/best.pt --n_samples 5 --out_dir results/interpretability
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


def plot_prior_vs_token(m_prior, scores, out_path, grid_size=14):
    """Compare M_prior (anatomical) vs M_token (learned)."""
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
    axes[0].set_title("M_prior (anatomical)")
    axes[0].axis("off")
    axes[1].imshow(token, cmap="hot", interpolation="nearest")
    axes[1].set_title("M_token (learned)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


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
        pretrained=not args.no_pretrained,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    model.eval()

    csv_path = os.path.join(args.manifest_dir, args.manifest)
    slice_cfg = {"slice_selector": "fixed", "data_root": args.data_root, "z_index": 77}
    ds = Nii2DSliceDataset(csv_path, label_map=label_map, **slice_cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    os.makedirs(args.out_dir, exist_ok=True)

    inv_label = {v: k for k, v in label_map.items()}
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

        if hasattr(model, "m_prior") and model.m_prior is not None and scores_list:
            m_prior = model.m_prior.squeeze(0).cpu()
            _, last_scores = scores_list[-1]
            out_path = os.path.join(args.out_dir, f"sample{idx}_{label_str}_prior_vs_token.png")
            plot_prior_vs_token(m_prior, last_scores, out_path, args.grid_size)
            print(f"Saved {out_path}")

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
