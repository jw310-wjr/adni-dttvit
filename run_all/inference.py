#!/usr/bin/env python3
"""
Inference on a manifest CSV. Outputs predictions to CSV.
Usage:
  python run_all/inference.py --ckpt runs/vit2d_baseline/best.pt --manifest_dir manifests --output predictions.csv
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset_nii2d import Nii2DSliceDataset
from src.models.vit2d import build_vit2d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint")
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--manifest", default="test.csv", help="Manifest file (e.g. test.csv)")
    p.add_argument("--data_root", default=".")
    p.add_argument("--label_map", default="CN=0,MCI=1,AD=2")
    p.add_argument("--slice_selector", default="fixed")
    p.add_argument("--z_index", type=int, default=77)
    p.add_argument("--thinning", action="store_true")
    p.add_argument("--thin_method", default="attn")
    p.add_argument("--early_exit", action="store_true")
    p.add_argument("--tau", type=float, default=None, help="Early-exit confidence threshold")
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def parse_label_map(s):
    return {k: int(v) for k, v in (x.split("=") for x in s.split(","))}


def main():
    args = parse_args()
    label_map = parse_label_map(args.label_map)
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(args.manifest_dir, args.manifest)
    slice_cfg = {"slice_selector": args.slice_selector, "data_root": args.data_root}
    if args.slice_selector == "fixed":
        slice_cfg["z_index"] = args.z_index
    ds = Nii2DSliceDataset(csv_path, label_map=label_map, **slice_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_vit2d(
        num_classes=num_classes,
        thinning=args.thinning,
        thin_method=args.thin_method,
        enable_early_exit=args.early_exit,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    model.eval()

    inv_label_map = {v: k for k, v in label_map.items()}
    preds, labels, subjects = [], [], []
    for x, y in loader:
        x = x.to(device)
        if args.tau is not None and args.early_exit:
            out, _ = model(x, tau=args.tau)
        else:
            out = model(x)
        pred = out.argmax(dim=1)
        preds.extend(pred.cpu().tolist())
        labels.extend(y.tolist())

    df = pd.read_csv(csv_path)
    subj_col = "Subject" if "Subject" in df.columns else "subject"
    if subj_col not in df.columns:
        subj_col = df.columns[0]
    df["pred"] = [inv_label_map.get(p, p) for p in preds]
    df["pred_idx"] = preds
    df["true_idx"] = labels
    df["correct"] = [p == t for p, t in zip(preds, labels)]
    df.to_csv(args.output, index=False)
    acc = sum(df["correct"]) / len(df)
    print(f"Accuracy: {acc:.4f} | Saved: {args.output}")


if __name__ == "__main__":
    main()
