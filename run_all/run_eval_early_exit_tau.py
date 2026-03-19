#!/usr/bin/env python3
"""
Evaluate early-exit at different tau values.
Reports accuracy and exit-layer distribution for each tau.
Usage:
  python run_all/run_eval_early_exit_tau.py --ckpt runs/vit2d_dtt_ee_attn/best.pt
  python run_all/run_eval_early_exit_tau.py --ckpt runs/student_distill/best.pt --tau_list 0.5 0.7 0.8 0.9
"""
import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src.data.dataset_nii2d import Nii2DSliceDataset
from src.models.vit2d import build_vit2d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (DTT+EE model)")
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--data_root", default=".")
    p.add_argument("--label_map", default="CN=0,MCI=1,AD=2")
    p.add_argument("--tau_list", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_csv", default="results/early_exit_tau_results.csv")
    p.add_argument("--thin_method", default="attn")
    return p.parse_args()


def parse_label_map(s):
    return {k: int(v) for k, v in (x.split("=") for x in s.split(","))}


@torch.no_grad()
def evaluate_tau(model, loader, device, tau):
    model.eval()
    correct = 0
    total = 0
    exit_counts = {1: 0, 2: 0, 3: 0}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, tau=tau)
        if isinstance(out, tuple) and len(out) == 2:
            logits, exit_layer = out
            for k in (1, 2, 3):
                exit_counts[k] += (exit_layer == k).sum().item()
        else:
            logits = out
            exit_counts[3] += x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    acc = correct / max(total, 1)
    avg_exit = (1 * exit_counts[1] + 2 * exit_counts[2] + 3 * exit_counts[3]) / max(total, 1)
    return acc, exit_counts, avg_exit


def main():
    args = parse_args()
    label_map = parse_label_map(args.label_map)
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_csv = os.path.join(args.manifest_dir, "val.csv")
    test_csv = os.path.join(args.manifest_dir, "test.csv")
    slice_cfg = {"slice_selector": "fixed", "data_root": args.data_root, "z_index": 77}
    val_ds = Nii2DSliceDataset(val_csv, label_map=label_map, **slice_cfg)
    test_ds = Nii2DSliceDataset(test_csv, label_map=label_map, **slice_cfg)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_vit2d(
        num_classes=num_classes,
        thinning=True,
        thin_method=args.thin_method,
        enable_early_exit=True,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)

    rows = []
    for tau in args.tau_list:
        val_acc, val_exit, val_avg = evaluate_tau(model, val_loader, device, tau)
        test_acc, test_exit, test_avg = evaluate_tau(model, test_loader, device, tau)
        rows.append({
            "tau": tau,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "val_exit1": val_exit[1], "val_exit2": val_exit[2], "val_exit3": val_exit[3],
            "test_exit1": test_exit[1], "test_exit2": test_exit[2], "test_exit3": test_exit[3],
            "val_avg_exit": val_avg,
            "test_avg_exit": test_avg,
        })
        print(f"tau={tau:.2f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} val_exit={val_exit} val_avg_exit={val_avg:.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
