#!/usr/bin/env python3
"""
Sweep all candidate axial slice indices with K-fold cross-validation and rank by CV metric.

Each z is trained with --kfold K (default 5). Results are aggregated into a CSV and
ranked; top-k z indices are printed.

Sequential usage (slow, for small z_list or debugging):
  python run_all/run_slice_cv_sweep.py --z_list 77 115 127 144 --epochs 20

Dense sweep (many z, short epochs for ranking):
  python run_all/run_slice_cv_sweep.py --z_step 5 --epochs 15 --topk 5

Single-z mode (called by SLURM array, outputs per-z cv_results.json):
  python run_all/run_slice_cv_sweep.py --single_z 115 --kfold 5 --epochs 50

After all SLURM array tasks finish, aggregate with:
  python run_all/run_slice_cv_sweep.py --aggregate_only --out_base runs/slice_cv
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---- z candidates ----
    p.add_argument(
        "--z_list", type=int, nargs="+", default=None,
        help="Explicit list of z indices. If omitted, generate from --z_min/z_max/z_step.",
    )
    p.add_argument("--z_min",  type=int, default=10,
                   help="Start of z range (inclusive) when --z_list is not set.")
    p.add_argument("--z_max",  type=int, default=175,
                   help="End of z range (inclusive) when --z_list is not set.")
    p.add_argument("--z_step", type=int, default=10,
                   help="Step size for auto z grid.")

    # ---- single-z mode (for SLURM array) ----
    p.add_argument(
        "--single_z", type=int, default=None,
        help="Run CV for exactly this one z and exit. Used by SLURM --array.",
    )

    # ---- aggregation-only mode ----
    p.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip training; just collect existing cv_results.json files under --out_base and rank.",
    )

    # ---- CV / training settings ----
    p.add_argument("--kfold",      type=int, default=5)
    p.add_argument("--epochs",     type=int, default=20,
                   help="Epochs per fold. Use fewer epochs (e.g. 15-20) for sweeping, full epochs for final.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--data_root",    default=".")
    p.add_argument("--slice_stack_mode", default="stack3",
                   choices=["single", "stack3", "stack5"])

    # extra args forwarded verbatim to train.py (e.g. "--amp --train_aug")
    p.add_argument("--train_extra", default="",
                   help="Extra flags forwarded to train.py, e.g. '--amp --train_aug'.")

    # ---- output ----
    p.add_argument("--out_base",    default="runs/slice_cv",
                   help="Base dir; each z gets {out_base}/z_{z}/.")
    p.add_argument("--results_csv", default="results/slice_cv_results.csv",
                   help="Path for aggregated ranking CSV.")
    p.add_argument("--topk", type=int, default=5,
                   help="How many best z indices to highlight.")

    # ---- ranking metric ----
    p.add_argument(
        "--rank_metric",
        default="best_val_balanced_mean",
        choices=[
            "best_val_acc_mean", "best_val_balanced_mean", "best_val_hmean_mean",
            "test_acc_mean",     "test_balanced_mean",     "test_hmean_mean",
        ],
        help="CV metric used to rank z indices.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_z_list(args):
    if args.z_list is not None:
        return sorted(set(args.z_list))
    return list(range(args.z_min, args.z_max + 1, args.z_step))


def cv_results_path(out_base, z):
    return os.path.join(out_base, f"z_{z}", "cv_results.json")


def run_one_z(z, args):
    """Invoke train.py with --kfold for a single z. Blocks until done."""
    out_dir = os.path.join(PROJECT_ROOT, args.out_base, f"z_{z}")
    train_script = os.path.join(PROJECT_ROOT, "run_all", "train.py")

    cmd = [
        sys.executable, train_script,
        "--manifest_dir",    args.manifest_dir,
        "--data_root",       args.data_root,
        "--slice_selector",  "fixed",
        "--z_index",         str(z),
        "--slice_stack_mode", args.slice_stack_mode,
        "--kfold",           str(args.kfold),
        "--epochs",          str(args.epochs),
        "--batch_size",      str(args.batch_size),
        "--out_dir",         out_dir,
    ]
    if args.train_extra:
        cmd.extend(args.train_extra.split())

    print(f"\n{'='*60}")
    print(f"z={z:3d}  out={out_dir}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        print(f"WARNING: train.py failed for z={z} (exit {ret.returncode})")
        return False
    return True


def load_cv_row(z, out_base, rank_metric):
    """Load cv_results.json for one z; return a flat dict."""
    path = cv_results_path(out_base, z)
    row = {"z": z}
    if not os.path.exists(path):
        return row
    with open(path) as f:
        r = json.load(f)
    for key in [
        "best_val_acc_mean", "best_val_acc_std",
        "best_val_balanced_mean", "best_val_balanced_std",
        "best_val_hmean_mean", "best_val_hmean_std",
        "test_acc_mean", "test_acc_std",
        "test_balanced_mean", "test_balanced_std",
        "test_hmean_mean", "test_hmean_std",
    ]:
        row[key] = r.get(key)
    return row


def aggregate_and_rank(z_list, args):
    rows = [load_cv_row(z, os.path.join(PROJECT_ROOT, args.out_base), args.rank_metric)
            for z in z_list]
    df = pd.DataFrame(rows)

    results_dir = os.path.dirname(os.path.join(PROJECT_ROOT, args.results_csv))
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(PROJECT_ROOT, args.results_csv), index=False)
    print(f"\nAggregated results saved to: {args.results_csv}")

    metric = args.rank_metric
    std_col = metric.replace("_mean", "_std")
    valid = df[df[metric].notna()].copy()
    if valid.empty:
        print("No valid results found.")
        return df

    valid = valid.sort_values(metric, ascending=False).reset_index(drop=True)
    valid["rank"] = valid.index + 1

    # ---- pretty table ----
    cols = ["rank", "z", metric, std_col,
            "best_val_acc_mean", "best_val_balanced_mean",
            "test_acc_mean",     "test_balanced_mean"]
    cols = [c for c in cols if c in valid.columns]
    print("\n" + "="*70)
    print(f"SLICE CV SWEEP RANKING  (metric={metric})")
    print("="*70)
    print(valid[cols].to_string(index=False))

    # ---- top-k ----
    k = min(args.topk, len(valid))
    topk_z = valid["z"].iloc[:k].tolist()
    topk_vals = valid[metric].iloc[:k].tolist()
    topk_stds = valid[std_col].iloc[:k].tolist() if std_col in valid.columns else [None] * k

    print(f"\nTop-{k} z indices (by {metric}):")
    for rank_i, (z, val, std) in enumerate(zip(topk_z, topk_vals, topk_stds), 1):
        std_str = f" ± {std:.4f}" if std is not None else ""
        print(f"  #{rank_i}  z={z:3d}   {metric}={val:.4f}{std_str}")

    print(f"\nSuggested --z_list for follow-up full training:")
    print(f"  --z_list {' '.join(str(z) for z in topk_z)}")

    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- single-z mode (SLURM array task) ----
    if args.single_z is not None:
        run_one_z(args.single_z, args)
        return

    # ---- aggregate-only mode ----
    if args.aggregate_only:
        z_list = build_z_list(args)
        aggregate_and_rank(z_list, args)
        return

    # ---- sequential sweep ----
    z_list = build_z_list(args)
    print(f"Slice CV sweep: {len(z_list)} z values, {args.kfold}-fold CV, {args.epochs} epochs/fold")
    print(f"z_list = {z_list}")
    print(f"Total training runs: {len(z_list) * args.kfold}")

    for z in z_list:
        run_one_z(z, args)

    aggregate_and_rank(z_list, args)


if __name__ == "__main__":
    main()
