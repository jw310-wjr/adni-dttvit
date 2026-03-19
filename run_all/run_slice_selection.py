#!/usr/bin/env python3
"""
Batch run baseline for different fixed z slice indices and aggregate results.
Usage:
  python run_all/run_slice_selection.py
  python run_all/run_slice_selection.py --z_list 19 48 77 115 144 173
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--z_list",
        type=int,
        nargs="+",
        default=[19, 48, 77, 115, 144, 173],
        help="List of z indices to try",
    )
    p.add_argument(
        "--manifest_dir",
        default="manifests",
        help="Directory containing train.csv/val.csv/test.csv",
    )
    p.add_argument(
        "--data_root",
        default=".",
        help="Project root for resolving relative paths",
    )
    p.add_argument(
        "--out_base",
        default="runs/slice_selection",
        help="Base dir for each z run (each z gets runs/slice_selection/z_19, etc.)",
    )
    p.add_argument(
        "--results_csv",
        default="results/z_results.csv",
        help="Output CSV path for aggregated results",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs per run",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    p.add_argument(
        "--train_extra",
        default="",
        help="Extra args passed to train.py (e.g. --amp)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    train_script = os.path.join(PROJECT_ROOT, "run_all", "train.py")

    rows = []
    for z in args.z_list:
        out_dir = os.path.join(args.out_base, f"z_{z}")
        cmd = [
            sys.executable,
            train_script,
            "--manifest_dir", args.manifest_dir,
            "--data_root", args.data_root,
            "--slice_selector", "fixed",
            "--z_index", str(z),
            "--out_dir", out_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ]
        if args.train_extra:
            cmd.extend(args.train_extra.split())

        print(f"\n{'='*60}")
        print(f"Running z_index={z} -> {out_dir}")
        print(f"{'='*60}")
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if ret.returncode != 0:
            print(f"WARNING: train.py failed for z={z} (exit {ret.returncode})")
            rows.append({"z": z, "val_acc": None, "test_acc": None})
            continue

        results_path = os.path.join(out_dir, "results.json")
        if os.path.exists(results_path):
            import json
            with open(results_path) as f:
                r = json.load(f)
            rows.append({
                "z": z,
                "val_acc": r.get("best_val_acc"),
                "test_acc": r.get("test_acc"),
            })
        else:
            rows.append({"z": z, "val_acc": None, "test_acc": None})

    df = pd.DataFrame(rows)
    results_dir = os.path.dirname(args.results_csv)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    df.to_csv(args.results_csv, index=False)
    print(f"\nAggregated results saved to: {args.results_csv}")
    print(df.to_string(index=False))

    best_row = df.loc[df["val_acc"].idxmax()] if df["val_acc"].notna().any() else None
    if best_row is not None:
        print(f"\nBest z (by val_acc): {int(best_row['z'])} (val={best_row['val_acc']:.4f}, test={best_row['test_acc']:.4f})")


if __name__ == "__main__":
    main()
