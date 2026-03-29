#!/usr/bin/env python3
"""
Stage 1: Compare DTT methods (DTT only, no Early Exit).
Pick best DTT method for DTT+EE and Teacher-Student.

Pipeline: DTT_only(L2/Attn/Random) -> pick best -> DTT+EE -> Teacher-Student

Usage:
  python run_all/run_compare_dtt_only.py
  python run_all/run_compare_dtt_only.py --epochs 50
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--data_root", default=".")
    p.add_argument("--out_base", default="runs/compare_dtt_only")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--methods", nargs="+", default=["l2", "attn", "random", "learnable"],
                   help="DTT methods (learnable = Proposal §3.3 ScoreHead)")
    p.add_argument("--train_extra", default="", help="Extra args for train.py (e.g. --amp)")
    p.add_argument(
        "--z_index",
        type=int,
        default=144,
        help="Fixed slice index (match slice_selection / baseline)",
    )
    p.add_argument("--skip_train", action="store_true", help="Skip training, load existing results")
    return p.parse_args()


def run_training(args, train_script):
    rows = []
    for method in args.methods:
        out_dir = os.path.join(args.out_base, method)
        cmd = [
            sys.executable,
            train_script,
            "--manifest_dir", args.manifest_dir,
            "--data_root", args.data_root,
            "--slice_selector", "fixed",
            "--z_index", str(args.z_index),
            "--out_dir", out_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--thinning",
            "--thin_method", method,
        ]
        if args.train_extra:
            cmd.extend(args.train_extra.split())

        print(f"\n{'='*60}")
        print(f"Running DTT({method})_only -> {out_dir}")
        print(f"{'='*60}")
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if ret.returncode != 0:
            print(f"WARNING: train.py failed for {method} (exit {ret.returncode})")
            rows.append({"thin_method": method, "val_acc": None, "test_acc": None})
            continue

        results_path = os.path.join(out_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                r = json.load(f)
            rows.append({
                "thin_method": method,
                "val_acc": r.get("best_val_acc"),
                "test_acc": r.get("test_acc"),
            })
        else:
            rows.append({"thin_method": method, "val_acc": None, "test_acc": None})
    return pd.DataFrame(rows)


def load_existing_results(args):
    rows = []
    for method in args.methods:
        out_dir = os.path.join(args.out_base, method)
        results_path = os.path.join(out_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                r = json.load(f)
            rows.append({
                "thin_method": method,
                "val_acc": r.get("best_val_acc"),
                "test_acc": r.get("test_acc"),
            })
        else:
            rows.append({"thin_method": method, "val_acc": None, "test_acc": None})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    train_script = os.path.join(PROJECT_ROOT, "run_all", "train.py")

    os.makedirs(args.results_dir, exist_ok=True)

    if args.skip_train:
        df = load_existing_results(args)
        print("Skipping training, loading existing results.")
    else:
        df = run_training(args, train_script)

    print(f"\n{'='*60}")
    print("DTT-only comparison (pick best for DTT+EE):")
    print(df.to_string(index=False))

    csv_path = os.path.join(args.results_dir, "dtt_only_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
