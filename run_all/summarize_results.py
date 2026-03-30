#!/usr/bin/env python3
"""
Summarize training results from results.json under run directories.

Typical HPC usage (from project root, e.g. /scratch/.../adni-dttvit):

  python run_all/summarize_results.py
  python run_all/summarize_results.py --glob 'runs/vit2d_baseline_stack3_z*/results.json'
  python run_all/summarize_results.py --glob 'runs/*/results.json' --sort test_balanced
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _hmean_acc_bal(a: Optional[float], b: Optional[float], eps: float = 1e-8) -> Optional[float]:
    """Harmonic mean of acc and balanced_acc (same as train.py val_hmean)."""
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if a <= eps or b <= eps:
        return None
    return 2.0 * a * b / (a + b)


def main():
    p = argparse.ArgumentParser(description="Print table from results.json files")
    p.add_argument(
        "--glob",
        default="runs/vit2d_baseline_*/results.json",
        help="Glob relative to cwd (default: 2.5D / single baseline vit2d_baseline_* runs)",
    )
    p.add_argument(
        "--cwd",
        default=".",
        help="Working directory for glob (default: current directory)",
    )
    p.add_argument(
        "--sort",
        default="dir",
        choices=[
            "dir",
            "z_index",
            "best_val_balanced",
            "test_balanced",
            "best_val_acc",
            "test_acc",
            "best_val_hmean",
            "test_hmean",
        ],
        help="Sort key (z_index requires z in run path or results.json)",
    )
    args = p.parse_args()
    root = os.path.abspath(args.cwd)
    pattern = os.path.join(root, args.glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files matched: {pattern}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in paths:
        run_dir = os.path.dirname(path)
        rel = os.path.relpath(run_dir, root)
        try:
            d = load_results(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue
        z = d.get("z_index")
        ta, tb = d.get("test_acc"), d.get("test_balanced")
        test_hmean = d.get("test_hmean")
        if test_hmean is None:
            test_hmean = _hmean_acc_bal(ta, tb)
        rows.append({
            "dir": rel,
            "z_index": z,
            "slice_stack": d.get("slice_stack_mode", ""),
            "in_chans": d.get("in_chans", ""),
            "best_val_acc": d.get("best_val_acc"),
            "best_val_balanced": d.get("best_val_balanced"),
            "best_val_hmean": d.get("best_val_hmean"),
            "best_val_loss": d.get("best_val_loss"),
            "test_acc": ta,
            "test_balanced": tb,
            "test_hmean": test_hmean,
        })

    if args.sort != "dir":
        rev = True

        def key(r):
            v = r.get(args.sort)
            if v is None:
                return float("-inf")
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("-inf")

        rows.sort(key=key, reverse=rev)
    else:
        rows.sort(key=lambda r: r["dir"])

    # header (val_hmean = harmonic mean val_acc & val_bal; test_hmean same on test set)
    cols = [
        "dir", "z", "mode", "C", "val_acc", "val_bal", "val_hmean",
        "val_loss", "test_acc", "test_bal", "test_hmean",
    ]
    print("\t".join(cols))
    for r in rows:
        print(
            f"{r['dir']}\t"
            f"{r['z_index']}\t"
            f"{r['slice_stack']}\t"
            f"{r['in_chans']}\t"
            f"{r['best_val_acc']}\t"
            f"{r['best_val_balanced']}\t"
            f"{r['best_val_hmean']}\t"
            f"{r['best_val_loss']}\t"
            f"{r['test_acc']}\t"
            f"{r['test_balanced']}\t"
            f"{r['test_hmean']}"
        )
    print(f"\n# n={len(rows)}  glob={args.glob}  cwd={root}", file=sys.stderr)


if __name__ == "__main__":
    main()
