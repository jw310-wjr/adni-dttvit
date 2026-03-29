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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


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
        choices=["dir", "z_index", "best_val_balanced", "test_balanced", "best_val_acc", "test_acc"],
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
        rows.append({
            "dir": rel,
            "z_index": z,
            "slice_stack": d.get("slice_stack_mode", ""),
            "in_chans": d.get("in_chans", ""),
            "best_val_acc": d.get("best_val_acc"),
            "best_val_balanced": d.get("best_val_balanced"),
            "best_val_hmean": d.get("best_val_hmean"),
            "test_acc": d.get("test_acc"),
            "test_balanced": d.get("test_balanced"),
        })

    if args.sort != "dir":
        rev = True

        def key(r):
            v = r.get(args.sort)
            if v is None:
                return float("-inf")
            return float(v)

        rows.sort(key=key, reverse=rev)
    else:
        rows.sort(key=lambda r: r["dir"])

    # header
    cols = ["dir", "z", "mode", "C", "val_acc", "val_bal", "val_hmean", "test_acc", "test_bal"]
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
            f"{r['test_acc']}\t"
            f"{r['test_balanced']}"
        )
    print(f"\n# n={len(rows)}  glob={args.glob}  cwd={root}", file=sys.stderr)


if __name__ == "__main__":
    main()
