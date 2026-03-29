#!/usr/bin/env python3
"""
Batch run DTT+Early Exit for each thin_method (l2, attn, random, learnable) and aggregate results.
Generates figures and text summaries suitable for papers.
Usage:
  python run_all/run_compare_thin_methods.py
  python run_all/run_compare_thin_methods.py --epochs 50
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--data_root", default=".")
    p.add_argument("--out_base", default="runs/compare_thin_methods")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--methods", nargs="+", default=["baseline", "l2", "attn", "random", "learnable"],
                   help="DTT+EE methods: l2/attn/random (heuristic), learnable (ScoreHead)")
    p.add_argument("--train_extra", default="", help="Extra args for train.py (e.g. --amp)")
    p.add_argument(
        "--z_index",
        type=int,
        default=144,
        help="Fixed slice index (match slice_selection / baseline)",
    )
    p.add_argument("--skip_train", action="store_true", help="Skip training, only plot from existing results")
    return p.parse_args()


METHOD_LABELS = {"baseline": "Baseline", "l2": "DTT+L2", "attn": "DTT+CLS-attn", "random": "DTT+Random", "learnable": "DTT+Learnable"}


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
        ]
        if method == "baseline":
            # Baseline: no DTT, no Early Exit
            pass
        else:
            cmd.extend(["--thinning", "--thin_method", method, "--early_exit"])
        if args.train_extra:
            cmd.extend(args.train_extra.split())

        print(f"\n{'='*60}")
        print(f"Running {method} -> {out_dir}")
        print(f"{'='*60}")
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if ret.returncode != 0:
            print(f"WARNING: train.py failed for method={method} (exit {ret.returncode})")
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


def plot_results(df, out_path):
    """Bar chart: val_acc and test_acc per method."""
    df_valid = df.dropna(subset=["val_acc", "test_acc"])
    if df_valid.empty:
        print("No valid results to plot.")
        return

    labels = [METHOD_LABELS.get(m, m) for m in df_valid["thin_method"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, df_valid["val_acc"] * 100, width, label="Val Acc (%)", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, df_valid["test_acc"] * 100, width, label="Test Acc (%)", color="#3498db")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Baseline vs DTT+Early Exit Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}, {pdf_path}")


def write_text_results(df, results_dir):
    """Write CSV, Markdown table, and LaTeX table."""
    csv_path = os.path.join(results_dir, "thin_method_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # Markdown table
    md_path = os.path.join(results_dir, "thin_method_results.md")
    with open(md_path, "w") as f:
        f.write("# Baseline vs DTT+Early Exit Comparison\n\n")
        f.write("| Method | Val Acc (%) | Test Acc (%) |\n")
        f.write("|--------|-------------|---------------|\n")
        for _, row in df.iterrows():
            va = f"{row['val_acc']*100:.2f}" if pd.notna(row["val_acc"]) else "—"
            ta = f"{row['test_acc']*100:.2f}" if pd.notna(row["test_acc"]) else "—"
            label = METHOD_LABELS.get(row["thin_method"], row["thin_method"])
            f.write(f"| {label} | {va} | {ta} |\n")
        f.write("\n")
    print(f"Markdown saved: {md_path}")

    # LaTeX table (for paper)
    tex_path = os.path.join(results_dir, "thin_method_results.tex")
    with open(tex_path, "w") as f:
        f.write("% DTT Token Thinning Methods - LaTeX table for paper\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline vs DTT+Early Exit on ADNI classification.}\n")
        f.write("\\label{tab:thin_methods}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Method & Val Acc (\\%) & Test Acc (\\%)\\\\\n")
        f.write("\\hline\n")
        for _, row in df.iterrows():
            va = f"{row['val_acc']*100:.2f}" if pd.notna(row["val_acc"]) else "---"
            ta = f"{row['test_acc']*100:.2f}" if pd.notna(row["test_acc"]) else "---"
            label = METHOD_LABELS.get(row["thin_method"], row["thin_method"])
            f.write(f"{label} & {va} & {ta} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved: {tex_path}")


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
    print("Results:")
    print(df.to_string(index=False))

    best_row = df.loc[df["val_acc"].idxmax()] if df["val_acc"].notna().any() else None
    if best_row is not None:
        print(f"\nBest thin_method (by val_acc): {best_row['thin_method']} "
              f"(val={best_row['val_acc']:.4f}, test={best_row['test_acc']:.4f})")

    # Generate outputs for paper
    plot_results(df, os.path.join(args.results_dir, "thin_method_comparison.png"))
    write_text_results(df, args.results_dir)


if __name__ == "__main__":
    main()
