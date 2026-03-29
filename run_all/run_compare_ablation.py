#!/usr/bin/env python3
"""
Ablation: Baseline vs +DTT vs +DTT+Early Exit.
Shows incremental effect of adding DTT, then Early Exit.
Usage:
  python run_all/run_compare_ablation.py
  python run_all/run_compare_ablation.py --epochs 50 --thin_method attn
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
    p.add_argument("--out_base", default="runs/compare_ablation")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--thin_method", default="attn", choices=["l2", "attn", "random", "learnable"],
                   help="DTT method: l2/attn/random (heuristic) or learnable (ScoreHead)")
    p.add_argument("--train_extra", default="", help="Extra args for train.py (e.g. --amp)")
    p.add_argument(
        "--z_index",
        type=int,
        default=144,
        help="Fixed slice index for all ablation configs (run slice_selection first to pick a good z)",
    )
    p.add_argument("--skip_train", action="store_true", help="Skip training, only plot from existing results")
    return p.parse_args()


CONFIGS = [
    ("baseline", "Baseline", False, False),
    ("only_ee", "Only EE", False, True),
    ("dtt_only", "Baseline+DTT", True, False),
    ("dtt_ee", "Baseline+DTT+EE", True, True),
]


def run_training(args, train_script):
    rows = []
    for config_id, label, thinning, early_exit in CONFIGS:
        out_dir = os.path.join(args.out_base, config_id)
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
        if thinning:
            cmd.extend(["--thinning", "--thin_method", args.thin_method])
        if early_exit:
            cmd.extend(["--early_exit"])
        if args.train_extra:
            cmd.extend(args.train_extra.split())

        print(f"\n{'='*60}")
        print(f"Running {label} -> {out_dir}")
        print(f"{'='*60}")
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if ret.returncode != 0:
            print(f"WARNING: train.py failed for {config_id} (exit {ret.returncode})")
            rows.append({"config": label, "val_acc": None, "test_acc": None})
            continue

        results_path = os.path.join(out_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                r = json.load(f)
            rows.append({
                "config": label,
                "val_acc": r.get("best_val_acc"),
                "test_acc": r.get("test_acc"),
            })
        else:
            rows.append({"config": label, "val_acc": None, "test_acc": None})
    return pd.DataFrame(rows)


def load_existing_results(args):
    rows = []
    for config_id, label, _, _ in CONFIGS:
        out_dir = os.path.join(args.out_base, config_id)
        results_path = os.path.join(out_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                r = json.load(f)
            rows.append({
                "config": label,
                "val_acc": r.get("best_val_acc"),
                "test_acc": r.get("test_acc"),
            })
        else:
            rows.append({"config": label, "val_acc": None, "test_acc": None})
    return pd.DataFrame(rows)


def plot_results(df, out_path):
    df_valid = df.dropna(subset=["val_acc", "test_acc"])
    if df_valid.empty:
        print("No valid results to plot.")
        return

    labels = df_valid["config"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, df_valid["val_acc"] * 100, width, label="Val Acc (%)", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, df_valid["test_acc"] * 100, width, label="Test Acc (%)", color="#3498db")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation: Baseline vs +DTT vs +DTT+Early Exit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
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
    csv_path = os.path.join(results_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    md_path = os.path.join(results_dir, "ablation_results.md")
    with open(md_path, "w") as f:
        f.write("# Ablation: Baseline vs +DTT vs +DTT+Early Exit\n\n")
        f.write("| Config | Val Acc (%) | Test Acc (%) |\n")
        f.write("|--------|-------------|---------------|\n")
        for _, row in df.iterrows():
            va = f"{row['val_acc']*100:.2f}" if pd.notna(row["val_acc"]) else "—"
            ta = f"{row['test_acc']*100:.2f}" if pd.notna(row["test_acc"]) else "—"
            f.write(f"| {row['config']} | {va} | {ta} |\n")
        f.write("\n")
    print(f"Markdown saved: {md_path}")

    tex_path = os.path.join(results_dir, "ablation_results.tex")
    with open(tex_path, "w") as f:
        f.write("% Ablation - LaTeX table for paper\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation: Baseline vs +DTT vs +DTT+Early Exit on ADNI classification.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Config & Val Acc (\\%) & Test Acc (\\%)\\\\\n")
        f.write("\\hline\n")
        for _, row in df.iterrows():
            va = f"{row['val_acc']*100:.2f}" if pd.notna(row["val_acc"]) else "---"
            ta = f"{row['test_acc']*100:.2f}" if pd.notna(row["test_acc"]) else "---"
            f.write(f"{row['config']} & {va} & {ta} \\\\\n")
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

    plot_results(df, os.path.join(args.results_dir, "ablation_comparison.png"))
    write_text_results(df, args.results_dir)


if __name__ == "__main__":
    main()
