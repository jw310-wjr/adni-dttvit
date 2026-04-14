"""Convert existing train_log.csv files to TensorBoard tfevents format.

Usage:
    python run_all/csv_to_tb.py                        # convert all runs
    python run_all/csv_to_tb.py --runs_dir runs/compare_ablation
"""
import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.tensorboard import SummaryWriter


def convert(csv_path: str):
    tb_dir = os.path.join(os.path.dirname(csv_path), "tb")
    if os.path.exists(tb_dir) and os.listdir(tb_dir):
        print(f"  skip (tb already exists): {tb_dir}")
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  skip (empty): {csv_path}")
        return
    cols = rows[0].keys()
    writer = SummaryWriter(log_dir=tb_dir)
    for row in rows:
        ep = int(row["epoch"])
        loss_d = {"train": float(row["train_loss"])}
        if "val_loss" in cols:
            loss_d["val"] = float(row["val_loss"])
        writer.add_scalars("loss", loss_d, ep)

        acc_d = {"val": float(row["val_acc"])}
        if "train_acc" in cols:
            acc_d["train"] = float(row["train_acc"])
        writer.add_scalars("acc", acc_d, ep)

        if "train_balanced" in cols and "val_balanced" in cols:
            writer.add_scalars("acc_balanced",
                {"train": float(row["train_balanced"]), "val": float(row["val_balanced"])}, ep)
    writer.close()
    print(f"  converted ({len(rows)} epochs): {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    args = p.parse_args()

    runs_dir = os.path.join(PROJECT_ROOT, args.runs_dir)
    found = 0
    for root, dirs, files in os.walk(runs_dir):
        if "train_log.csv" in files:
            found += 1
            convert(os.path.join(root, "train_log.csv"))
    print(f"\nDone. Found {found} run(s).")


if __name__ == "__main__":
    main()
