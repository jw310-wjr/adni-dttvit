#!/usr/bin/env python3
"""
Two-stage Teacher-Student training (Knowledge Distillation).

Stage 1: Train teacher (e.g. full ViT baseline)
Stage 2: Train student (e.g. DTT+Early Exit) with distillation from teacher

Usage:
  python run_all/run_teacher_student.py
  python run_all/run_teacher_student.py --epochs 50 --distill_alpha 0.3
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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--teacher_out",
        default="runs/teacher",
        help="Output dir for teacher (stage 1)",
    )
    p.add_argument(
        "--student_out",
        default="runs/student_distill",
        help="Output dir for student (stage 2)",
    )
    p.add_argument(
        "--distill_temp",
        type=float,
        default=4.0,
        help="Distillation temperature",
    )
    p.add_argument(
        "--distill_alpha",
        type=float,
        default=0.5,
        help="CE weight; (1-alpha) for KD. loss = alpha*CE + (1-alpha)*KD",
    )
    p.add_argument(
        "--thin_method",
        default="learnable",
        choices=["l2", "attn", "random", "learnable"],
        help="Student thinning method (Proposal: learnable)",
    )
    p.add_argument(
        "--lambda_feat",
        type=float,
        default=0.1,
        help="Proposal §3.7: feature distillation weight",
    )
    p.add_argument(
        "--skip_teacher",
        action="store_true",
        help="Skip stage 1, use existing teacher (teacher_out/best.pt)",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--train_extra", default="")
    p.add_argument("--results_dir", default="results")
    p.add_argument(
        "--z_index",
        type=int,
        default=144,
        help="Fixed slice index (match slice_selection / baseline)",
    )
    return p.parse_args()


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(desc)
    print(f"{'='*60}")
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    args = parse_args()
    train_script = os.path.join(PROJECT_ROOT, "run_all", "train.py")

    teacher_ckpt = os.path.join(args.teacher_out, "best.pt")

    # Stage 1: Train teacher (full ViT, no DTT, no EE)
    if not args.skip_teacher:
        cmd = [
            sys.executable,
            train_script,
            "--manifest_dir", args.manifest_dir,
            "--data_root", args.data_root,
            "--slice_selector", "fixed",
            "--z_index", str(args.z_index),
            "--out_dir", args.teacher_out,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ]
        if args.amp:
            cmd.append("--amp")
        if args.train_extra:
            cmd.extend(args.train_extra.split())
        run_cmd(cmd, "Stage 1: Training teacher (full ViT baseline)")
    else:
        if not os.path.exists(teacher_ckpt):
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

    # Stage 2: Train student (DTT + Early Exit) with distillation (Proposal JSD-ViT)
    cmd = [
        sys.executable,
        train_script,
        "--manifest_dir", args.manifest_dir,
        "--data_root", args.data_root,
        "--slice_selector", "fixed",
        "--z_index", str(args.z_index),
        "--thinning",
        "--thin_method", args.thin_method,
        "--early_exit",
        "--teacher_ckpt", teacher_ckpt,
        "--distill_temp", str(args.distill_temp),
        "--distill_alpha", str(args.distill_alpha),
        "--lambda_feat", str(args.lambda_feat),
        "--out_dir", args.student_out,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]
    if args.amp:
        cmd.append("--amp")
    if args.train_extra:
        cmd.extend(args.train_extra.split())
    run_cmd(cmd, "Stage 2: Training student (DTT+EE) with knowledge distillation")

    # Aggregate results for paper
    os.makedirs(args.results_dir, exist_ok=True)
    rows = []
    for name, out_dir in [("Teacher", args.teacher_out), ("Student", args.student_out)]:
        rpath = os.path.join(out_dir, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            rows.append({"model": name, "val_acc": r.get("best_val_acc"), "test_acc": r.get("test_acc")})
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.results_dir, "teacher_student_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved: {csv_path}")
        print(df.to_string(index=False))

    print(f"\nDone. Teacher: {args.teacher_out} | Student: {args.student_out}")


if __name__ == "__main__":
    main()
