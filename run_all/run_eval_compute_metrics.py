#!/usr/bin/env python3
"""
Evaluate multiple models: accuracy, inference time, speedup, avg exit, params, peak memory.

Pipeline:
  1. Baseline, Only EE
  2. DTT only (L2/Attn/Random) -> pick best
  3. DTT + EE
  4. Teacher-Student (DTT+EE as student)

Metrics (compute savings): ms/sample, speedup, avg_exit, num_params_M, peak_memory_MB, GFLOPs.
Usage:
  python run_all/run_eval_compute_metrics.py  # default configs
  python run_all/run_eval_compute_metrics.py --run_dirs ...
"""
import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset_nii2d import Nii2DSliceDataset
from src.models.vit2d import build_vit2d

try:
    from ptflops import get_model_complexity_info  # type: ignore
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", default=None,
                   help="Run dirs with best.pt and results.json. Default: common configs")
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--manifest", default="test.csv")
    p.add_argument("--data_root", default=".")
    p.add_argument("--label_map", default="CN=0,MCI=1,AD=2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.8, help="Early-exit threshold")
    p.add_argument("--warmup", type=int, default=5, help="Warmup batches before timing")
    p.add_argument("--out_csv", default="results/compute_metrics.csv")
    return p.parse_args()


def parse_label_map(s):
    return {k: int(v) for k, v in (x.split("=") for x in s.split(","))}


def load_config(run_dir):
    rpath = os.path.join(run_dir, "results.json")
    if not os.path.exists(rpath):
        return None
    with open(rpath) as f:
        return json.load(f)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, device, input_shape=(1, 224, 224)):
    """Return GFLOPs (2*MACs/1e9). Returns None if ptflops unavailable or fails."""
    if not HAS_PTFLOPS:
        return None
    try:
        model.eval()
        # input_constructor: ptflops passes (C,H,W), return dict for model(x=...)
        macs, _ = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            backend="aten",
            input_constructor=lambda shp: {"x": torch.randn(1, *shp).to(device)},
        )
        # GFLOPs = 2 * MACs / 1e9 (1 MAC ≈ 2 FLOPs for matmul)
        return round(2 * macs / 1e9, 2) if macs else None
    except Exception as e:
        print(f"  [FLOPs warning] {e}")
        return None


@torch.no_grad()
def evaluate_with_timing(model, loader, device, tau=None, warmup=5):
    model.eval()
    correct = 0
    total = 0
    exit_counts = {1: 0, 2: 0, 3: 0} if tau else None
    times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if tau:
            out = model(x, tau=tau)
            if isinstance(out, tuple) and len(out) == 2:
                logits, exit_layer = out
                for k in (1, 2, 3):
                    exit_counts[k] += (exit_layer == k).sum().item()
            else:
                logits = out
                exit_counts[3] += x.size(0)
        else:
            logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    acc = correct / max(total, 1)
    n_batches_timed = len(times)
    ms_per_sample = (sum(times) / max(n_batches_timed, 1) / loader.batch_size) * 1000
    avg_exit = None
    if exit_counts and total > 0:
        avg_exit = (1 * exit_counts[1] + 2 * exit_counts[2] + 3 * exit_counts[3]) / total

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return acc, ms_per_sample, avg_exit, exit_counts, peak_mb


def default_run_dirs(project_root):
    """
    Pipeline: Baseline -> Only EE -> DTT only (3 methods) -> DTT+EE -> Student
    """
    base = project_root
    return [
        ("Baseline", os.path.join(base, "runs/vit2d_baseline")),
        ("Only_EE", os.path.join(base, "runs/compare_ablation/only_ee")),
        ("DTT_L2_only", os.path.join(base, "runs/compare_dtt_only/l2")),
        ("DTT_Attn_only", os.path.join(base, "runs/compare_dtt_only/attn")),
        ("DTT_Random_only", os.path.join(base, "runs/compare_dtt_only/random")),
        ("DTT_L2+EE", os.path.join(base, "runs/compare_thin_methods/l2")),
        ("DTT_Attn+EE", os.path.join(base, "runs/compare_thin_methods/attn")),
        ("DTT_Random+EE", os.path.join(base, "runs/compare_thin_methods/random")),
        ("Student", os.path.join(base, "runs/student_distill")),
    ]


def main():
    args = parse_args()
    label_map = parse_label_map(args.label_map)
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(args.manifest_dir, args.manifest)
    slice_cfg = {"slice_selector": "fixed", "data_root": args.data_root, "z_index": 77}
    ds = Nii2DSliceDataset(csv_path, label_map=label_map, **slice_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.run_dirs:
        run_list = [(os.path.basename(d.rstrip("/")), d) for d in args.run_dirs]
    else:
        run_list = default_run_dirs(PROJECT_ROOT)

    rows = []
    ref_time_ms = None  # first model as reference for speedup

    for name, run_dir in run_list:
        ckpt_path = os.path.join(run_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skip {name}: {ckpt_path} not found")
            rows.append({
                "config": name, "test_acc": None, "ms_per_sample": None,
                "speedup": None, "avg_exit": None, "num_params_M": None, "peak_memory_MB": None,
                "GFLOPs": None,
            })
            continue

        cfg = load_config(run_dir)
        thinning = cfg.get("thinning", False) if cfg else False
        thin_method = cfg.get("thin_method", "attn") if cfg else "attn"
        early_exit = cfg.get("early_exit", False) if cfg else False

        model = build_vit2d(
            num_classes=num_classes,
            thinning=thinning,
            thin_method=thin_method,
            enable_early_exit=early_exit,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)

        tau = args.tau if early_exit else None
        acc, ms, avg_exit, _, peak_mb = evaluate_with_timing(
            model, loader, device, tau=tau, warmup=args.warmup
        )

        num_params = count_params(model)
        gflops = count_flops(model, device)
        if ref_time_ms is None:
            ref_time_ms = ms
        speedup = ref_time_ms / ms if ref_time_ms and ms else None

        rows.append({
            "config": name,
            "test_acc": acc,
            "ms_per_sample": ms,
            "speedup": speedup,
            "avg_exit": avg_exit,
            "num_params_M": round(num_params / 1e6, 2),
            "peak_memory_MB": round(peak_mb, 1) if peak_mb else None,
            "GFLOPs": gflops,
        })
        gf = f" GFLOPs={gflops}" if gflops else ""
        pm = f" peak_mb={peak_mb:.1f}" if peak_mb else ""
        print(f"{name}: acc={acc:.4f} ms/sample={ms:.2f} speedup={speedup:.2f}x avg_exit={avg_exit}{pm}{gf}")

    # Speedup = ref_time / this_time (first model as ref)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")

    # Also write markdown
    md_path = args.out_csv.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write("# Accuracy vs Compute\n\n")
        f.write("| Config | Test Acc (%) | ms/sample | Speedup | Avg Exit | Params (M) | Peak Mem (MB) | GFLOPs |\n")
        f.write("|--------|--------------|------------|---------|----------|------------|---------------|--------|\n")
        for _, r in df.iterrows():
            acc_s = f"{r['test_acc']*100:.2f}" if pd.notna(r["test_acc"]) else "—"
            ms_s = f"{r['ms_per_sample']:.1f}" if pd.notna(r["ms_per_sample"]) else "—"
            sp_s = f"{r['speedup']:.2f}x" if pd.notna(r["speedup"]) else "—"
            ex_s = f"{r['avg_exit']:.2f}" if pd.notna(r["avg_exit"]) else "—"
            pm_s = f"{r['num_params_M']:.2f}" if pd.notna(r["num_params_M"]) else "—"
            mem_s = f"{r['peak_memory_MB']:.1f}" if pd.notna(r["peak_memory_MB"]) else "—"
            gf_s = f"{r['GFLOPs']:.2f}" if pd.notna(r["GFLOPs"]) else "—"
            f.write(f"| {r['config']} | {acc_s} | {ms_s} | {sp_s} | {ex_s} | {pm_s} | {mem_s} | {gf_s} |\n")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
