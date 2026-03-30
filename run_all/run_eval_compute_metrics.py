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

from src.data.dataset_nii2d import Nii2DSliceDataset, in_chans_for_slice_stack_mode
from src.models.vit2d import build_vit2d

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
    p.add_argument("--label_map", default="CN=0,AD=1", help="Binary: CN=0,AD=1")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.8, help="Early-exit confidence threshold")
    p.add_argument("--tau_u", type=float, default=None, help="Proposal §3.6: entropy threshold for uncertainty-guided exit")
    p.add_argument("--warmup", type=int, default=5, help="Warmup batches before timing")
    p.add_argument("--out_csv", default="results/compute_metrics.csv")
    p.add_argument("--no_pretrained", action="store_true", help="Skip ImageNet weights (use when loading from checkpoint)")
    p.add_argument("--z_index", type=int, default=144, help="Fixed slice for eval dataloader (match training)")
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
def evaluate_with_timing(model, loader, device, tau=None, tau_u=None, warmup=5, collect_for_auroc=False):
    """Proposal §4.2: Accuracy, AUROC, ms/sample, avg_exit, peak_memory."""
    model.eval()
    correct = 0
    total = 0
    exit_counts = {1: 0, 2: 0, 3: 0} if tau else None
    times = []
    all_logits = []
    all_labels = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if tau:
            out = model(x, tau=tau, tau_u=tau_u)
            if isinstance(out, tuple) and len(out) >= 2:
                logits, exit_layer = out[0], out[1]
                for k in (1, 2, 3):
                    exit_counts[k] += (exit_layer == k).sum().item()
            else:
                logits = out
                exit_counts[3] += x.size(0)
        else:
            out = model(x) if not hasattr(model, "forward_all") else model.forward_all(x)[2]
            logits = out[0] if isinstance(out, tuple) else out
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if collect_for_auroc and HAS_SKLEARN:
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    acc = correct / max(total, 1)
    n_batches_timed = len(times)
    ms_per_sample = (sum(times) / max(n_batches_timed, 1) / loader.batch_size) * 1000
    avg_exit = None
    if exit_counts and total > 0:
        avg_exit = (1 * exit_counts[1] + 2 * exit_counts[2] + 3 * exit_counts[3]) / total

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    auroc = None
    if collect_for_auroc and HAS_SKLEARN and all_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        probs = torch.softmax(logits_cat, dim=1)
        num_classes = probs.size(1)
        if num_classes == 2:
            auroc = roc_auc_score(labels_cat.numpy(), probs[:, 1].numpy())
        elif num_classes > 2:
            auroc = roc_auc_score(labels_cat.numpy(), probs.numpy(), multi_class="ovr", average="macro")

    return acc, auroc, ms_per_sample, avg_exit, exit_counts, peak_mb


def default_run_dirs(project_root):
    """
    Default table order: first row = reference for **speedup** (wall-clock ref / this_model).

    Mainline (stack3, z=115) + optional DTT / thin-method dirs if present (skipped if no best.pt).

    Notes:
    - **ms/sample**, **avg_exit** (early-exit), **peak_memory_MB**: measured at inference.
    - **GFLOPs**: static estimate (full forward); dynamic token dropping in DTT is better reflected by **ms/sample** than GFLOPs alone.
    """
    base = project_root
    r = os.path.join
    return [
        ("Baseline_stack3_z115", r(base, "runs/vit2d_baseline_stack3_z115")),
        ("DTT_EE_attn", r(base, "runs/vit2d_dtt_ee_attn")),
        ("Student_distill_z115", r(base, "runs/student_distill_stack3_z115")),
        ("DTT_only_l2", r(base, "runs/compare_dtt_only/l2")),
        ("DTT_only_attn", r(base, "runs/compare_dtt_only/attn")),
        ("DTT_only_random", r(base, "runs/compare_dtt_only/random")),
        ("DTT_only_learnable", r(base, "runs/compare_dtt_only/learnable")),
        ("Thin_baseline", r(base, "runs/compare_thin_methods/baseline")),
        ("Thin_l2", r(base, "runs/compare_thin_methods/l2")),
        ("Thin_attn", r(base, "runs/compare_thin_methods/attn")),
        ("Thin_random", r(base, "runs/compare_thin_methods/random")),
        ("Thin_learnable", r(base, "runs/compare_thin_methods/learnable")),
        # Legacy / ablation paths (may be absent)
        ("Baseline_legacy", r(base, "runs/vit2d_baseline")),
        ("Only_EE", r(base, "runs/compare_ablation/only_ee")),
        ("Student_legacy", r(base, "runs/student_distill")),
    ]


def main():
    args = parse_args()
    label_map = parse_label_map(args.label_map)
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(args.manifest_dir, args.manifest)

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
                "config": name, "test_acc": None, "test_auroc": None, "ms_per_sample": None,
                "speedup": None, "avg_exit": None, "num_params_M": None, "peak_memory_MB": None,
                "GFLOPs": None,
            })
            continue

        cfg = load_config(run_dir)
        thinning = cfg.get("thinning", False) if cfg else False
        thin_method = cfg.get("thin_method", "attn") if cfg else "attn"
        early_exit = cfg.get("early_exit", False) if cfg else False
        use_anatomical_prior = cfg.get("use_anatomical_prior", False) if cfg else False

        slice_stack_mode = (cfg.get("slice_stack_mode", "single") if cfg else "single") or "single"
        z_for_ds = cfg.get("z_index", args.z_index) if cfg and cfg.get("z_index") is not None else args.z_index
        in_chans = cfg.get("in_chans") if cfg else None
        if in_chans is None:
            in_chans = in_chans_for_slice_stack_mode(slice_stack_mode)

        slice_cfg = {
            "slice_selector": "fixed",
            "data_root": args.data_root,
            "z_index": z_for_ds,
            "slice_stack_mode": slice_stack_mode,
        }
        ds = Nii2DSliceDataset(csv_path, label_map=label_map, **slice_cfg)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = build_vit2d(
            num_classes=num_classes,
            thinning=thinning,
            thin_method=thin_method,
            enable_early_exit=early_exit,
            use_anatomical_prior=use_anatomical_prior,
            anatomical_prior_path=cfg.get("anatomical_prior_path") if cfg else None,
            anatomical_prior_slice=cfg.get("anatomical_prior_slice") if cfg else None,
            pretrained=not args.no_pretrained,
            in_chans=in_chans,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)

        tau = args.tau if early_exit else None
        acc, auroc, ms, avg_exit, _, peak_mb = evaluate_with_timing(
            model, loader, device, tau=tau, tau_u=args.tau_u, warmup=args.warmup,
            collect_for_auroc=True,
        )

        num_params = count_params(model)
        gflops = count_flops(model, device, input_shape=(in_chans, 224, 224))
        if ref_time_ms is None:
            ref_time_ms = ms
        speedup = ref_time_ms / ms if ref_time_ms and ms else None

        rows.append({
            "config": name,
            "test_acc": acc,
            "test_auroc": auroc,
            "ms_per_sample": ms,
            "speedup": speedup,
            "avg_exit": avg_exit,
            "num_params_M": round(num_params / 1e6, 2),
            "peak_memory_MB": round(peak_mb, 1) if peak_mb else None,
            "GFLOPs": gflops,
        })
        gf = f" GFLOPs={gflops}" if gflops else ""
        pm = f" peak_mb={peak_mb:.1f}" if peak_mb else ""
        auc_s = f" AUROC={auroc:.4f}" if auroc is not None else ""
        print(f"{name}: acc={acc:.4f}{auc_s} ms/sample={ms:.2f} speedup={speedup:.2f}x avg_exit={avg_exit}{pm}{gf}")

    # speedup column = ref_ms / this_ms where ref is the **first row that had a valid checkpoint**
    print("\n# speedup is vs first successful model in this table (see config column order).", file=sys.stderr)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")

    # Proposal §4.3: Pareto front (accuracy vs FLOPs)
    if "GFLOPs" in df.columns and "test_acc" in df.columns:
        valid = df.dropna(subset=["test_acc", "GFLOPs"])
        if len(valid) >= 2:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                for _, r in valid.iterrows():
                    ax.scatter(r["GFLOPs"], r["test_acc"] * 100, label=r["config"], s=80)
                ax.set_xlabel("GFLOPs")
                ax.set_ylabel("Test Accuracy (%)")
                ax.set_title("Proposal §4.3: Token Retention vs Performance (Accuracy vs FLOPs)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                ax.grid(True, alpha=0.3)
                pareto_path = args.out_csv.replace(".csv", "_pareto.png")
                plt.tight_layout()
                plt.savefig(pareto_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Pareto plot: {pareto_path}")
            except ImportError:
                pass

    # Also write markdown
    md_path = args.out_csv.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write("# Accuracy / AUROC vs Compute (Proposal §4.2)\n\n")
        f.write("| Config | Test Acc (%) | AUROC | ms/sample | Speedup | Avg Exit | Params (M) | Peak Mem (MB) | GFLOPs |\n")
        f.write("|--------|--------------|-------|------------|---------|----------|------------|---------------|--------|\n")
        for _, r in df.iterrows():
            acc_s = f"{r['test_acc']*100:.2f}" if pd.notna(r["test_acc"]) else "—"
            auc_s = f"{r['test_auroc']:.4f}" if "test_auroc" in r and pd.notna(r.get("test_auroc")) else "—"
            ms_s = f"{r['ms_per_sample']:.1f}" if pd.notna(r["ms_per_sample"]) else "—"
            sp_s = f"{r['speedup']:.2f}x" if pd.notna(r["speedup"]) else "—"
            ex_s = f"{r['avg_exit']:.2f}" if pd.notna(r["avg_exit"]) else "—"
            pm_s = f"{r['num_params_M']:.2f}" if pd.notna(r["num_params_M"]) else "—"
            mem_s = f"{r['peak_memory_MB']:.1f}" if pd.notna(r["peak_memory_MB"]) else "—"
            gf_s = f"{r['GFLOPs']:.2f}" if pd.notna(r["GFLOPs"]) else "—"
            f.write(f"| {r['config']} | {acc_s} | {auc_s} | {ms_s} | {sp_s} | {ex_s} | {pm_s} | {mem_s} | {gf_s} |\n")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
