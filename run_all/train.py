import argparse
import os, sys

# Add project root to PYTHONPATH so "import src...." works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import json
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.dataset_nii2d import Nii2DSliceDataset
from src.data.dataset_npy2d import ADNINPY2DDataset
from src.models.vit2d import build_vit2d


def parse_args():
    p = argparse.ArgumentParser()

    # manifests
    p.add_argument(
        "--manifest_dir",
        default="manifests",
        help="Directory containing train.csv/val.csv/test.csv"
    )
    p.add_argument(
        "--data_root",
        default=".",
        help="Project root for resolving relative paths in manifest"
    )

    # labels
    p.add_argument(
        "--label_map",
        default="CN=0,AD=1",
        help='Binary (CN vs AD): "CN=0,AD=1"'
    )

    # ---- Input type ----
    p.add_argument(
        "--input_type",
        default="nifti",
        choices=["nifti", "npy"],
        help="nifti: read nifti_path and select slice on the fly; npy: read pre-extracted 2D slice from npy_path"
    )

    # ---- Slice selection ----
    p.add_argument(
        "--slice_selector",
        default="fixed",
        choices=["middle", "entropy", "fixed"],
        help="2D slice selection method for 3D NIfTI"
    )
    p.add_argument(
        "--entropy_topk",
        type=int,
        default=5,
        help="Robust entropy selection: pick median index among top-k entropy slices"
    )
    p.add_argument(
        "--z_index",
        type=int,
        default=77,
        help="Fixed z slice index (used when slice_selector=fixed). 77 ≈ middle for ~155 slices"
    )
    p.add_argument(
        "--z_frac",
        type=float,
        default=None,
        help="Fixed z as fraction of depth (0-1, used when slice_selector=fixed if z_index not set)"
    )

    # training
    p.add_argument("--epochs", type=int, default=50, help="Default 50, aligned with SLURM baseline config")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)

    # output
    p.add_argument("--out_dir", default="runs/vit2d_baseline")

    # perf
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision if CUDA is available"
    )

    # ---- DTT / token thinning ----
    p.add_argument(
        "--thinning",
        action="store_true",
        help="Enable token thinning (DTT v1)"
    )
    p.add_argument(
        "--thin_method",
        default="l2",
        choices=["l2", "random", "attn", "learnable"],
        help="Token thinning method (learnable = Proposal §3.3 ScoreHead)"
    )
    p.add_argument(
        "--gumbel_tau",
        type=float,
        default=1.0,
        help="Proposal §3.4: Gumbel-Softmax temperature for learnable thinning (0=hard only)"
    )
    p.add_argument(
        "--debug_thin",
        action="store_true",
        help="Print token counts at thinning layers (sanity check)"
    )

    # ---- Early exit ----
    p.add_argument(
        "--early_exit",
        action="store_true",
        help="Enable early-exit heads (train multi-head CE)."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Weight for CE loss at exit head 1 (stage 1)."
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Weight for CE loss at exit head 2 (stage 2)."
    )

    # (optional) inference-time early exit evaluation
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="If set (e.g., 0.8), evaluate with confidence-based early exit at threshold tau."
    )
    p.add_argument(
        "--tau_u",
        type=float,
        default=None,
        help="Proposal §3.6: entropy threshold for uncertainty-guided early exit."
    )
    p.add_argument(
        "--exit_blocks",
        type=int,
        nargs="+",
        default=[3, 7, 11],
        help="Block indices for early-exit heads (stage ends)"
    )

    # ---- Teacher-Student (Knowledge Distillation) ----
    p.add_argument(
        "--teacher_ckpt",
        default=None,
        help="Path to teacher checkpoint for knowledge distillation"
    )
    p.add_argument(
        "--distill_temp",
        type=float,
        default=4.0,
        help="Temperature for distillation softmax (higher = softer)"
    )
    p.add_argument(
        "--distill_alpha",
        type=float,
        default=0.5,
        help="Weight for CE loss; (1-alpha) for KD loss. loss = alpha*CE + (1-alpha)*KD"
    )
    p.add_argument(
        "--teacher_thinning",
        action="store_true",
        help="Teacher uses thinning (must match teacher checkpoint)"
    )
    p.add_argument(
        "--teacher_early_exit",
        action="store_true",
        help="Teacher uses early exit (must match teacher checkpoint)"
    )
    p.add_argument(
        "--lambda_feat",
        type=float,
        default=0.0,
        help="Proposal §3.7: weight for feature distillation L_feat = ||f_student - f_teacher||²"
    )

    # ---- Proposal: anatomical regularization & budget-aware ----
    p.add_argument(
        "--use_anatomical_prior",
        action="store_true",
        help="Proposal §3.3: L_anatomy = ||M_token - M_prior||²"
    )
    p.add_argument(
        "--anatomical_prior_path",
        default=None,
        help="Path to atlas mask NIfTI (e.g. hippocampus ROI) for anatomically grounded prior. If unset, uses spatial center prior."
    )
    p.add_argument(
        "--anatomical_prior_slice",
        type=int,
        default=None,
        help="Slice index for anatomical mask (default: middle). Used with --anatomical_prior_path."
    )
    p.add_argument(
        "--lambda_anatomy",
        type=float,
        default=0.0,
        help="Weight for anatomical regularization"
    )
    p.add_argument(
        "--lambda_sparse",
        type=float,
        default=0.0,
        help="Proposal §3.8: weight for budget-aware L_sparse = Σ(N_ℓ/N_0 - r_ℓ)²"
    )
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use randomly initialized ViT (no ImageNet weights). For offline/air-gapped runs."
    )

    return p.parse_args()


def parse_label_map(s: str):
    return {k: int(v) for k, v in (x.split("=") for x in s.split(","))}


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device, tau=None, tau_u=None, criterion=None):
    """
    If tau is set: uncertainty-guided early exit (Proposal §3.6) with tau, tau_u.
    If criterion is set: also returns val_loss for logging.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    exit_counts = {1: 0, 2: 0, 3: 0} if tau is not None else None

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if tau is None:
            logits = (model.forward_all(x)[2] if hasattr(model, "forward_all") else model(x))
            pred = logits.argmax(dim=1)
        else:
            out = model(x, tau=tau, tau_u=tau_u)
            if isinstance(out, tuple) and len(out) >= 2:
                logits, exit_layer = out[0], out[1]
                for k in (1, 2, 3):
                    exit_counts[k] += (exit_layer == k).sum().item()
            else:
                logits = out
            pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()
        if criterion is not None:
            total_loss += criterion(logits, y).item()

    acc = correct / max(total, 1)
    val_loss = total_loss / max(len(loader), 1) if criterion is not None else None
    if tau is not None:
        avg_exit = (1 * exit_counts[1] + 2 * exit_counts[2] + 3 * exit_counts[3]) / max(total, 1)
        if criterion is not None:
            return acc, exit_counts, avg_exit, val_loss
        return acc, exit_counts, avg_exit
    return (acc, val_loss) if criterion is not None else acc


def _compute_aux_losses(model, aux, lambda_anatomy, lambda_sparse, schedule):
    """Proposal §3.3 L_anatomy, §3.8 L_sparse."""
    loss_aux = 0.0
    if lambda_anatomy > 0 and hasattr(model, "m_prior") and model.m_prior is not None:
        m_prior = model.m_prior.squeeze(0)  # [N]
        m_prior = m_prior / (m_prior.sum() + 1e-8)
        for m_token in aux.get("m_token_list", []):
            # m_token: [B, N], normalize to distribution
            m_token_norm = F.softmax(m_token, dim=1)
            loss_aux = loss_aux + ((m_token_norm - m_prior) ** 2).mean() * lambda_anatomy
    if lambda_sparse > 0:
        N0 = aux.get("N0", 196)
        for blk_idx, N_l in aux.get("token_counts", []):
            r_l = schedule.ratio_after_block(blk_idx)
            if r_l is not None:
                loss_aux = loss_aux + ((N_l / N0) - r_l) ** 2 * lambda_sparse
    return loss_aux


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    early_exit=False,
    alpha=0.3,
    beta=0.3,
    lambda_anatomy=0.0,
    lambda_sparse=0.0,
):
    """
    If early_exit: multi-head CE + optional L_anatomy, L_sparse (Proposal).
    Returns (train_loss, train_acc).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    schedule = getattr(model, "schedule", None) or type("S", (), {"ratio_after_block": lambda i: None})()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        def _step():
            if hasattr(model, "forward_all"):
                logits1, logits2, logits3, aux = model.forward_all(x)
                loss = criterion(logits3, y)
                if early_exit:
                    loss = loss + alpha * criterion(logits1, y) + beta * criterion(logits2, y)
                loss = loss + _compute_aux_losses(model, aux, lambda_anatomy, lambda_sparse, schedule)
                logits = logits3
            else:
                logits = model(x)
                loss = criterion(logits, y)
            return loss, logits

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, logits = _step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, logits = _step()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    train_loss = total_loss / max(len(loader), 1)
    train_acc = correct / max(total, 1)
    return train_loss, train_acc


def train_one_epoch_distill(
    student,
    teacher,
    loader,
    optimizer,
    criterion,
    device,
    temp: float,
    alpha: float,
    scaler=None,
    early_exit=False,
    alpha_ee=0.3,
    beta_ee=0.3,
    lambda_feat: float = 0.0,
    lambda_anatomy: float = 0.0,
    lambda_sparse: float = 0.0,
):
    """
    Proposal §3.7: L = L_cls + λ1*L_KD + λ2*L_feat + λ3*L_sparse + λ4*L_anatomy
    L_feat = ||f_student - f_teacher||²
    Returns (train_loss, train_acc).
    """
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    schedule = getattr(student, "schedule", None) or type("S", (), {"ratio_after_block": lambda i: None})()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            if hasattr(teacher, "forward_all"):
                t_logits1, t_logits2, t_logits3, t_aux = teacher.forward_all(x)
                teacher_logits = t_logits3
                teacher_feat = t_aux.get("cls_feature")
            else:
                teacher_logits = teacher(x)
                teacher_feat = None

        if hasattr(student, "forward_all"):
            logits1, logits2, logits3, aux = student.forward_all(x)
            student_logits = logits3
            student_feat = aux.get("cls_feature")
        else:
            student_logits = student(x)
            aux = {}
            student_feat = None
            logits1 = logits2 = None

        loss_ce = criterion(student_logits, y)
        if early_exit and logits1 is not None and logits2 is not None:
            loss_ce = loss_ce + alpha_ee * criterion(logits1, y) + beta_ee * criterion(logits2, y)

        soft_student = F.log_softmax(student_logits / temp, dim=1)
        soft_teacher = F.softmax(teacher_logits / temp, dim=1)
        loss_kd = kl_fn(soft_student, soft_teacher) * (temp * temp)

        loss = alpha * loss_ce + (1 - alpha) * loss_kd

        if lambda_feat > 0 and student_feat is not None and teacher_feat is not None:
            loss = loss + lambda_feat * F.mse_loss(student_feat, teacher_feat)

        if lambda_anatomy > 0 or lambda_sparse > 0:
            loss = loss + _compute_aux_losses(student, aux, lambda_anatomy, lambda_sparse, schedule)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = student_logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    train_loss = total_loss / max(len(loader), 1)
    train_acc = correct / max(total, 1)
    return train_loss, train_acc


def ensure_outdir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)


def init_log(out_dir: str):
    log_path = os.path.join(out_dir, "logs", "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "epoch", "train_loss", "train_acc", "val_acc", "val_loss"])
    return log_path


def append_log(log_path: str, epoch: int, train_loss: float, train_acc: float, val_acc: float, val_loss: float):
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), epoch,
            f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_acc:.6f}", f"{val_loss:.6f}"
        ])


def main():
    args = parse_args()
    set_seed(args.seed)

    label_map = parse_label_map(args.label_map)
    num_classes = len(label_map)

    device = get_device()

    train_csv = os.path.join(args.manifest_dir, "train.csv")
    val_csv = os.path.join(args.manifest_dir, "val.csv")
    test_csv = os.path.join(args.manifest_dir, "test.csv")

    # ---- choose dataset by input_type ----
    if args.input_type == "npy":
        train_ds = ADNINPY2DDataset(train_csv, label_map=label_map, data_root=args.data_root)
        val_ds   = ADNINPY2DDataset(val_csv,   label_map=label_map, data_root=args.data_root)
        test_ds  = ADNINPY2DDataset(test_csv,  label_map=label_map, data_root=args.data_root)
    else:
        # ---- slice selector config passed into NIfTI Dataset ----
        slice_cfg = {"slice_selector": args.slice_selector, "data_root": args.data_root}
        if args.slice_selector == "entropy":
            slice_cfg.update({
                "num_bins": 256,
                "entropy_topk": args.entropy_topk,
            })
        elif args.slice_selector == "fixed":
            if args.z_frac is not None:
                slice_cfg["z_frac"] = args.z_frac
            else:
                slice_cfg["z_index"] = args.z_index  # default 77 (middle)
        train_ds = Nii2DSliceDataset(train_csv, label_map=label_map, **slice_cfg)
        val_ds   = Nii2DSliceDataset(val_csv,   label_map=label_map, **slice_cfg)
        test_ds  = Nii2DSliceDataset(test_csv,  label_map=label_map, **slice_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_vit2d(
        num_classes=num_classes,
        thinning=args.thinning,
        thin_method=args.thin_method,
        debug_thin=args.debug_thin,
        enable_early_exit=args.early_exit,
        exit_blocks=tuple(args.exit_blocks),
        use_anatomical_prior=args.use_anatomical_prior,
        anatomical_prior_path=args.anatomical_prior_path,
        anatomical_prior_slice=args.anatomical_prior_slice,
        gumbel_tau=args.gumbel_tau if args.thin_method == "learnable" else 0.0,
        pretrained=not args.no_pretrained,
    ).to(device)

    teacher = None
    if args.teacher_ckpt:
        teacher = build_vit2d(
            num_classes=num_classes,
            thinning=args.teacher_thinning,
            thin_method=args.thin_method,
            debug_thin=False,
            enable_early_exit=args.teacher_early_exit,
            exit_blocks=tuple(args.exit_blocks),
            use_anatomical_prior=args.use_anatomical_prior,
            anatomical_prior_path=args.anatomical_prior_path,
            anatomical_prior_slice=args.anatomical_prior_slice,
            pretrained=not args.no_pretrained,
        ).to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Teacher loaded from {args.teacher_ckpt} (frozen)")

    # ---- inspect ViT depth ----
    depth = None
    if hasattr(model, "blocks"):
        depth = len(model.blocks)
    elif hasattr(model, "vit") and hasattr(model.vit, "blocks"):
        depth = len(model.vit.blocks)

    print(f"ViT depth (num blocks): {depth}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ensure_outdir(args.out_dir)
    log_path = init_log(args.out_dir)

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Train: {train_csv} (n={len(train_ds)})")
    print(f"Val:   {val_csv} (n={len(val_ds)})")
    print(f"Test:  {test_csv} (n={len(test_ds)})")
    print(f"Saving to: {args.out_dir}")
    if args.input_type == "nifti":
        sel_info = f"Slice selector: {args.slice_selector}"
        if args.slice_selector == "entropy":
            sel_info += f" | entropy_topk={args.entropy_topk}"
        elif args.slice_selector == "fixed":
            sel_info += f" | z_index={args.z_index} | z_frac={args.z_frac}"
        print(sel_info)
    else:
        print("Input: pre-extracted fixed-z 2D slices (.npy); slice selector not used.")

    print(f"Thinning: {args.thinning} | method: {args.thin_method}")
    print(
        f"Early-exit (model): {getattr(model, 'enable_early_exit', False)} "
        f"| alpha={args.alpha} beta={args.beta}"
    )
    if args.tau is not None:
        print(f"Early-exit eval tau: {args.tau}")
    if teacher is not None:
        print(f"Distillation: temp={args.distill_temp} alpha={args.distill_alpha}")

    use_distill = teacher is not None

    for ep in range(args.epochs):
        t0 = time.time()
        if use_distill:
            train_loss, train_acc = train_one_epoch_distill(
                model,
                teacher,
                train_loader,
                optimizer,
                criterion,
                device,
                temp=args.distill_temp,
                alpha=args.distill_alpha,
                scaler=scaler,
                early_exit=args.early_exit,
                alpha_ee=args.alpha,
                beta_ee=args.beta,
                lambda_feat=args.lambda_feat,
                lambda_anatomy=args.lambda_anatomy,
                lambda_sparse=args.lambda_sparse,
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                early_exit=args.early_exit,
                alpha=args.alpha,
                beta=args.beta,
                lambda_anatomy=args.lambda_anatomy,
                lambda_sparse=args.lambda_sparse,
            )

        if args.tau is None:
            val_acc, val_loss = evaluate(model, val_loader, device, criterion=criterion)
        else:
            val_acc, exit_counts, avg_exit, val_loss = evaluate(
                model, val_loader, device, tau=args.tau, tau_u=args.tau_u, criterion=criterion
            )
            print(f"[val early-exit] tau={args.tau} exit_counts={exit_counts} avg_exit={avg_exit:.3f}")

        dt = time.time() - t0
        print(f"[epoch {ep}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f} time={dt:.1f}s")
        append_log(log_path, ep, train_loss, train_acc, val_acc, val_loss)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

    # final test
    model.load_state_dict(torch.load(best_path, map_location=device))
    if args.tau is None:
        test_acc = evaluate(model, test_loader, device)
        print(f"Done. best_val_acc={best_val:.4f} test_acc={test_acc:.4f}")
    else:
        test_acc, exit_counts, avg_exit = evaluate(
            model, test_loader, device, tau=args.tau, tau_u=args.tau_u
        )
        print(f"Done. best_val_acc={best_val:.4f} test_acc={test_acc:.4f} (early-exit tau={args.tau})")
        print(f"Exit counts: {exit_counts} | Avg exit: {avg_exit:.3f}")

    print(f"Best checkpoint: {best_path}")
    print(f"Log: {log_path}")

    # save results.json for run_slice_selection.py / run_compare_thin_methods.py
    results = {
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "slice_selector": args.slice_selector,
        "z_index": args.z_index,
        "z_frac": args.z_frac,
        "thinning": args.thinning,
        "thin_method": args.thin_method,
        "gumbel_tau": args.gumbel_tau,
        "early_exit": args.early_exit,
        "tau": args.tau,
        "tau_u": args.tau_u,
        "use_anatomical_prior": args.use_anatomical_prior,
        "anatomical_prior_path": args.anatomical_prior_path,
        "anatomical_prior_slice": args.anatomical_prior_slice,
        "distill": args.teacher_ckpt is not None,
        "teacher_ckpt": args.teacher_ckpt,
        "distill_temp": args.distill_temp,
        "distill_alpha": args.distill_alpha,
        "lambda_feat": args.lambda_feat,
    }
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
