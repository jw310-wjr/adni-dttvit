import argparse
import os, sys

# Add project root to PYTHONPATH so "import src...." works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import json
import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset_nii2d import (
    Nii2DSliceDataset,
    in_chans_for_slice_stack_mode,
)
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
        default=144,
        help="Fixed z slice index (used when slice_selector=fixed); align with slice_selection on MNI volumes",
    )
    p.add_argument(
        "--z_frac",
        type=float,
        default=None,
        help="Fixed z as fraction of depth (0-1, used when slice_selector=fixed if z_index not set)"
    )
    p.add_argument(
        "--slice_stack_mode",
        default="single",
        choices=["single", "stack3", "stack5"],
        help="2.5D: single=one axial slice (C=1); stack3=[z-1,z,z+1]; stack5=[z-2..z+2] (clamped). "
        "Compare centers with e.g. --z_index 77|115|127|144 under the same training recipe.",
    )

    # training
    p.add_argument("--epochs", type=int, default=50, help="Default 50, aligned with SLURM baseline config")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Conservative default for pretrained ViT fine-tune (try 1e-4–2.5e-4)",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW L2; low WD (e.g. 0.005–0.02) with pretrained ViT + CE class weights",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.02,
        help="Mild CE smoothing (0=off; keep small when WD is already low)",
    )
    p.add_argument(
        "--ce_class_weights",
        default="auto",
        choices=["auto", "none"],
        help="auto: sklearn-style inverse frequency from train.csv (reduces majority-class collapse)",
    )
    p.add_argument(
        "--best_metric",
        default="balanced_acc",
        choices=["acc", "balanced_acc", "loss", "hmean"],
        help="hmean: harmonic mean(val_acc, val_balanced)—high acc without single-class collapse",
    )
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many epochs without best_metric improvement (0=run all epochs)",
    )
    p.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.005,
        help="Minimum improvement in best_metric to reset patience (acc/balanced_acc in [0,1])",
    )
    p.add_argument(
        "--lr_scheduler",
        default="cosine",
        choices=["none", "cosine"],
        help="cosine: CosineAnnealingLR to eta_min (softens late-epoch overfitting)",
    )
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.05,
        help="eta_min = lr * lr_min_ratio when lr_scheduler=cosine",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Linear LR warmup (0=off). Used before cosine decay when lr_scheduler=cosine.",
    )
    p.add_argument(
        "--train_sampler",
        default="shuffle",
        choices=["shuffle", "balanced"],
        help="shuffle: default. balanced: WeightedRandomSampler—do not combine with --ce_class_weights auto "
        "(double imbalance correction); use one or the other.",
    )
    p.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.0,
        help="timm stochastic depth (0=off; try 0.08–0.12 on baseline to curb overfit)",
    )
    p.add_argument(
        "--drop_rate",
        type=float,
        default=0.0,
        help="timm dropout on classifier head (often 0; small e.g. 0.1 if needed)",
    )
    p.add_argument(
        "--train_aug",
        action="store_true",
        help="Training-only augment: optional translate (default off) + noise. No flip/rotation (MNI + L/R anatomy).",
    )
    p.add_argument(
        "--aug_translate",
        type=float,
        default=0.0,
        help="Max shift as fraction of H,W when --train_aug (default 0: registered MNI space; avoid reintroducing misalignment)",
    )
    p.add_argument("--aug_noise", type=float, default=0.015, help="Gaussian noise std on normalized slice")
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


def train_class_counts_from_csv(train_csv: str, label_map: dict, num_classes: int) -> Tuple[List[int], int]:
    label_col = None
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for c in ("label", "Group", "group"):
            if c in fields:
                label_col = c
                break
        if label_col is None:
            raise ValueError(f"No label column (label/Group/group) in {train_csv}")
        counts = [0] * num_classes
        n = 0
        for row in reader:
            y = int(label_map[str(row[label_col].strip())])
            counts[y] += 1
            n += 1
    if n == 0:
        raise ValueError(f"Empty training CSV: {train_csv}")
    if any(c == 0 for c in counts):
        raise ValueError(f"Missing some classes in {train_csv}: counts={counts}")
    return counts, n


def ce_class_weight_tensor(
    train_csv: str,
    label_map: dict,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    counts, n = train_class_counts_from_csv(train_csv, label_map, num_classes)
    w = [n / (num_classes * c) for c in counts]
    t = torch.tensor(w, dtype=torch.float32, device=device)
    return t, counts


def per_sample_inv_freq_weights(train_csv: str, label_map: dict, num_classes: int) -> List[float]:
    """One weight per training row (CSV order = Dataset idx): 1 / count[class]."""
    label_col = None
    ys: List[int] = []
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for c in ("label", "Group", "group"):
            if c in fields:
                label_col = c
                break
        if label_col is None:
            raise ValueError(f"No label column in {train_csv}")
        for row in reader:
            y = int(label_map[str(row[label_col].strip())])
            ys.append(y)
    counts = [0] * num_classes
    for y in ys:
        counts[y] += 1
    if any(c == 0 for c in counts):
        raise ValueError(f"Missing class in {train_csv}: counts={counts}")
    return [1.0 / counts[y] for y in ys]


def set_epoch_lr(optimizer: optim.Optimizer, epoch: int, args) -> float:
    """Linear warmup then cosine decay to eta_min (when lr_scheduler=cosine)."""
    base = args.lr
    eta = max(1e-8, float(base * args.lr_min_ratio))
    if args.lr_scheduler == "none":
        lr = base
    else:
        w = max(0, int(args.warmup_epochs))
        if w > 0 and epoch < w:
            lr = base * float(epoch + 1) / float(w)
        else:
            denom = max(1, int(args.epochs) - w)
            t = (epoch - w) / float(denom)
            t = min(1.0, max(0.0, t))
            lr = eta + (base - eta) * 0.5 * (1.0 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return float(lr)


def val_hmean_acc_balanced(val_acc: float, val_balanced: float, eps: float = 1e-8) -> float:
    a, b = float(val_acc), float(val_balanced)
    if a <= eps or b <= eps:
        return 0.0
    return 2.0 * a * b / (a + b)


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
def evaluate(
    model,
    loader,
    device,
    tau=None,
    tau_u=None,
    criterion=None,
    num_classes: int = 2,
):
    """
    If tau is set: uncertainty-guided early exit (Proposal §3.6) with tau, tau_u.
    If criterion is set: also returns val_loss for logging.
    When tau is None: returns balanced_acc (mean per-class recall) over the split.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    exit_counts = {1: 0, 2: 0, 3: 0} if tau is not None else None
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
        for c in range(num_classes):
            m = y == c
            class_total[c] += int(m.sum().item())
            class_correct[c] += int(((pred == y) & m).sum().item())
        if criterion is not None:
            total_loss += criterion(logits, y).item()

    acc = correct / max(total, 1)
    if sum(class_total) > 0:
        ncls = sum(1 for c in range(num_classes) if class_total[c] > 0)
        balanced_acc = float(
            sum(class_correct[c] / class_total[c] for c in range(num_classes) if class_total[c] > 0)
            / max(1, ncls)
        )
    else:
        balanced_acc = 0.0
    val_loss = total_loss / max(len(loader), 1) if criterion is not None else None
    if tau is not None:
        avg_exit = (1 * exit_counts[1] + 2 * exit_counts[2] + 3 * exit_counts[3]) / max(total, 1)
        if criterion is not None:
            return acc, exit_counts, avg_exit, val_loss, balanced_acc
        return acc, exit_counts, avg_exit, balanced_acc
    if criterion is not None:
        return acc, val_loss, balanced_acc
    return acc, balanced_acc


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


def _balanced_acc_from_counts(class_correct: List[int], class_total: List[int]) -> float:
    ncls = sum(1 for c in range(len(class_total)) if class_total[c] > 0)
    if ncls == 0:
        return 0.0
    return float(
        sum(class_correct[c] / class_total[c] for c in range(len(class_total)) if class_total[c] > 0) / ncls
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    num_classes: int,
    scaler=None,
    early_exit=False,
    alpha=0.3,
    beta=0.3,
    lambda_anatomy=0.0,
    lambda_sparse=0.0,
):
    """
    If early_exit: multi-head CE + optional L_anatomy, L_sparse (Proposal).
    Returns (train_loss, train_acc, train_balanced).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
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
        for c in range(num_classes):
            m = y == c
            class_total[c] += int(m.sum().item())
            class_correct[c] += int(((pred == y) & m).sum().item())

    train_loss = total_loss / max(len(loader), 1)
    train_acc = correct / max(total, 1)
    train_balanced = _balanced_acc_from_counts(class_correct, class_total)
    return train_loss, train_acc, train_balanced


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
    num_classes: int = 2,
):
    """
    Proposal §3.7: L = L_cls + λ1*L_KD + λ2*L_feat + λ3*L_sparse + λ4*L_anatomy
    L_feat = ||f_student - f_teacher||²
    Returns (train_loss, train_acc, train_balanced).
    """
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
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
        for c in range(num_classes):
            m = y == c
            class_total[c] += int(m.sum().item())
            class_correct[c] += int(((pred == y) & m).sum().item())

    train_loss = total_loss / max(len(loader), 1)
    train_acc = correct / max(total, 1)
    train_balanced = _balanced_acc_from_counts(class_correct, class_total)
    return train_loss, train_acc, train_balanced


def ensure_outdir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)


def init_log(out_dir: str):
    log_path = os.path.join(out_dir, "logs", "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "time",
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_balanced",
                    "val_acc",
                    "val_balanced",
                    "val_loss",
                ]
            )
    return log_path


def append_log(
    log_path: str,
    epoch: int,
    train_loss: float,
    train_acc: float,
    train_balanced: float,
    val_acc: float,
    val_balanced: float,
    val_loss: float,
):
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), epoch,
            f"{train_loss:.6f}", f"{train_acc:.6f}", f"{train_balanced:.6f}",
            f"{val_acc:.6f}", f"{val_balanced:.6f}", f"{val_loss:.6f}",
        ])


def main():
    args = parse_args()
    if args.ce_class_weights == "auto" and args.train_sampler == "balanced":
        raise ValueError(
            "Do not combine --ce_class_weights auto with --train_sampler balanced "
            "(double imbalance correction). Use shuffle + auto, or balanced + --ce_class_weights none."
        )
    set_seed(args.seed)

    label_map = parse_label_map(args.label_map)
    num_classes = len(label_map)

    device = get_device()

    train_csv = os.path.join(args.manifest_dir, "train.csv")
    val_csv = os.path.join(args.manifest_dir, "val.csv")
    test_csv = os.path.join(args.manifest_dir, "test.csv")

    # ---- choose dataset by input_type ----
    vit_in_chans = 1
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
        slice_cfg["slice_stack_mode"] = args.slice_stack_mode
        train_ds = Nii2DSliceDataset(
            train_csv,
            label_map=label_map,
            augment=args.train_aug,
            aug_translate=args.aug_translate,
            aug_noise=args.aug_noise,
            **slice_cfg,
        )
        val_ds = Nii2DSliceDataset(val_csv, label_map=label_map, **slice_cfg)
        test_ds = Nii2DSliceDataset(test_csv, label_map=label_map, **slice_cfg)
        vit_in_chans = in_chans_for_slice_stack_mode(args.slice_stack_mode)

    train_sampler = None
    if args.train_sampler == "balanced":
        psw = per_sample_inv_freq_weights(train_csv, label_map, num_classes)
        if len(psw) != len(train_ds):
            raise RuntimeError(
                f"Sampler length {len(psw)} != train_ds {len(train_ds)} (check train.csv rows)"
            )
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(psw, dtype=torch.double),
            num_samples=len(psw),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Train sampler: balanced (WeightedRandomSampler, n={len(psw)})")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print("Train sampler: shuffle")
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
        in_chans=vit_in_chans,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
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
            in_chans=vit_in_chans,
            drop_rate=0.0,
            drop_path_rate=0.0,
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

    ce_weight: Optional[torch.Tensor] = None
    train_class_counts: Optional[List[int]] = None
    if args.ce_class_weights == "auto":
        ce_weight, train_class_counts = ce_class_weight_tensor(
            train_csv, label_map, num_classes, device
        )
        print(
            f"CE class weights (auto, train counts): {train_class_counts} -> "
            f"{ce_weight.detach().cpu().tolist()}"
        )
    criterion = nn.CrossEntropyLoss(
        weight=ce_weight,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ensure_outdir(args.out_dir)
    log_path = init_log(args.out_dir)

    best_track = float("-inf")  # maximize; for loss metric use -val_loss
    best_path = os.path.join(args.out_dir, "best.pt")
    patience_ctr = 0
    best_val_acc = -1.0
    best_val_balanced = -1.0
    best_val_loss = float("inf")

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
        sel_info += f" | slice_stack_mode={args.slice_stack_mode} (in_chans={vit_in_chans})"
        print(sel_info)
    else:
        print("Input: pre-extracted fixed-z 2D slices (.npy); slice selector not used.")

    print(f"Thinning: {args.thinning} | method: {args.thin_method}")
    print(
        f"Early-exit (model): {getattr(model, 'enable_early_exit', False)} "
        f"| alpha={args.alpha} beta={args.beta}"
    )
    print(
        f"Regularization: weight_decay={args.weight_decay} label_smoothing={args.label_smoothing} "
        f"| drop_path_rate={args.drop_path_rate} drop_rate={args.drop_rate} "
        f"| ce_class_weights={args.ce_class_weights} | best_metric={args.best_metric} "
        f"| early_stop: patience={args.early_stopping_patience} min_delta={args.early_stopping_min_delta} | "
        f"lr_scheduler={args.lr_scheduler} warmup_epochs={args.warmup_epochs} "
        f"(eta_min={args.lr * args.lr_min_ratio:.2e}) | "
        f"train_sampler={args.train_sampler} | "
        f"train_aug={args.train_aug} (translate={args.aug_translate}, "
        f"noise={args.aug_noise}; no L-R flip, no rotation)"
    )
    if args.tau is not None:
        print(f"Early-exit eval tau: {args.tau}")
    if teacher is not None:
        print(f"Distillation: temp={args.distill_temp} alpha={args.distill_alpha}")

    use_distill = teacher is not None

    for ep in range(args.epochs):
        t0 = time.time()
        set_epoch_lr(optimizer, ep, args)
        if use_distill:
            train_loss, train_acc, train_balanced = train_one_epoch_distill(
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
                num_classes=num_classes,
            )
        else:
            train_loss, train_acc, train_balanced = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                num_classes,
                scaler=scaler,
                early_exit=args.early_exit,
                alpha=args.alpha,
                beta=args.beta,
                lambda_anatomy=args.lambda_anatomy,
                lambda_sparse=args.lambda_sparse,
            )

        if args.tau is None:
            val_acc, val_loss, val_balanced = evaluate(
                model, val_loader, device, criterion=criterion, num_classes=num_classes
            )
        else:
            val_acc, exit_counts, avg_exit, val_loss, val_balanced = evaluate(
                model,
                val_loader,
                device,
                tau=args.tau,
                tau_u=args.tau_u,
                criterion=criterion,
                num_classes=num_classes,
            )
            print(f"[val early-exit] tau={args.tau} exit_counts={exit_counts} avg_exit={avg_exit:.3f}")

        if args.best_metric == "acc":
            track_now = val_acc
        elif args.best_metric == "balanced_acc":
            track_now = val_balanced
        elif args.best_metric == "hmean":
            track_now = val_hmean_acc_balanced(val_acc, val_balanced)
        else:
            track_now = -float(val_loss)

        md = args.early_stopping_min_delta
        if best_track == float("-inf"):
            improved = True
        else:
            improved = track_now > best_track + md
        if improved:
            best_track = track_now
            best_val_acc = val_acc
            best_val_balanced = val_balanced
            best_val_loss = float(val_loss)
            torch.save(model.state_dict(), best_path)
            patience_ctr = 0
        elif args.early_stopping_patience > 0:
            patience_ctr += 1

        dt = time.time() - t0
        lr_cur = optimizer.param_groups[0]["lr"]
        val_hm = val_hmean_acc_balanced(val_acc, val_balanced)
        if args.tau is None:
            print(
                f"[epoch {ep}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_balanced={train_balanced:.4f} "
                f"val_acc={val_acc:.4f} val_balanced={val_balanced:.4f} val_hmean={val_hm:.4f} "
                f"val_loss={val_loss:.4f} lr={lr_cur:.2e} time={dt:.1f}s"
            )
        else:
            print(
                f"[epoch {ep}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_balanced={train_balanced:.4f} "
                f"val_acc={val_acc:.4f} val_balanced={val_balanced:.4f} val_hmean={val_hm:.4f} "
                f"val_loss={val_loss:.4f} lr={lr_cur:.2e} time={dt:.1f}s"
            )
        append_log(
            log_path, ep, train_loss, train_acc, train_balanced, val_acc, val_balanced, val_loss
        )

        if args.early_stopping_patience > 0 and patience_ctr >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {ep} (no {args.best_metric} improvement for "
                f"{args.early_stopping_patience} epochs)"
            )
            break

    # final test
    model.load_state_dict(torch.load(best_path, map_location=device))
    if args.tau is None:
        test_acc, test_balanced = evaluate(model, test_loader, device, num_classes=num_classes)
        print(
            f"Done. best.checkpoint: val_acc={best_val_acc:.4f} val_balanced={best_val_balanced:.4f} "
            f"val_loss={best_val_loss:.4f} | test_acc={test_acc:.4f} test_balanced={test_balanced:.4f}"
        )
    else:
        test_acc, exit_counts, avg_exit, test_balanced = evaluate(
            model,
            test_loader,
            device,
            tau=args.tau,
            tau_u=args.tau_u,
            num_classes=num_classes,
        )
        print(
            f"Done. best val_acc={best_val_acc:.4f} val_balanced={best_val_balanced:.4f} "
            f"test_acc={test_acc:.4f} test_balanced={test_balanced:.4f} (early-exit tau={args.tau})"
        )
        print(f"Exit counts: {exit_counts} | Avg exit: {avg_exit:.3f}")

    print(f"Best checkpoint: {best_path}")
    print(f"Log: {log_path}")

    # save results.json for run_slice_selection.py / run_compare_thin_methods.py
    results = {
        "best_val_acc": best_val_acc,
        "best_val_balanced": best_val_balanced,
        "best_val_hmean": val_hmean_acc_balanced(best_val_acc, best_val_balanced),
        "best_val_loss": best_val_loss,
        "best_metric": args.best_metric,
        "test_acc": test_acc,
        "test_balanced": test_balanced,
        "slice_selector": args.slice_selector,
        "z_index": args.z_index,
        "z_frac": args.z_frac,
        "slice_stack_mode": args.slice_stack_mode,
        "in_chans": vit_in_chans,
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
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "label_smoothing": args.label_smoothing,
        "ce_class_weights": args.ce_class_weights,
        "train_class_counts": train_class_counts,
        "ce_weight": ce_weight.detach().cpu().tolist() if ce_weight is not None else None,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "lr_scheduler": args.lr_scheduler,
        "lr_min_ratio": args.lr_min_ratio,
        "warmup_epochs": args.warmup_epochs,
        "train_sampler": args.train_sampler,
        "drop_path_rate": args.drop_path_rate,
        "drop_rate": args.drop_rate,
        "train_aug": args.train_aug,
        "aug_translate": args.aug_translate,
        "aug_noise": args.aug_noise,
    }
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
