# src/data/dataset_nii2d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .slice_selectors import SliceSelector, MiddleSliceSelector, build_selector


# -------------------------
# Utilities
# -------------------------

def normalize_2d(
    img: np.ndarray,
    clip_percentiles=(1.0, 99.0),
    eps: float = 1e-6
) -> np.ndarray:
    img = img.astype(np.float32)
    mask = np.isfinite(img)
    if mask.sum() < 10:
        return np.zeros_like(img, dtype=np.float32)

    v = img[mask]
    lo, hi = np.percentile(v, clip_percentiles)
    if hi <= lo:
        mu, sd = v.mean(), v.std() + eps
        return (img - mu) / sd

    img = np.clip(img, lo, hi)
    mu, sd = img[mask].mean(), img[mask].std() + eps
    img = (img - mu) / sd
    img[~mask] = 0.0
    return img


def apply_mri_slice_augment(
    x: torch.Tensor,
    *,
    deg: float = 12.0,
    translate_frac: float = 0.04,
    noise_std: float = 0.03,
) -> torch.Tensor:
    """
    Light 2D augment on normalized slice tensor [1, H, W] (train only).
    Uses torchvision if available; otherwise only Gaussian noise when noise_std > 0.

    No left-right flip: axial brain MRI is L/R asymmetric in MNI/orientation; mirroring
    swaps hemispheres and must not be used for anatomy-sensitive AD vs CN.
    """
    try:
        import torchvision.transforms.functional as TF
    except ImportError:
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return x

    if deg > 0.0:
        angle = float(torch.empty(1, device=x.device).uniform_(-deg, deg))
        x = TF.rotate(
            x,
            angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=[0.0],
        )
    if translate_frac > 0.0:
        _, h, w = x.shape
        max_tx = max(1, int(translate_frac * w))
        max_ty = max(1, int(translate_frac * h))
        tx = int(torch.randint(-max_tx, max_tx + 1, (1,), device=x.device).item())
        ty = int(torch.randint(-max_ty, max_ty + 1, (1,), device=x.device).item())
        if tx != 0 or ty != 0:
            x = TF.affine(
                x,
                angle=0.0,
                translate=[tx, ty],
                scale=1.0,
                shear=[0.0],
                fill=[0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std
    return x


def to_3ch_resize(img: np.ndarray, out_size: int = 224) -> torch.Tensor:
    """Resize to out_size; returns (1,H,W) for ViT in_chans=1 (grayscale MRI)."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)  # (1,H,W) - single channel for in_chans=1


# -------------------------
# Dataset
# -------------------------

@dataclass
class Nii2DConfig:
    image_col: str = "nifti_path"
    label_col: str = "label"
    subject_id_col: str = "subject"
    axis: int = 2
    out_size: int = 224
    return_meta: bool = False


def _resolve_col(df: pd.DataFrame, preferred: str, fallbacks: list) -> str:
    if preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    raise ValueError(f"Missing column '{preferred}' or any of {fallbacks} in CSV")


class Nii2DSliceDataset(Dataset):
    """
    3D NIfTI -> single 2D slice Dataset.
    Uses selector (middle / entropy / fixed) to pick slice.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        selector: Optional[SliceSelector] = None,
        cfg: Optional[Nii2DConfig] = None,
        label_map: Optional[Dict[str, int]] = None,
        data_root: str = ".",
        slice_selector: Optional[str] = None,
        augment: bool = False,
        aug_deg: float = 12.0,
        aug_translate: float = 0.04,
        aug_noise: float = 0.03,
        **selector_kwargs,
    ):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg or Nii2DConfig()
        self.data_root = data_root
        self.label_map = label_map
        self.augment = augment
        self.aug_deg = aug_deg
        self.aug_translate = aug_translate
        self.aug_noise = aug_noise

        # column fallback for common manifest formats (Group/label, Subject/subject)
        self.cfg.image_col = _resolve_col(self.df, self.cfg.image_col, ["nifti_path", "path"])
        self.cfg.label_col = _resolve_col(self.df, self.cfg.label_col, ["Group", "group", "label"])
        self.cfg.subject_id_col = _resolve_col(self.df, self.cfg.subject_id_col, ["Subject", "subject"])

        if selector is not None:
            self.selector = selector
        elif slice_selector is not None:
            self.selector = build_selector(slice_selector, axis=self.cfg.axis, **selector_kwargs)
        else:
            self.selector = MiddleSliceSelector(axis=self.cfg.axis)

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, path: str) -> str:
        p = str(path)
        if not os.path.isabs(p):
            p = os.path.join(self.data_root, p)
        return p

    def _load_volume(self, path: str) -> np.ndarray:
        p = self._resolve_path(path)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        vol = nib.load(p).get_fdata(dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI, got {vol.shape}")
        return vol

    def _get_label(self, row: pd.Series) -> int:
        y = row[self.cfg.label_col]
        if self.label_map is not None:
            return int(self.label_map[str(y)])
        return int(y)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        vol = self._load_volume(row[self.cfg.image_col].strip())
        z = self.selector.select(vol)

        vol = np.moveaxis(vol, self.cfg.axis, -1)  # (H,W,S)
        img2d = vol[..., z]

        img2d = normalize_2d(img2d)
        x = to_3ch_resize(img2d, self.cfg.out_size)
        if self.augment:
            x = apply_mri_slice_augment(
                x,
                deg=self.aug_deg,
                translate_frac=self.aug_translate,
                noise_std=self.aug_noise,
            )
        y = self._get_label(row)

        if self.cfg.return_meta:
            meta = {
                "subject": row[self.cfg.subject_id_col],
                "slice_idx": z,
                "path": row[self.cfg.image_col],
            }
            return x, y, meta

        return x, y
# Backward-compatible alias for older training code
ADNIViT2DDataset = Nii2DSliceDataset
