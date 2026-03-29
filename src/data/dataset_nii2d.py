# src/data/dataset_nii2d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Sequence, List
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .slice_selectors import SliceSelector, MiddleSliceSelector, build_selector


# -------------------------
# 2.5D multi-slice (axial stack as channels)
# -------------------------

SLICE_STACK_MODES = ("single", "stack3", "stack5")

_SLICE_STACK_OFFSETS: Dict[str, Tuple[int, ...]] = {
    "single": (0,),
    "stack3": (-1, 0, 1),
    "stack5": (-2, -1, 0, 1, 2),
}


def slice_stack_offsets(mode: str) -> Tuple[int, ...]:
    if mode not in _SLICE_STACK_OFFSETS:
        raise ValueError(
            f"Unknown slice_stack_mode={mode!r}; expected one of {SLICE_STACK_MODES}"
        )
    return _SLICE_STACK_OFFSETS[mode]


def in_chans_for_slice_stack_mode(mode: str) -> int:
    return len(slice_stack_offsets(mode))


def get_multi_slices(
    volume: np.ndarray,
    center_z: int,
    offsets: Sequence[int],
    *,
    axis: int = 2,
) -> np.ndarray:
    """
    Stack neighboring axial slices as channels (2.5D input).

    Args:
        volume: 3D array (any layout); slice dimension is ``axis`` (default 2).
        center_z: Index along ``axis`` (same space as ``FixedZSliceSelector`` / entropy).
        offsets: e.g. (-1, 0, 1) for [z-1, z, z+1]. Each index is clamped to [0, D-1].

    Returns:
        (C, H, W) float32
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape}")
    depth = int(volume.shape[axis])
    moved = np.moveaxis(volume, axis, -1)
    slices: List[np.ndarray] = []
    for off in offsets:
        zi = int(min(max(center_z + int(off), 0), depth - 1))
        slices.append(moved[..., zi])
    return np.stack(slices, axis=0).astype(np.float32)


def resize_stack_chw(chw: np.ndarray, out_size: int) -> torch.Tensor:
    """Bilinear resize (C,H,W) -> (C,out_size,out_size)."""
    t = torch.from_numpy(chw).float().unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)


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
    translate_frac: float = 0.0,
    noise_std: float = 0.03,
) -> torch.Tensor:
    """
    Light 2D augment on normalized tensor [C, H, W] (train only); same geometry/noise for all channels.
    Uses torchvision if available; otherwise only Gaussian noise when noise_std > 0.

    No left-right flip (L/R anatomy). No rotation: axial brain slices are registered to a fixed
    orientation; rotating breaks correspondence with left/right and standard atlas alignment.
    """
    try:
        import torchvision.transforms.functional as TF
    except ImportError:
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return x

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
    """Resize one (H,W) slice to out_size; returns (1,H,W) for single-slice ViT."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)  # (1,H,W)


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
    slice_stack_mode: str = "single"


def _resolve_col(df: pd.DataFrame, preferred: str, fallbacks: list) -> str:
    if preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    raise ValueError(f"Missing column '{preferred}' or any of {fallbacks} in CSV")


class Nii2DSliceDataset(Dataset):
    """
    3D NIfTI -> 2D (single-slice) or 2.5D multi-slice stack (channels).
    Uses selector (middle / entropy / fixed) to pick the **center** slice index ``z``;
    neighbors are ``z + offset`` with indices clamped to the volume depth.
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
        slice_stack_mode: Optional[str] = None,
        augment: bool = False,
        aug_translate: float = 0.0,
        aug_noise: float = 0.03,
        **selector_kwargs,
    ):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg or Nii2DConfig()
        if slice_stack_mode is not None:
            self.cfg.slice_stack_mode = slice_stack_mode
        if self.cfg.slice_stack_mode not in _SLICE_STACK_OFFSETS:
            raise ValueError(
                f"Invalid slice_stack_mode={self.cfg.slice_stack_mode!r}; "
                f"expected one of {SLICE_STACK_MODES}"
            )
        self._slice_offsets = slice_stack_offsets(self.cfg.slice_stack_mode)
        self.data_root = data_root
        self.label_map = label_map
        self.augment = augment
        self.aug_translate = aug_translate
        self.aug_noise = aug_noise

        # column fallback for common manifest formats (Group/label, Subject/subject)
        self.cfg.image_col = _resolve_col(self.df, self.cfg.image_col, ["nifti_path", "path"])
        self.cfg.label_col = _resolve_col(self.df, self.cfg.label_col, ["Group", "group", "label"])
        self.cfg.subject_id_col = _resolve_col(self.df, self.cfg.subject_id_col, ["Subject", "subject"])

        if selector is not None:
            self.selector = selector
        elif slice_selector is not None:
            selector_kwargs.pop("aug_deg", None)  # legacy; rotation removed for brain MRI
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

    @property
    def in_chans(self) -> int:
        return len(self._slice_offsets)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        vol = self._load_volume(row[self.cfg.image_col].strip())
        z = self.selector.select(vol)

        depth = int(vol.shape[self.cfg.axis])
        chw = get_multi_slices(vol, z, self._slice_offsets, axis=self.cfg.axis)

        z_indices = [
            int(min(max(z + int(off), 0), depth - 1)) for off in self._slice_offsets
        ]

        chw_norm = np.stack([normalize_2d(chw[c]) for c in range(chw.shape[0])], axis=0)
        x = resize_stack_chw(chw_norm, self.cfg.out_size)
        if self.augment:
            x = apply_mri_slice_augment(
                x,
                translate_frac=self.aug_translate,
                noise_std=self.aug_noise,
            )
        y = self._get_label(row)

        if self.cfg.return_meta:
            meta = {
                "subject": row[self.cfg.subject_id_col],
                "slice_idx": z,
                "slice_stack_mode": self.cfg.slice_stack_mode,
                "slice_z_indices": z_indices,
                "path": row[self.cfg.image_col],
            }
            return x, y, meta

        return x, y
# Backward-compatible alias for older training code
ADNIViT2DDataset = Nii2DSliceDataset
