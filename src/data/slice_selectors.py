# src/data/slice_selectors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Dict, Any
import numpy as np


class SliceSelector(Protocol):
    """Protocol for slice selectors."""
    def select(
        self,
        volume: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> int:
        ...


@dataclass
class MiddleSliceSelector:
    """Select the middle slice along `axis` (deterministic baseline)."""
    axis: int = 2

    def select(self, volume: np.ndarray, *, mask=None, meta=None) -> int:
        n = int(volume.shape[self.axis])
        return n // 2


@dataclass
class EntropySliceSelector:
    """
    Select slice with highest entropy of intensities (within mask if provided).
    This is a generic heuristic; kept for comparisons / fallback.
    """
    axis: int = 2
    num_bins: int = 128
    min_mask_pixels: int = 1000
    eps: float = 1e-12

    def select(self, volume: np.ndarray, *, mask: Optional[np.ndarray] = None, meta=None) -> int:
        vol = np.moveaxis(volume, self.axis, -1)  # (..., S)
        msk = None
        if mask is not None:
            msk = np.moveaxis(mask, self.axis, -1)

        S = vol.shape[-1]
        ent = np.zeros((S,), dtype=np.float32)

        # robust intensity range
        v = vol[np.isfinite(vol)]
        if v.size == 0:
            return S // 2
        lo, hi = np.percentile(v, [1, 99])
        if hi <= lo:
            return S // 2

        for z in range(S):
            img = vol[..., z]
            if msk is not None:
                m = msk[..., z] > 0.5
                if int(m.sum()) < self.min_mask_pixels:
                    ent[z] = 0.0
                    continue
                img = img[m]

            img = img[np.isfinite(img)]
            if img.size < 32:
                ent[z] = 0.0
                continue

            img = np.clip(img, lo, hi)
            hist, _ = np.histogram(img, bins=self.num_bins, range=(lo, hi), density=True)
            p = hist.astype(np.float32)
            p = p / (p.sum() + self.eps)
            ent[z] = float(-(p * np.log(p + self.eps)).sum())

        best = int(np.flatnonzero(ent == ent.max())[0])  # deterministic tie-break
        return best


@dataclass
class FixedZSliceSelector:
    """
    Use a fixed z index or fraction for slice selection (anatomically meaningful on registered volumes).
    """
    axis: int = 2
    z_index: Optional[int] = None
    z_frac: Optional[float] = None

    def select(self, volume: np.ndarray, *, mask=None, meta=None) -> int:
        n = int(volume.shape[self.axis])
        if self.z_index is not None:
            return int(np.clip(self.z_index, 0, n - 1))
        if self.z_frac is not None:
            z = int(self.z_frac * n)
            return int(np.clip(z, 0, n - 1))
        return n // 2


def build_selector(name: str, **kwargs) -> SliceSelector:
    """
    Factory for selectors.
    name: 'middle' | 'entropy' | 'fixed'
    """
    name = name.lower().strip()
    if name in ("middle", "mid", "center", "centre"):
        return MiddleSliceSelector(**kwargs)
    if name in ("entropy",):
        return EntropySliceSelector(**kwargs)
    if name in ("fixed",):
        return FixedZSliceSelector(**kwargs)
    raise ValueError(f"Unknown selector name: {name}")
