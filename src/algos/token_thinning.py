# src/algos/token_thinning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class ThinningSchedule:
    """
    A simple schedule mapping "after which block" -> "keep ratio".

    Interpretation:
      After block i, if i is in the dict, thin patch tokens to that keep ratio.
      CLS token is kept by default.
    """
    keep_ratio_by_block: Dict[int, float]

    def ratio_after_block(self, block_idx: int) -> Optional[float]:
        return self.keep_ratio_by_block.get(block_idx, None)


def _validate_x(x: torch.Tensor) -> None:
    if x.dim() != 3:
        raise ValueError(f"Expected x to be 3D [B, T, C], got shape={tuple(x.shape)}")
    if x.size(1) < 2:
        raise ValueError(f"Expected at least 2 tokens (CLS+patches). Got T={x.size(1)}")


def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    scores: [B, N]
    returns idx: [B, k] (unordered)
    """
    return torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices


def _split_cls_patches(x: torch.Tensor, keep_cls: bool) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    if keep_cls:
        _validate_x(x)
        cls = x[:, :1, :]      # [B,1,C]
        toks = x[:, 1:, :]     # [B,N,C]
        return cls, toks
    else:
        return None, x         # [B,N,C]


def _gather_tokens(toks: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    toks: [B, N, C]
    idx:  [B, k]
    return: [B, k, C]
    """
    B, _, C = toks.shape
    k = idx.size(1)
    idx_exp = idx.unsqueeze(-1).expand(B, k, C)   # [B,k,C]
    return torch.gather(toks, dim=1, index=idx_exp)


def _compute_k(N: int, keep_ratio: float) -> int:
    # k at least 1 to avoid empty token set
    return max(1, int(round(N * keep_ratio)))


# -------------------------
# v1: L2-norm thinning
# -------------------------
def thin_tokens_l2(
    x: torch.Tensor,
    keep_ratio: float,
    keep_cls: bool = True,
) -> torch.Tensor:
    """
    Token thinning by L2 norm of token embeddings.

    Args:
      x: [B, 1+N, C] if keep_cls else [B, N, C]
      keep_ratio: fraction of patch tokens to keep in (0, 1]
      keep_cls: keep the first token as CLS

    Returns:
      x_thin: [B, 1+k, C] (if keep_cls) or [B, k, C]
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}")
    if keep_ratio >= 1.0:
        return x

    cls, toks = _split_cls_patches(x, keep_cls=keep_cls)
    B, N, _ = toks.shape
    k = _compute_k(N, keep_ratio)
    if k >= N:
        return x

    scores = torch.norm(toks, dim=-1)  # [B,N]
    idx = _topk_indices(scores, k)     # [B,k]
    toks_kept = _gather_tokens(toks, idx)

    return torch.cat([cls, toks_kept], dim=1) if keep_cls else toks_kept


def thin_tokens_random(
    x: torch.Tensor,
    keep_ratio: float,
    keep_cls: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Random token thinning (useful for ablations / sanity checks).
    Deterministic if you pass a torch.Generator with a fixed seed.
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}")
    if keep_ratio >= 1.0:
        return x

    cls, toks = _split_cls_patches(x, keep_cls=keep_cls)
    B, N, C = toks.shape
    k = _compute_k(N, keep_ratio)
    if k >= N:
        return x

    idx_list = []
    for _ in range(B):
        perm = torch.randperm(N, device=toks.device, generator=generator)
        idx_list.append(perm[:k])
    idx = torch.stack(idx_list, dim=0)  # [B,k]

    toks_kept = _gather_tokens(toks, idx)
    return torch.cat([cls, toks_kept], dim=1) if keep_cls else toks_kept


# -------------------------
# method: attention-based thinning (CLS->token)
# -------------------------
def thin_tokens_attn_cls(
    x: torch.Tensor,
    attn: torch.Tensor,
    keep_ratio: float,
    keep_cls: bool = True,
) -> torch.Tensor:
    """
    Token thinning by CLS->token attention importance (method Eq.(1)).

    Expected attn shapes (any of these):
      - [B, H, T, T]  (preferred)
      - [H, B, T, T]
      - [B, T, T]    (single-head or already-averaged)
      - [T, T]       (shared; will be broadcast)

    Where T = 1 + N (CLS + patch tokens).

    Importance score for patch token i:
      score_i = mean_h attn[h, CLS, i]   (CLS index = 0)

    Notes:
      - We always preserve CLS token in output when keep_cls=True.
      - We rank only patch tokens (indices 1..T-1).
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}")
    if keep_ratio >= 1.0:
        return x

    _validate_x(x)
    B, T, _ = x.shape
    N = T - 1
    k = _compute_k(N, keep_ratio)
    if k >= N:
        return x

    # ---- normalize attn shape to [B, H, T, T] or [B, 1, T, T] ----
    if attn.dim() == 4:
        # could be [B,H,T,T] or [H,B,T,T]
        if attn.size(0) == B:
            attn_bhtt = attn
        elif attn.size(1) == B:
            attn_bhtt = attn.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError(f"attn 4D but cannot match batch B={B}, got shape={tuple(attn.shape)}")
    elif attn.dim() == 3:
        # [B,T,T] -> add head dim
        if attn.size(0) != B:
            raise ValueError(f"attn 3D expects first dim B={B}, got shape={tuple(attn.shape)}")
        attn_bhtt = attn.unsqueeze(1)  # [B,1,T,T]
    elif attn.dim() == 2:
        # [T,T] -> broadcast to batch and add head
        if attn.size(0) != T or attn.size(1) != T:
            raise ValueError(f"attn 2D expects [T,T] with T={T}, got shape={tuple(attn.shape)}")
        attn_bhtt = attn.unsqueeze(0).unsqueeze(1).expand(B, 1, T, T).contiguous()
    else:
        raise ValueError(f"Unsupported attn shape: {tuple(attn.shape)}")

    if attn_bhtt.size(2) != T or attn_bhtt.size(3) != T:
        raise ValueError(f"attn last dims must be [T,T] with T={T}, got {tuple(attn_bhtt.shape)}")

    # ---- CLS->patch attention scores ----
    # attn[:, :, 0, 1:] -> [B,H,N]
    scores = attn_bhtt[:, :, 0, 1:].mean(dim=1)  # [B,N]

    idx = _topk_indices(scores, k)  # [B,k] in patch-token coordinates [0..N-1]

    # gather patch tokens
    cls = x[:, :1, :]      # [B,1,C]
    toks = x[:, 1:, :]     # [B,N,C]
    toks_kept = _gather_tokens(toks, idx)

    return torch.cat([cls, toks_kept], dim=1) if keep_cls else toks_kept


def maybe_thin_after_block(
    x: torch.Tensor,
    block_idx: int,
    schedule: ThinningSchedule,
    method: str = "l2",
    attn: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Helper: after running block_idx, apply thinning if schedule says so.

    method:
      - "l2": keep tokens with largest L2 norm (v1 baseline)
      - "random": random ablation
      - "attn": keep tokens with highest CLS->token attention (method Eq.(1))
               requires attn to be provided for that block.
    """
    r = schedule.ratio_after_block(block_idx)
    if r is None or r >= 1.0:
        return x

    if method == "l2":
        return thin_tokens_l2(x, keep_ratio=r, keep_cls=True)
    if method == "random":
        return thin_tokens_random(x, keep_ratio=r, keep_cls=True)
    if method == "attn":
        if attn is None:
            raise ValueError("method='attn' requires attn tensor (attention weights) for this block")
        return thin_tokens_attn_cls(x, attn=attn, keep_ratio=r, keep_cls=True)

    raise ValueError(f"Unknown thinning method: {method}")
