import timm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from src.algos.token_thinning import (
    ThinningSchedule,
    maybe_thin_after_block,
    ScoreHead,
)


def _build_spatial_center_prior(num_patches: int, grid_size: int, sigma: float = 3.0) -> torch.Tensor:
    """
    Spatial center prior (weak proxy for anatomical prior).
    Weights center-medial region higher; approximates that hippocampal/ventricular
    regions often lie near the center in axial MRI. NOT an atlas-based anatomical
    prior (e.g. hippocampus ROI mask). Use anatomical_prior_path for true anatomical prior.
    Returns [N] normalized importance in [0,1].
    """
    h = w = grid_size  # e.g. 14 for 224/16
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
    x = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    prior = torch.exp(-dist_sq / (2 * sigma ** 2))
    prior = prior.flatten()  # [N]
    prior = prior / (prior.max() + 1e-8)
    return prior


def _build_anatomical_prior_from_mask(
    mask_path: str,
    grid_size: int = 14,
    slice_idx: Optional[int] = None,
    axis: int = 2,
) -> torch.Tensor:
    """
    Build M_prior from atlas mask NIfTI (e.g. hippocampus ROI).
    Extracts 2D slice, downsamples to grid_size x grid_size, normalizes.
    Returns [N] importance. Use for anatomically grounded prior.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required for anatomical mask prior")
    img = nib.load(mask_path)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Mask must be 3D, got shape {vol.shape}")
    vol = np.moveaxis(vol, axis, -1)  # (H,W,S)
    z = slice_idx if slice_idx is not None else vol.shape[-1] // 2
    slc = vol[..., z]  # (H,W)
    # Downsample to grid_size
    t = torch.from_numpy(slc).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    prior = t.squeeze().flatten()
    prior = prior / (prior.max() + 1e-8)
    return prior


# -------------------------
# Early-exit helpers / heads
# -------------------------
class EarlyExitHead(nn.Module):
    """Lightweight classifier head operating on CLS token."""
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, cls_vec: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(cls_vec))


def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """U = -Σ p_k log p_k. p: [B, C]"""
    return -(p * (p + eps).log()).sum(dim=1)


@torch.no_grad()
def early_exit_select(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    logits3: torch.Tensor,
    tau: float,
    tau_u: Optional[float] = None,
):
    """
    Early exit (Proposal §3.6).
    - Confidence-based: exit if max(softmax) >= tau.
    - Uncertainty-guided: if tau_u is set, also require entropy <= tau_u.
      Exit only when BOTH confidence >= tau AND entropy <= tau_u.
    """
    p1 = F.softmax(logits1, dim=1)
    conf1, _ = p1.max(dim=1)
    u1 = _entropy(p1)
    exit1 = (conf1 >= tau) & (u1 <= tau_u if tau_u is not None else torch.ones_like(conf1, dtype=torch.bool))

    p2 = F.softmax(logits2, dim=1)
    conf2, _ = p2.max(dim=1)
    u2 = _entropy(p2)
    exit2 = (conf2 >= tau) & (u2 <= tau_u if tau_u is not None else torch.ones_like(conf2, dtype=torch.bool)) & (~exit1)

    exit3 = ~(exit1 | exit2)

    final_logits = logits3.clone()
    final_logits[exit1] = logits1[exit1]
    final_logits[exit2] = logits2[exit2]

    exit_layer = torch.empty(logits1.size(0), device=logits1.device, dtype=torch.long)
    exit_layer[exit1] = 1
    exit_layer[exit2] = 2
    exit_layer[exit3] = 3

    return final_logits, exit_layer, (exit1, exit2, exit3)


# -------------------------
# Attention capture (your original)
# -------------------------
class _AttnInputCatcher:
    """Pre-hook to capture the input token sequence to blk.attn (already normed in timm Block)."""
    def __init__(self):
        self.x_in = None

    def reset(self):
        self.x_in = None

    def __call__(self, module, inputs):
        # inputs[0] should be x: [B, T, C]
        self.x_in = inputs[0]


def _compute_attn_matrix_from_timm_attn(attn_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Recompute attention matrix A from a timm ViT Attention module and its input x.

    Returns:
      attn: [B, H, T, T]  (after softmax)
    """
    if not hasattr(attn_module, "qkv"):
        raise RuntimeError(
            "Cannot recompute attn: attn_module has no attribute 'qkv'. "
            "Please print vit.blocks[0].attn to inspect the module."
        )
    if not hasattr(attn_module, "num_heads"):
        raise RuntimeError(
            "Cannot recompute attn: attn_module has no attribute 'num_heads'. "
            "Please print vit.blocks[0].attn to inspect the module."
        )

    B, T, C = x.shape
    H = attn_module.num_heads
    head_dim = C // H
    scale = getattr(attn_module, "scale", head_dim ** -0.5)

    qkv = attn_module.qkv(x)  # [B, T, 3*C]
    qkv = qkv.reshape(B, T, 3, H, head_dim).permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
    q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, T, D]

    attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]
    attn = attn.softmax(dim=-1)
    return attn


# -------------------------
# Main wrapper with thinning + early exit
# -------------------------
class TimmViTWithThinning(nn.Module):
    """
    JSD-ViT: Joint Spatial-Depth Adaptive ViT (Proposal).
    Token thinning (l2/random/attn/learnable) + uncertainty-guided early exit.
    """

    def __init__(
        self,
        vit: nn.Module,
        schedule: ThinningSchedule,
        method: str = "l2",
        debug: bool = False,
        enable_early_exit: bool = False,
        exit_blocks=(3, 7, 11),
        exit_dropout: float = 0.0,
        use_anatomical_prior: bool = False,
        anatomical_prior_path: Optional[str] = None,
        anatomical_prior_slice: Optional[int] = None,
        num_patches: int = 196,
        grid_size: int = 14,
        gumbel_tau: float = 0.0,
    ):
        super().__init__()
        self.vit = vit
        self.schedule = schedule
        self.method = method
        self.debug = debug
        self.use_anatomical_prior = use_anatomical_prior
        self.num_patches = num_patches
        self.gumbel_tau = gumbel_tau

        self.enable_early_exit = enable_early_exit
        self.exit_blocks = tuple(exit_blocks)

        embed_dim = getattr(vit, "embed_dim", 768)
        num_classes = vit.head.out_features if hasattr(vit.head, "out_features") else None
        if num_classes is None:
            raise RuntimeError("Cannot infer num_classes from vit.head.")

        if self.enable_early_exit:
            self.exit_head1 = EarlyExitHead(embed_dim, num_classes, dropout=exit_dropout)
            self.exit_head2 = EarlyExitHead(embed_dim, num_classes, dropout=exit_dropout)
        else:
            self.exit_head1 = None
            self.exit_head2 = None

        # Learnable ScoreHeads (Proposal §3.3) - one per thinning block
        self.score_heads = nn.ModuleDict()
        if self.method == "learnable":
            for blk_idx in self.schedule.keep_ratio_by_block:
                self.score_heads[str(blk_idx)] = ScoreHead(embed_dim)

        # Anatomical prior M_prior (Proposal §3.3).
        # - anatomical_prior_path: atlas mask NIfTI (e.g. hippocampus ROI) -> anatomically grounded.
        # - else: spatial center prior (weak proxy, not atlas-based).
        if use_anatomical_prior:
            if anatomical_prior_path:
                prior = _build_anatomical_prior_from_mask(
                    anatomical_prior_path, grid_size,
                    slice_idx=anatomical_prior_slice, axis=2,
                )
            else:
                prior = _build_spatial_center_prior(num_patches, grid_size)
            self.register_buffer("m_prior", prior.unsqueeze(0))
        else:
            self.register_buffer("m_prior", None)

        self._catchers = []
        self._prehooks = []
        if self.method == "attn":
            self._install_attn_input_hooks()

    def _install_attn_input_hooks(self):
        self._catchers = []
        self._prehooks = []
        for i, blk in enumerate(self.vit.blocks):
            if not hasattr(blk, "attn"):
                raise AttributeError(f"Block {i} has no attribute 'attn'.")
            catcher = _AttnInputCatcher()
            h = blk.attn.register_forward_pre_hook(catcher)
            self._catchers.append(catcher)
            self._prehooks.append(h)

    def _forward_tokens_through_blocks(self, x: torch.Tensor):
        """
        Forward tokens through timm blocks with thinning.
        Returns logits1, logits2, logits3 and aux dict with token_counts, m_token for losses.
        """
        vit = self.vit
        logits1 = logits2 = None
        token_counts = []  # (block_idx, N_ℓ) for L_sparse
        m_token_list = []  # [B,N] importance from ScoreHead for L_anatomy
        N0 = x.shape[1] - 1  # initial patch count

        for i, blk in enumerate(vit.blocks):
            if self.method == "attn":
                self._catchers[i].reset()

            x = blk(x)

            if self.debug and i in self.schedule.keep_ratio_by_block:
                print(f"[thin] before block {i}: tokens={x.shape[1]}")

            r = self.schedule.ratio_after_block(i)
            if r is not None and r < 1.0:
                if self.method == "attn":
                    x_in = self._catchers[i].x_in
                    if x_in is None:
                        raise RuntimeError(f"Failed to capture attn for block {i}")
                    attn = _compute_attn_matrix_from_timm_attn(blk.attn, x_in)
                    x = maybe_thin_after_block(
                        x, block_idx=i, schedule=self.schedule, method="attn", attn=attn
                    )
                elif self.method == "learnable" and str(i) in self.score_heads:
                    sh = self.score_heads[str(i)]
                    toks = x[:, 1:, :]
                    scores = sh(toks)  # [B, N]
                    if self.use_anatomical_prior and self.m_prior is not None:
                        m_token_list.append(scores)  # for L_anatomy
                    x = maybe_thin_after_block(
                        x, block_idx=i, schedule=self.schedule,
                        method="learnable", score_head=sh, gumbel_tau=self.gumbel_tau
                    )
                else:
                    x = maybe_thin_after_block(
                        x, block_idx=i, schedule=self.schedule, method=self.method
                    )
                token_counts.append((i, x.shape[1] - 1))

            if self.debug and i in self.schedule.keep_ratio_by_block:
                print(f"[thin] after  block {i}: tokens={x.shape[1]}")

            if self.enable_early_exit:
                if i == self.exit_blocks[0]:
                    cls_vec = vit.norm(x)[:, 0]
                    logits1 = self.exit_head1(cls_vec)
                elif i == self.exit_blocks[1]:
                    cls_vec = vit.norm(x)[:, 0]
                    logits2 = self.exit_head2(cls_vec)

        x = vit.norm(x)
        cls_feature = x[:, 0]  # [B, D] for feature distillation (Proposal §3.7)
        logits3 = vit.head(cls_feature)

        aux = {
            "token_counts": token_counts,
            "N0": N0,
            "m_token_list": m_token_list,
            "cls_feature": cls_feature,
        }
        return logits1, logits2, logits3, aux

    def forward_all(self, x: torch.Tensor):
        """
        Returns (logits1, logits2, logits3, aux).
        aux: dict with token_counts, N0, m_token_list for L_sparse, L_anatomy.
        """
        vit = self.vit
        x = vit.patch_embed(x)
        cls = vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + vit.pos_embed[:, : x.shape[1], :]
        x = vit.pos_drop(x)
        return self._forward_tokens_through_blocks(x)

    def forward(
        self,
        x: torch.Tensor,
        tau: Optional[float] = None,
        tau_u: Optional[float] = None,
        return_aux: bool = False,
    ):
        """
        tau: confidence threshold for early exit.
        tau_u: entropy threshold (Proposal §3.6); exit only if entropy <= tau_u.
        return_aux: if True, return (logits, exit_layer?, aux) for training losses.
        """
        logits1, logits2, logits3, aux = self.forward_all(x)

        if tau is None:
            if return_aux:
                return logits3, aux
            return logits3

        if (logits1 is None) or (logits2 is None):
            exit_layer = torch.full((logits3.size(0),), 3, device=logits3.device, dtype=torch.long)
            if return_aux:
                return logits3, exit_layer, aux
            return logits3, exit_layer

        final_logits, exit_layer, _ = early_exit_select(
            logits1, logits2, logits3, tau, tau_u=tau_u
        )
        if return_aux:
            return final_logits, exit_layer, aux
        return final_logits, exit_layer


def build_vit2d(
    num_classes: int,
    thinning: bool = False,
    thin_method: str = "l2",   # "l2" | "random" | "attn" | "learnable"
    debug_thin: bool = False,
    enable_early_exit: bool = False,
    exit_blocks=(3, 7, 11),
    exit_dropout: float = 0.0,
    use_anatomical_prior: bool = False,
    anatomical_prior_path: Optional[str] = None,
    anatomical_prior_slice: Optional[int] = None,
    gumbel_tau: float = 0.0,
    pretrained: bool = True,
    in_chans: int = 1,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if not thinning and not enable_early_exit:
        # pure timm ViT
        vit.enable_early_exit = False
        return vit
    num_patches = 196  # vit_base_patch16_224: 14*14
    grid_size = 14

    if enable_early_exit and not thinning:
        return TimmViTWithThinning(
            vit=vit,
            schedule=ThinningSchedule(keep_ratio_by_block={}),
            method=thin_method,
            debug=debug_thin,
            enable_early_exit=True,
            exit_blocks=exit_blocks,
            exit_dropout=exit_dropout,
            use_anatomical_prior=use_anatomical_prior,
            anatomical_prior_path=anatomical_prior_path,
            anatomical_prior_slice=anatomical_prior_slice,
            num_patches=num_patches,
            grid_size=grid_size,
            gumbel_tau=gumbel_tau,
        )

    schedule = ThinningSchedule(
        keep_ratio_by_block={
            4: 0.75,
            5: 0.75,
            6: 0.75,
            7: 0.75,
            8: 0.50,
            9: 0.50,
            10: 0.50,
            11: 0.50,
        }
    )

    return TimmViTWithThinning(
        vit=vit,
        schedule=schedule if thinning else ThinningSchedule(keep_ratio_by_block={}),
        method=thin_method,
        debug=debug_thin,
        enable_early_exit=enable_early_exit,
        exit_blocks=exit_blocks,
        exit_dropout=exit_dropout,
        use_anatomical_prior=use_anatomical_prior,
        anatomical_prior_path=anatomical_prior_path,
        anatomical_prior_slice=anatomical_prior_slice,
        num_patches=num_patches,
        grid_size=grid_size,
        gumbel_tau=gumbel_tau,
    )
