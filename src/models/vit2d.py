import timm
import torch
from torch import nn
import torch.nn.functional as F

from src.algos.token_thinning import (
    ThinningSchedule,
    maybe_thin_after_block,
)
from src.models.early_exit import build_early_exit_heads

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


@torch.no_grad()
def early_exit_select(logits1: torch.Tensor,
                      logits2: torch.Tensor,
                      logits3: torch.Tensor,
                      tau: float):
    """
    Confidence-based early exit:
      exit at stage k if max softmax prob >= tau.
      stage 3 is always taken if not exited earlier.

    Returns:
      final_logits: [B, C]
      exit_layer:  [B] in {1,2,3}
      masks: (exit1, exit2, exit3) boolean masks
    """
    p1 = F.softmax(logits1, dim=1)
    conf1, _ = p1.max(dim=1)
    exit1 = conf1 >= tau

    p2 = F.softmax(logits2, dim=1)
    conf2, _ = p2.max(dim=1)
    exit2 = (conf2 >= tau) & (~exit1)

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
    Wrapper around timm ViT to support token thinning + early exit heads.

    methods:
      - "l2" / "random": baseline thinning
      - "attn": CLS->token attention importance thinning
    """

    def __init__(
        self,
        vit: nn.Module,
        schedule: ThinningSchedule,
        method: str = "l2",
        debug: bool = False,
        enable_early_exit: bool = False,
        exit_blocks=(3, 7, 11),   # stage ends in block indices
        exit_dropout: float = 0.0,
    ):
        super().__init__()
        self.vit = vit
        self.schedule = schedule
        self.method = method
        self.debug = debug

        # early-exit config
        self.enable_early_exit = enable_early_exit
        self.exit_blocks = tuple(exit_blocks)

        # vit_base_patch16_224: embed dim = 768
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

        # attention hooks for attn thinning
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
        If enable_early_exit, collect CLS logits at stage ends.

        Returns:
          logits1, logits2, logits3 (if enable_early_exit)
          otherwise returns only logits3 and None, None
        """
        vit = self.vit
        logits1 = logits2 = None

        # ---- transformer blocks ----
        for i, blk in enumerate(vit.blocks):
            if self.method == "attn":
                self._catchers[i].reset()

            x = blk(x)

            if self.debug and i in self.schedule.keep_ratio_by_block:
                print(f"[thin] before block {i}: tokens={x.shape[1]}")

            # apply thinning after selected blocks
            if self.method == "attn" and i in self.schedule.keep_ratio_by_block:
                x_in = self._catchers[i].x_in
                if x_in is None:
                    raise RuntimeError(
                        f"Failed to capture attn input for block {i}. "
                        "Unexpected: pre-hook did not fire."
                    )
                attn = _compute_attn_matrix_from_timm_attn(blk.attn, x_in)  # [B,H,T,T]
                x = maybe_thin_after_block(
                    x, block_idx=i, schedule=self.schedule, method="attn", attn=attn
                )
            else:
                x = maybe_thin_after_block(
                    x, block_idx=i, schedule=self.schedule, method=self.method
                )

            if self.debug and i in self.schedule.keep_ratio_by_block:
                print(f"[thin] after  block {i}: tokens={x.shape[1]}")

            # stage end -> compute early-exit logits from CLS
            if self.enable_early_exit:
                if i == self.exit_blocks[0]:
                    cls_vec = vit.norm(x)[:, 0]      # [B, D]
                    logits1 = self.exit_head1(cls_vec)
                elif i == self.exit_blocks[1]:
                    cls_vec = vit.norm(x)[:, 0]
                    logits2 = self.exit_head2(cls_vec)

        # final head
        x = vit.norm(x)
        logits3 = vit.head(x[:, 0])
        return logits1, logits2, logits3

    def forward_all(self, x: torch.Tensor):
        """
        Always returns 3 logits:
          logits1, logits2 can be None if early-exit disabled.
        """
        vit = self.vit

        # ---- patch embedding ----
        x = vit.patch_embed(x)  # [B, N, C]

        # ---- add CLS token ----
        cls = vit.cls_token.expand(x.shape[0], -1, -1)  # [B,1,C]
        x = torch.cat((cls, x), dim=1)                  # [B, 1+N, C]

        # ---- positional embedding + dropout ----
        x = x + vit.pos_embed[:, : x.shape[1], :]
        x = vit.pos_drop(x)

        return self._forward_tokens_through_blocks(x)

    def forward(self, x: torch.Tensor, tau: float = None):
        """
        Default behavior:
          - tau is None: return final logits (stage 3) for training / normal eval
          - tau is not None: perform confidence-based early exit, return (final_logits, exit_layer)
        """
        logits1, logits2, logits3 = self.forward_all(x)

        if tau is None:
            return logits3

        # If early-exit is not enabled, fall back to final logits
        if (logits1 is None) or (logits2 is None):
            exit_layer = torch.full((logits3.size(0),), 3, device=logits3.device, dtype=torch.long)
            return logits3, exit_layer

        final_logits, exit_layer, _ = early_exit_select(logits1, logits2, logits3, tau)
        return final_logits, exit_layer


def build_vit2d(
    num_classes: int,
    thinning: bool = False,
    thin_method: str = "l2",   # "l2" | "random" | "attn"
    debug_thin: bool = False,
    enable_early_exit: bool = False,
    exit_blocks=(3, 7, 11),
    exit_dropout: float = 0.0,
):
    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        in_chans=1,
        num_classes=num_classes,
    )

    if not thinning and not enable_early_exit:
        # pure timm ViT
        vit.enable_early_exit = False
        return vit
    # ---- Early-exit only (no thinning) ----
    if enable_early_exit and not thinning:
        return TimmViTWithThinning(
            vit=vit,
            schedule=ThinningSchedule(keep_ratio_by_block={}),  # no thinning
            method=thin_method,
            debug=debug_thin,
            enable_early_exit=True,
            exit_blocks=exit_blocks,
            exit_dropout=exit_dropout,
        )

    # keep your schedule
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
    )
