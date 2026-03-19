# models/early_exit.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyExitHead(nn.Module):
    """Lightweight classifier head operating on CLS token."""
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, cls_vec: torch.Tensor) -> torch.Tensor:
        # cls_vec: [B, D]
        return self.fc(self.dropout(cls_vec))


@torch.no_grad()
def early_exit_select(logits1: torch.Tensor,
                      logits2: torch.Tensor,
                      logits3: torch.Tensor,
                      tau: float):
    """
    Implements: exit at stage k if max softmax prob >= tau.
    Stage 3 is always taken if not exited earlier.
    Returns:
      final_logits: [B, C]
      exit_layer:  [B] in {1,2,3}
    """
    p1 = F.softmax(logits1, dim=1)
    conf1, _ = p1.max(dim=1)
    exit1 = conf1 >= tau

    p2 = F.softmax(logits2, dim=1)
    conf2, _ = p2.max(dim=1)
    exit2 = (conf2 >= tau) & (~exit1)

    exit3 = ~(exit1 | exit2)

    # choose logits per sample
    final_logits = logits3.clone()
    final_logits[exit1] = logits1[exit1]
    final_logits[exit2] = logits2[exit2]

    exit_layer = torch.empty(logits1.size(0), device=logits1.device, dtype=torch.long)
    exit_layer[exit1] = 1
    exit_layer[exit2] = 2
    exit_layer[exit3] = 3

    return final_logits, exit_layer, (exit1, exit2, exit3)
def build_early_exit_heads(
    vit,
    num_classes: int,
    exit_blocks,
    dropout: float = 0.0,
):
    """
    Build early-exit heads for ViT.
    Each head operates on CLS token embedding.
    """

    # timm ViT usually has one of these
    in_dim = getattr(vit, "num_features", None)
    if in_dim is None:
        in_dim = getattr(vit, "embed_dim", None)
    if in_dim is None:
        raise RuntimeError("Cannot infer ViT embedding dim for early-exit heads.")

    heads = nn.ModuleList(
        [EarlyExitHead(in_dim, num_classes, dropout=dropout)
         for _ in exit_blocks]
    )
    return heads
