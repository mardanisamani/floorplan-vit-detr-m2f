"""
Mask2Former-lite: a compact, educational re-implementation of
Mask2Former (Cheng et al., CVPR 2022) for semantic segmentation.

Architecture:
  1. CNN backbone produces multi-scale features.
  2. Pixel decoder (FPN-like) up-samples the deepest feature to /4 resolution.
  3. Transformer decoder with `n_queries` learnable embeddings attends to
     the backbone features. Each query yields (class logits, mask embedding).
  4. Per-pixel mask is computed as dot-product between each mask embedding
     and the pixel-decoder feature map.

For semantic segmentation we post-process by taking the argmax over
`sum_q softmax(class) * mask(q)` across queries.

Designed to train on CPU in minutes on the tiny synthetic dataset.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone (shared style with DETR-lite but exposes multi-scale features)
# ---------------------------------------------------------------------------

def _conv_block(ci, co, stride=1):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, co),
        nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1, bias=False),
        nn.GroupNorm(8, co),
        nn.ReLU(inplace=True),
    )


class MultiScaleBackbone(nn.Module):
    """Return features at strides 4, 8, 16."""
    def __init__(self):
        super().__init__()
        self.s2 = _conv_block(3,   32, stride=2)   # /2
        self.s4 = _conv_block(32,  64, stride=2)   # /4
        self.s8 = _conv_block(64,  128, stride=2)  # /8
        self.s16 = _conv_block(128, 192, stride=2) # /16

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.s2(x)
        f4 = self.s4(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        return [f4, f8, f16]                                       # (B, 64, H/4, W/4), etc.


# ---------------------------------------------------------------------------
# Pixel decoder (simple FPN upsample to /4 resolution)
# ---------------------------------------------------------------------------

class PixelDecoder(nn.Module):
    def __init__(self, channels=(64, 128, 192), out_ch: int = 128):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in channels])
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in channels
        ])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats: [f4, f8, f16] ordered by resolution (high to low)
        x = self.lat[2](feats[2])                                  # /16
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x + self.lat[1](feats[1])                              # /8
        x = self.smooth[1](x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x + self.lat[0](feats[0])                              # /4
        x = self.smooth[0](x)
        return x                                                   # (B, out_ch, H/4, W/4)


# ---------------------------------------------------------------------------
# Transformer decoder with masked cross-attention
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, ff: int = 384, dropout: float = 0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.self_ = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(ff, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # q: (B, Q, D); memory: (B, N, D); attn_mask: (B*n_heads, Q, N) bool
        ca, _ = self.cross(q, memory, memory, attn_mask=attn_mask,
                           need_weights=False)
        q = self.norm1(q + self.drop(ca))
        sa, _ = self.self_(q, q, q, need_weights=False)
        q = self.norm2(q + self.drop(sa))
        q = self.norm3(q + self.drop(self.ffn(q)))
        return q


# ---------------------------------------------------------------------------
# Mask2Former-lite
# ---------------------------------------------------------------------------

class Mask2FormerLite(nn.Module):
    """Predicts per-pixel semantic segmentation via query-based masks.

    forward returns:
        {
            "mask_logits": (B, Q, H, W)  # pixel-level mask scores
            "class_logits": (B, Q, C+1)  # per-query class (last = no-object)
            "seg_logits": (B, C, H, W)   # aggregated semantic-seg logits
        }

    where H = W = input / 4 (pixel-decoder resolution).
    """

    def __init__(self,
                 n_classes: int = 5,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_decoder: int = 3,
                 n_queries: int = 20,
                 ff: int = 384,
                 dropout: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = MultiScaleBackbone()
        self.pixel_decoder = PixelDecoder(out_ch=d_model)

        # Project backbone features into decoder memory at each resolution.
        self.mem_proj = nn.ModuleList([
            nn.Conv2d(c, d_model, 1) for c in (64, 128, 192)
        ])

        self.query_embed = nn.Embedding(n_queries, d_model)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, ff, dropout)
            for _ in range(n_decoder)
        ])
        self.class_head = nn.Linear(d_model, n_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)                                   # [f4, f8, f16]
        pixel = self.pixel_decoder(feats)                          # (B, D, H/4, W/4)

        B, D, Hp, Wp = pixel.shape
        memory_tokens = []
        for proj, f in zip(self.mem_proj, feats):
            m = proj(f).flatten(2).transpose(1, 2)                 # (B, N_i, D)
            memory_tokens.append(m)
        memory = torch.cat(memory_tokens, dim=1)                   # (B, N_total, D)

        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) # (B, Q, D)
        for layer in self.decoder_layers:
            q = layer(q, memory)

        cls_logits = self.class_head(q)                            # (B, Q, C+1)
        mask_emb = self.mask_head(q)                               # (B, Q, D)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_emb, pixel)

        # Semantic-seg logits: sum over queries of softmax(class) * sigmoid(mask)
        cls_prob = cls_logits[..., :-1].softmax(-1)                # drop no-obj
        mask_prob = mask_logits.sigmoid()
        seg = torch.einsum("bqc,bqhw->bchw", cls_prob, mask_prob)

        return {
            "mask_logits": mask_logits,
            "class_logits": cls_logits,
            "seg_logits": seg,
            "pixel_features": pixel,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. pred and target: (B, C, H, W), pred in [0,1]."""
    pred = pred.flatten(2)
    target = target.flatten(2)
    inter = (pred * target).sum(-1)
    denom = pred.sum(-1) + target.sum(-1)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def mask2former_loss(outputs: Dict[str, torch.Tensor],
                     seg_targets: torch.Tensor,
                     n_classes: int,
                     w_ce: float = 1.0, w_dice: float = 1.0) -> Dict[str, torch.Tensor]:
    """Simplified loss for semantic segmentation supervision.

    We bypass set-prediction matching by supervising the aggregated
    per-class `seg_logits` directly (cross-entropy + dice). This is a
    practical simplification that works well for small demos while still
    exercising the query-based architecture.

    seg_targets: (B, H_full, W_full) int64
    """
    seg = outputs["seg_logits"]                                    # (B, C, H/4, W/4)
    _, _, Hp, Wp = seg.shape
    target_resized = F.interpolate(
        seg_targets.float().unsqueeze(1), size=(Hp, Wp), mode="nearest"
    ).long().squeeze(1)

    ce = F.cross_entropy(seg, target_resized)

    # One-hot for dice
    oh = F.one_hot(target_resized, num_classes=n_classes).permute(0, 3, 1, 2).float()
    prob = seg.softmax(1)
    dice = _dice_loss(prob, oh)

    loss = w_ce * ce + w_dice * dice
    return {"loss": loss, "loss_ce": ce, "loss_dice": dice}
