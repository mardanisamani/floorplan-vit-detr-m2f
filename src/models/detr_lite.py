"""
DETR-lite: a compact re-implementation of DETR
(Carion et al., 'End-to-End Object Detection with Transformers', ECCV 2020).

Differences vs. the original for compactness / CPU training:
- Small custom CNN backbone (no ResNet-50).
- 3 encoder + 3 decoder layers.
- 50 object queries by default.
- Matcher + loss follow the original (Hungarian matching + CE + L1 + GIoU).

Forward output:
    {
        "pred_logits": (B, Q, n_classes + 1)   # last = no-object
        "pred_boxes":  (B, Q, 4)               # cxcywh in [0, 1]
    }
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone + positional encoding
# ---------------------------------------------------------------------------

class SmallCNNBackbone(nn.Module):
    """Tiny CNN that downsamples by 16 and returns feature map + HxW."""
    def __init__(self, out_ch: int = 128):
        super().__init__()
        def block(ci, co, stride=2):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(8, co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.GroupNorm(8, co),
                nn.ReLU(inplace=True),
            )
        self.stem = block(3,  32, stride=2)  # /2
        self.l2   = block(32, 64, stride=2)  # /4
        self.l3   = block(64, 128, stride=2) # /8
        self.l4   = block(128, out_ch, stride=2)  # /16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x                                                  # (B, C, H/16, W/16)


class SinePositional2D(nn.Module):
    """2D sinusoidal positional encoding."""
    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.temperature = temperature

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y = torch.arange(h, dtype=torch.float32, device=device)
        x = torch.arange(w, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yy = yy / max(h - 1, 1)
        xx = xx / max(w - 1, 1)
        d = self.dim // 4
        i = torch.arange(d, dtype=torch.float32, device=device)
        freq = 1.0 / (self.temperature ** (2 * (i // 2) / (2 * d)))
        pe_x = xx[..., None] * freq
        pe_y = yy[..., None] * freq
        pe = torch.cat([pe_x.sin(), pe_x.cos(), pe_y.sin(), pe_y.cos()], dim=-1)
        return pe.permute(2, 0, 1).contiguous()                    # (dim, H, W)


# ---------------------------------------------------------------------------
# Transformer encoder / decoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        q = k = src + pos
        attn_out, _ = self.attn(q, k, src, need_weights=False)
        src = self.norm1(src + self.drop1(attn_out))
        src = self.norm2(src + self.drop2(self.ff(src)))
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos_mem: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        q = k = tgt + query_pos
        sa, _ = self.self_attn(q, k, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.drop1(sa))

        ca, _ = self.cross_attn(tgt + query_pos, memory + pos_mem, memory,
                                need_weights=False)
        tgt = self.norm2(tgt + self.drop2(ca))

        tgt = self.norm3(tgt + self.drop3(self.ff(tgt)))
        return tgt


# ---------------------------------------------------------------------------
# DETR-lite
# ---------------------------------------------------------------------------

class DETRLite(nn.Module):
    def __init__(self,
                 n_classes: int = 5,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_encoder: int = 3,
                 n_decoder: int = 3,
                 ff: int = 512,
                 n_queries: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = SmallCNNBackbone(out_ch=d_model)
        self.pe = SinePositional2D(d_model)

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ff, dropout)
            for _ in range(n_encoder)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, ff, dropout)
            for _ in range(n_decoder)
        ])

        self.query_embed = nn.Embedding(n_queries, d_model)
        self.cls_head = nn.Linear(d_model, n_classes + 1)            # +1 for "no object"
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)                                   # (B, C, H, W)
        B, C, H, W = feats.shape
        pe = self.pe(H, W, x.device).unsqueeze(0).expand(B, -1, -1, -1)
        src = feats.flatten(2).transpose(1, 2)                     # (B, N, C)
        pos = pe.flatten(2).transpose(1, 2)                        # (B, N, C)

        for layer in self.encoder:
            src = layer(src, pos)

        q_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(q_embed)
        for layer in self.decoder:
            tgt = layer(tgt, src, pos, q_embed)

        logits = self.cls_head(tgt)                                # (B, Q, C+1)
        boxes = self.box_head(tgt).sigmoid()                       # cxcywh in [0,1]
        return {"pred_logits": logits, "pred_boxes": boxes}


# ---------------------------------------------------------------------------
# Hungarian matcher + losses
# ---------------------------------------------------------------------------

def _box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _box_iou_and_giou(boxes1: torch.Tensor, boxes2: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return IoU and GIoU matrix between two sets of xyxy boxes.

    boxes1: (N, 4), boxes2: (M, 4) -> (N, M), (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    iou = inter / union

    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    enc = wh_enc[..., 0] * wh_enc[..., 1] + 1e-6
    giou = iou - (enc - union) / enc
    return iou, giou


class HungarianMatcher(nn.Module):
    """Matches predicted queries to ground-truth boxes via
    scipy.optimize.linear_sum_assignment over a cost combining
    class prob, L1 box distance, and GIoU."""

    def __init__(self, w_cls: float = 1.0, w_bbox: float = 5.0,
                 w_giou: float = 2.0):
        super().__init__()
        self.w_cls = w_cls
        self.w_bbox = w_bbox
        self.w_giou = w_giou

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[Dict[str, torch.Tensor]]
                ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        from scipy.optimize import linear_sum_assignment

        bs, nq = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)              # (B, Q, C+1)
        out_bbox = outputs["pred_boxes"]                           # (B, Q, 4)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]                         # (M,)
            tgt_box = targets[b]["boxes"]                          # (M, 4) cxcywh

            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
                continue

            cost_cls = -out_prob[b][:, tgt_ids]                    # (Q, M)
            cost_bbox = torch.cdist(out_bbox[b], tgt_box, p=1)     # (Q, M)
            _, giou = _box_iou_and_giou(_box_cxcywh_to_xyxy(out_bbox[b]),
                                        _box_cxcywh_to_xyxy(tgt_box))
            cost_giou = -giou

            C = (self.w_cls * cost_cls
                 + self.w_bbox * cost_bbox
                 + self.w_giou * cost_giou).cpu().numpy()

            i, j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)))
        return indices


def detr_loss(outputs: Dict[str, torch.Tensor],
              targets: List[Dict[str, torch.Tensor]],
              matcher: HungarianMatcher,
              n_classes: int,
              w_cls: float = 1.0, w_bbox: float = 5.0, w_giou: float = 2.0,
              eos_coef: float = 0.1) -> Dict[str, torch.Tensor]:
    """DETR loss = classification (CE with down-weighted no-object) +
    L1 + GIoU box losses on matched predictions."""

    indices = matcher(outputs, targets)

    logits = outputs["pred_logits"]                                # (B, Q, C+1)
    bs, nq, _ = logits.shape

    # Classification target: default = no-object (index n_classes)
    tgt_class = torch.full((bs, nq), n_classes, dtype=torch.int64,
                           device=logits.device)
    for b, (pi, ti) in enumerate(indices):
        tgt_class[b, pi] = targets[b]["labels"][ti].to(logits.device)

    weight = torch.ones(n_classes + 1, device=logits.device)
    weight[-1] = eos_coef
    loss_cls = F.cross_entropy(logits.flatten(0, 1), tgt_class.flatten(),
                               weight=weight)

    # Box loss only on matched predictions
    pred_boxes = outputs["pred_boxes"]                             # (B, Q, 4)
    matched_pred, matched_tgt = [], []
    for b, (pi, ti) in enumerate(indices):
        if pi.numel() == 0:
            continue
        matched_pred.append(pred_boxes[b, pi])
        matched_tgt.append(targets[b]["boxes"][ti].to(pred_boxes.device))

    if matched_pred:
        p = torch.cat(matched_pred, 0)
        t = torch.cat(matched_tgt, 0)
        loss_bbox = F.l1_loss(p, t, reduction="mean")
        _, giou = _box_iou_and_giou(_box_cxcywh_to_xyxy(p),
                                    _box_cxcywh_to_xyxy(t))
        loss_giou = (1.0 - giou.diag()).mean()
    else:
        loss_bbox = torch.tensor(0.0, device=logits.device)
        loss_giou = torch.tensor(0.0, device=logits.device)

    total = w_cls * loss_cls + w_bbox * loss_bbox + w_giou * loss_giou
    return {"loss": total, "loss_cls": loss_cls,
            "loss_bbox": loss_bbox, "loss_giou": loss_giou}
