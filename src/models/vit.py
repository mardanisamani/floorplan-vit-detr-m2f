"""
Minimal Vision Transformer for room-type classification.

Follows Dosovitskiy et al. (2020) - "An Image is Worth 16x16 Words".

Default configuration (kept small so it trains in minutes on CPU):
    img_size = 256, patch_size = 16 -> 256 tokens
    embed_dim = 192, depth = 6, heads = 4  -> ~2.5M parameters
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image -> sequence of patch embeddings via a single strided conv."""
    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, C, H, W)
        x = self.proj(x)                                           # (B, D, G, G)
        return x.flatten(2).transpose(1, 2)                        # (B, G*G, D)


class MHSA(nn.Module):
    """Standard multi-head self-attention (pre-norm usage expected externally)."""
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                           # (3, B, h, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale              # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT classifier
# ---------------------------------------------------------------------------

class ViTClassifier(nn.Module):
    """Vision Transformer classifier.

    Args:
        n_classes: number of target classes (e.g., number of ROOM_TYPES).
    """

    def __init__(self,
                 img_size: int = 256,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 192,
                 depth: int = 6,
                 n_heads: int = 4,
                 mlp_mult: int = 4,
                 dropout: float = 0.1,
                 n_classes: int = 6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_mult, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                                    # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                                        # (B, N+1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats[:, 0])                              # logits
