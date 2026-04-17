"""
PyTorch datasets wrapping the synthetic floorplan generator.

Three task-specific datasets are exposed so each model can use whichever
annotations it needs:

    FloorplanClsDataset - returns (image, cls_label)
    FloorplanDetDataset - returns (image, {"labels": ..., "boxes": cxcywh})
    FloorplanSegDataset - returns (image, seg_mask)

All return images as float32 tensors in [0, 1] with shape (3, H, W).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .synthetic_floorplan import (
    FloorplanConfig, generate_floorplan,
    ROOM_TYPES, COMPONENT_TYPES, SEG_CLASSES,
)


def _to_tensor_rgb(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)


def _boxes_xyxy_to_cxcywh_norm(boxes: np.ndarray, size: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    b = boxes.astype(np.float32) / float(size)
    cx = 0.5 * (b[:, 0] + b[:, 2])
    cy = 0.5 * (b[:, 1] + b[:, 3])
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    return np.stack([cx, cy, w, h], axis=1)


# ---------------------------------------------------------------------------
# On-the-fly generator-backed dataset
# ---------------------------------------------------------------------------

class _BaseSynthetic(Dataset):
    def __init__(self, n: int, size: int = 256, seed: int = 0):
        self.n = n
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def _sample(self, idx: int) -> Dict:
        cfg = FloorplanConfig(size=self.size, seed=self.seed + idx)
        return generate_floorplan(cfg)


class FloorplanClsDataset(_BaseSynthetic):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._sample(idx)
        img = _to_tensor_rgb(s["image"])
        return img, torch.tensor(s["cls_label"], dtype=torch.long)


class FloorplanDetDataset(_BaseSynthetic):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = self._sample(idx)
        img = _to_tensor_rgb(s["image"])
        tgt = {
            "labels": torch.from_numpy(s["box_labels"]).long(),
            "boxes":  torch.from_numpy(
                _boxes_xyxy_to_cxcywh_norm(s["boxes"], self.size)
            ).float(),
        }
        return img, tgt


class FloorplanSegDataset(_BaseSynthetic):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._sample(idx)
        img = _to_tensor_rgb(s["image"])
        seg = torch.from_numpy(s["seg_mask"]).long()
        return img, seg


# ---------------------------------------------------------------------------
# Disk-backed variants (use after running scripts/generate_dataset.py)
# ---------------------------------------------------------------------------

class DiskFloorplanClsDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.items = sorted([p for p in self.root.iterdir() if p.is_dir()])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import json
        d = self.items[idx]
        img = np.array(Image.open(d / "image.png").convert("RGB"))
        meta = json.loads((d / "meta.json").read_text())
        return _to_tensor_rgb(img), torch.tensor(meta["cls_label"], dtype=torch.long)


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------

def collate_detection(batch):
    """DETR wants variable-length target lists, so we stack images only."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    tgts = [b[1] for b in batch]
    return imgs, tgts
