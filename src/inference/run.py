"""Run all three trained models on a handful of val samples and save overlays.

Usage:
    python -m src.inference.run --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.data import ROOM_TYPES, COMPONENT_TYPES, SEG_CLASSES
from src.data.dataset import (
    FloorplanClsDataset, FloorplanDetDataset, FloorplanSegDataset,
)
from src.models.vit import ViTClassifier
from src.models.detr_lite import DETRLite
from src.models.mask2former_lite import Mask2FormerLite
from src.inference.visualize import overlay_segmentation, draw_boxes


def _load(path, model):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model.eval()


def _cxcywh_to_xyxy_pixels(box_norm, size):
    cx, cy, w, h = box_norm.unbind(-1)
    x0 = (cx - 0.5 * w) * size
    y0 = (cy - 0.5 * h) * size
    x1 = (cx + 0.5 * w) * size
    y1 = (cy + 0.5 * h) * size
    return torch.stack([x0, y0, x1, y1], dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=8)
    p.add_argument("--size",   type=int, default=256)
    p.add_argument("--vit",    default="outputs/checkpoints/vit.pt")
    p.add_argument("--detr",   default="outputs/checkpoints/detr.pt")
    p.add_argument("--m2f",    default="outputs/checkpoints/m2f.pt")
    p.add_argument("--out",    default="outputs/samples/predictions.png")
    p.add_argument("--conf",   type=float, default=0.5)
    args = p.parse_args()

    ds_cls = FloorplanClsDataset(n=args.n, size=args.size, seed=100_000)
    ds_det = FloorplanDetDataset(n=args.n, size=args.size, seed=100_000)
    ds_seg = FloorplanSegDataset(n=args.n, size=args.size, seed=100_000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = _load(args.vit,  ViTClassifier(img_size=args.size,
                                         n_classes=len(ROOM_TYPES))).to(device)
    detr = _load(args.detr, DETRLite(n_classes=len(COMPONENT_TYPES))).to(device)
    m2f = _load(args.m2f,  Mask2FormerLite(n_classes=len(SEG_CLASSES))).to(device)

    cols = 4
    fig, axes = plt.subplots(args.n, cols, figsize=(cols * 3, args.n * 3))

    with torch.no_grad():
        for i in range(args.n):
            img_t, cls_gt = ds_cls[i]
            _, det_gt = ds_det[i]
            _, seg_gt = ds_seg[i]
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Classification
            logits = vit(img_t.unsqueeze(0).to(device))
            pred_cls = int(logits.argmax(1).item())

            # Detection
            det_out = detr(img_t.unsqueeze(0).to(device))
            prob = det_out["pred_logits"].softmax(-1)[0]
            scores = prob[:, :-1].max(-1).values
            labels = prob[:, :-1].argmax(-1)
            keep = scores > args.conf
            pred_boxes_xyxy = _cxcywh_to_xyxy_pixels(
                det_out["pred_boxes"][0][keep], args.size).cpu().numpy()
            pred_labels_arr = labels[keep].cpu().numpy()
            pred_scores_arr = scores[keep].cpu().numpy()

            # Segmentation
            seg_out = m2f(img_t.unsqueeze(0).to(device))["seg_logits"]
            seg_up = F.interpolate(seg_out, size=(args.size, args.size),
                                   mode="bilinear", align_corners=False)
            pred_seg = seg_up.argmax(1)[0].cpu().numpy()

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(
                f"Input\nGT cls = {ROOM_TYPES[cls_gt.item()]}", fontsize=9)
            axes[i, 1].imshow(overlay_segmentation(img_np, seg_gt.numpy()))
            axes[i, 1].set_title("GT seg", fontsize=9)
            axes[i, 2].imshow(draw_boxes(img_np, pred_boxes_xyxy,
                                         pred_labels_arr,
                                         scores=pred_scores_arr))
            axes[i, 2].set_title(
                f"DETR pred\nViT = {ROOM_TYPES[pred_cls]}", fontsize=9)
            axes[i, 3].imshow(overlay_segmentation(img_np, pred_seg))
            axes[i, 3].set_title("Mask2Former pred", fontsize=9)
            for c in range(cols):
                axes[i, c].axis("off")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
