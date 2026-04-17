"""Train Mask2Former-lite for semantic segmentation of walls / floors / doors / windows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.data import SEG_CLASSES
from src.data.dataset import FloorplanSegDataset
from src.models.mask2former_lite import Mask2FormerLite, mask2former_loss


def _mean_iou(logits: torch.Tensor, target: torch.Tensor, n_classes: int
              ) -> float:
    """logits: (B, C, Hp, Wp) up-sampled to target shape; target: (B, H, W)."""
    pred = F.interpolate(logits, size=target.shape[-2:], mode="bilinear",
                         align_corners=False).argmax(1)
    ious = []
    for c in range(n_classes):
        p = (pred == c); t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / max(len(ious), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int, default=3)
    p.add_argument("--bs",      type=int, default=8)
    p.add_argument("--lr",      type=float, default=2e-4)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val",   type=int, default=40)
    p.add_argument("--size",    type=int, default=256)
    p.add_argument("--ckpt",    type=str, default="outputs/checkpoints/m2f.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_ds = FloorplanSegDataset(n=args.n_train, size=args.size, seed=0)
    val_ds   = FloorplanSegDataset(n=args.n_val,   size=args.size, seed=100_000)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=0)

    n_classes = len(SEG_CLASSES)
    model = Mask2FormerLite(n_classes=n_classes).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        running, n = 0.0, 0
        for imgs, seg in train_dl:
            imgs, seg = imgs.to(device), seg.to(device)
            out = model(imgs)
            losses = mask2former_loss(out, seg, n_classes=n_classes)
            opt.zero_grad(); losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += losses["loss"].item() * imgs.size(0); n += imgs.size(0)

        model.eval()
        iou_sum, nb = 0.0, 0
        with torch.no_grad():
            for imgs, seg in val_dl:
                imgs, seg = imgs.to(device), seg.to(device)
                out = model(imgs)
                iou_sum += _mean_iou(out["seg_logits"], seg, n_classes)
                nb += 1

        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={running/max(n,1):.3f}  "
              f"val_mIoU={iou_sum/max(nb,1):.3f}  "
              f"time={time.time()-t0:.1f}s")

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": SEG_CLASSES}, args.ckpt)
    print(f"saved -> {args.ckpt}")


if __name__ == "__main__":
    main()
