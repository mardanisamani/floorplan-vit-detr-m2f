"""Train DETR-lite on the synthetic floorplan dataset for component detection."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.data import COMPONENT_TYPES
from src.data.dataset import FloorplanDetDataset, collate_detection
from src.models.detr_lite import DETRLite, HungarianMatcher, detr_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int, default=3)
    p.add_argument("--bs",      type=int, default=8)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val",   type=int, default=40)
    p.add_argument("--size",    type=int, default=256)
    p.add_argument("--ckpt",    type=str, default="outputs/checkpoints/detr.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_ds = FloorplanDetDataset(n=args.n_train, size=args.size, seed=0)
    val_ds   = FloorplanDetDataset(n=args.n_val,   size=args.size, seed=100_000)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          collate_fn=collate_detection, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                          collate_fn=collate_detection, num_workers=0)

    model = DETRLite(n_classes=len(COMPONENT_TYPES)).to(device)
    matcher = HungarianMatcher()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "loss_cls": 0.0,
                   "loss_bbox": 0.0, "loss_giou": 0.0}
        n = 0
        for imgs, tgts in train_dl:
            imgs = imgs.to(device)
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            outs = model(imgs)
            losses = detr_loss(outs, tgts, matcher, n_classes=len(COMPONENT_TYPES))
            opt.zero_grad(); losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k in running:
                running[k] += losses[k].item() * imgs.size(0)
            n += imgs.size(0)

        avg = {k: v / max(n, 1) for k, v in running.items()}
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"loss={avg['loss']:.3f} (cls={avg['loss_cls']:.3f} "
              f"bbox={avg['loss_bbox']:.3f} giou={avg['loss_giou']:.3f})  "
              f"time={time.time()-t0:.1f}s")

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": COMPONENT_TYPES},
               args.ckpt)
    print(f"saved -> {args.ckpt}")


if __name__ == "__main__":
    main()
