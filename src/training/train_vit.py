"""Train the ViT room-type classifier on the synthetic floorplan dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.data import ROOM_TYPES
from src.data.dataset import FloorplanClsDataset
from src.models.vit import ViTClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int, default=3)
    p.add_argument("--bs",      type=int, default=16)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val",   type=int, default=40)
    p.add_argument("--size",    type=int, default=256)
    p.add_argument("--ckpt",    type=str, default="outputs/checkpoints/vit.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_ds = FloorplanClsDataset(n=args.n_train, size=args.size, seed=0)
    val_ds   = FloorplanClsDataset(n=args.n_val,   size=args.size, seed=100_000)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=0)

    model = ViTClassifier(img_size=args.size, n_classes=len(ROOM_TYPES)).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * imgs.size(0); n += imgs.size(0)

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={running/max(n,1):.4f}  "
              f"val_acc={correct/max(total,1):.3f}  "
              f"time={time.time()-t0:.1f}s")

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": ROOM_TYPES}, args.ckpt)
    print(f"saved -> {args.ckpt}")


if __name__ == "__main__":
    main()
