"""
Generate a synthetic CAD/BIM floorplan dataset and save to disk.

Usage:
    python scripts/generate_dataset.py --n-train 200 --n-val 40 --size 256

Creates:
    outputs/samples/train/<k>/image.png
    outputs/samples/train/<k>/seg.npy
    outputs/samples/train/<k>/boxes.npz       (boxes + labels)
    outputs/samples/train/<k>/meta.json       (cls_label, components list)
    outputs/samples/preview_grid.png          (visual sanity check)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic_floorplan import (  # noqa: E402
    FloorplanConfig,
    generate_floorplan,
    ROOM_TYPES,
    COMPONENT_TYPES,
    SEG_CLASSES,
)


def save_sample(out_dir: Path, sample: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sample["image"]).save(out_dir / "image.png")
    np.save(out_dir / "seg.npy", sample["seg_mask"].astype(np.int16))
    np.savez(out_dir / "boxes.npz",
             boxes=sample["boxes"].astype(np.float32),
             labels=sample["box_labels"].astype(np.int64))
    meta = {
        "cls_label": int(sample["cls_label"]),
        "cls_name": ROOM_TYPES[int(sample["cls_label"])],
        "n_components": int(sample["boxes"].shape[0]),
        "rooms": [
            {"type": r.room_type, "x0": int(r.x0), "y0": int(r.y0),
             "x1": int(r.x1), "y1": int(r.y1)}
            for r in sample["rooms"]
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _render_preview_grid(samples: list, path: Path, n: int = 9) -> None:
    import matplotlib.pyplot as plt
    n = min(n, len(samples))
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            s = samples[i]
            ax.imshow(s["image"])
            ax.set_title(f"cls={ROOM_TYPES[s['cls_label']]}  "
                         f"n_boxes={s['boxes'].shape[0]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val",   type=int, default=40)
    p.add_argument("--size",    type=int, default=256)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--out-root", type=str,
                   default=str(ROOT / "outputs" / "samples"))
    args = p.parse_args()

    out_root = Path(args.out_root)
    preview_samples = []

    for split, n in [("train", args.n_train), ("val", args.n_val)]:
        base_seed = args.seed + (0 if split == "train" else 100_000)
        for i in range(n):
            cfg = FloorplanConfig(size=args.size, seed=base_seed + i)
            s = generate_floorplan(cfg)
            save_sample(out_root / split / f"{i:05d}", s)
            if split == "train" and i < 9:
                preview_samples.append(s)
        print(f"[{split}] wrote {n} samples")

    _render_preview_grid(preview_samples, out_root / "preview_grid.png", n=9)
    print(f"preview -> {out_root / 'preview_grid.png'}")


if __name__ == "__main__":
    main()
