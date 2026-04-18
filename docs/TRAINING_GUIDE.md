# Training Guide

End-to-end instructions for running the three models, plus expected
behaviour and troubleshooting.

---

## 1. One-time setup

```bash
cd BIM_CAD_Project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `numpy`, `Pillow`, `matplotlib`,
`scipy` (for the Hungarian matcher), `tqdm`.

---

## 2. Generate the synthetic dataset

```bash
python scripts/generate_dataset.py --n-train 200 --n-val 40 --size 256
```

This writes:

```
outputs/samples/train/<k>/image.png
outputs/samples/train/<k>/seg.npy
outputs/samples/train/<k>/boxes.npz
outputs/samples/train/<k>/meta.json
outputs/samples/val/...
outputs/samples/preview_grid.png
```

Alternatively, the three PyTorch datasets (`FloorplanClsDataset`, etc.)
generate samples on the fly — no disk pass needed. Use the on-the-fly
datasets for development and the disk dataset for reproducibility /
larger runs.

---

## 3. Train each model

### 3.1 ViT classifier

```bash
python -m src.training.train_vit \
    --epochs 5 --bs 16 --lr 3e-4 \
    --n-train 400 --n-val 80 \
    --ckpt outputs/checkpoints/vit.pt
```

Typical CPU timing: ~20–40 s per epoch on 200 samples.

### 3.2 DETR-lite detector

```bash
python -m src.training.train_detr \
    --epochs 10 --bs 8 --lr 1e-4 \
    --n-train 400 --n-val 80 \
    --ckpt outputs/checkpoints/detr.pt
```

Typical CPU timing: ~60–120 s per epoch on 200 samples.
DETR is slower per step because of the Hungarian matcher.

### 3.3 Mask2Former-lite segmenter

```bash
python -m src.training.train_m2f \
    --epochs 10 --bs 8 --lr 2e-4 \
    --n-train 400 --n-val 80 \
    --ckpt outputs/checkpoints/m2f.pt
```

Typical CPU timing: ~60–90 s per epoch.

---

## 4. Inspect predictions

Runs all three trained models on a handful of val samples and saves a
side-by-side comparison PNG.

```bash
python -m src.inference.run --n 8 \
    --vit  outputs/checkpoints/vit.pt \
    --detr outputs/checkpoints/detr.pt \
    --m2f  outputs/checkpoints/m2f.pt \
    --out  outputs/samples/predictions.png
```

Open `outputs/samples/predictions.png` to see: input | GT seg | DETR
predictions + ViT label | Mask2Former prediction.

---

## 5. Hyperparameter reference

### 5.1 Shared

| Flag | Meaning | Default |
|---|---|---|
| `--epochs` | Training epochs | 3 |
| `--bs` | Batch size | 8–16 |
| `--lr` | Peak learning rate | 1e-4 … 3e-4 |
| `--size` | Square image side | 256 |
| `--n-train` | Training samples | 200 |
| `--n-val` | Validation samples | 40 |
| `--ckpt` | Output path for `state_dict` | `outputs/checkpoints/<model>.pt` |

### 5.2 ViT (`src/models/vit.py`)

```python
ViTClassifier(
    img_size=256, patch_size=16,
    embed_dim=192, depth=6, n_heads=4, mlp_mult=4,
    dropout=0.1, n_classes=len(ROOM_TYPES),
)
```

### 5.3 DETR-lite (`src/models/detr_lite.py`)

```python
DETRLite(
    n_classes=len(COMPONENT_TYPES),
    d_model=128, n_heads=4,
    n_encoder=3, n_decoder=3, ff=512,
    n_queries=50, dropout=0.1,
)
```

Loss weights: `w_cls=1, w_bbox=5, w_giou=2, eos_coef=0.1`.

### 5.4 Mask2Former-lite (`src/models/mask2former_lite.py`)

```python
Mask2FormerLite(
    n_classes=len(SEG_CLASSES),
    d_model=128, n_heads=4, n_decoder=3,
    n_queries=20, ff=384, dropout=0.1,
)
```

Loss weights: `w_ce=1, w_dice=1`.

---

## 6. Expected curves

### 6.1 ViT

```
epoch 1  train_loss=1.7  val_acc=0.40
epoch 3  train_loss=0.9  val_acc=0.65
epoch 5  train_loss=0.5  val_acc=0.78
epoch 10 train_loss=0.2  val_acc=0.88
```

### 6.2 DETR

```
epoch 1  loss=11.8 (cls=1.9  bbox=1.2 giou=0.8)
epoch 3  loss=6.5  (cls=1.0  bbox=0.6 giou=0.5)
epoch 10 loss=3.2  (cls=0.4  bbox=0.3 giou=0.3)
```

### 6.3 Mask2Former

```
epoch 1  train_loss=1.5  val_mIoU=0.25
epoch 3  train_loss=0.8  val_mIoU=0.42
epoch 10 train_loss=0.4  val_mIoU=0.60
```

These are rough CPU-scale indicators. Real numbers depend on seed and
dataset size.

---

## 7. Troubleshooting

**Q. Loss is NaN / exploding.**
Gradient clipping is already enabled for DETR and Mask2Former. Try
lowering `--lr` by 2×.

**Q. ViT val accuracy is stuck at ~17% (random 1/6).**
Usually under-fitting. Try `--epochs 20` and more training samples.
If still stuck, check that positional embeddings were initialised (look
for `nn.init.trunc_normal_` on `pos_embed` in `src/models/vit.py`).

**Q. DETR predicts "no object" for everything.**
`eos_coef` might be too high. Try `eos_coef=0.05`. Also check that your
target box format is cxcywh normalised to [0, 1], not xyxy pixels.

**Q. Mask2Former output is entirely one class.**
Probably class imbalance. Increase the Dice weight or add per-class
weighted CE. You can also start from a checkpoint trained on more data.

**Q. Out of memory on a small GPU.**
Shrink `--bs`, reduce `d_model` in the model files, or switch to
`--size 192`.

**Q. I want GPU training.**
No change needed — all scripts auto-detect CUDA. Just bump `--bs` to
32–64 on a consumer GPU.

---

## 8. Where to go next

- Swap the synthetic dataset for **CubiCasa5K** or **IFC rasters**.
- Upgrade backbones to pretrained `timm` Swin / ConvNeXt.
- Add panoptic mask matching to Mask2Former.
- Post-process ViT + DETR + Mask2Former outputs into a **routable
  navmesh** (corridor skeletonisation + door connectivity graph).
