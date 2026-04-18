# Architecture

High-level view of how the code is organised and how data flows through
the pipeline at train and inference time.

---

## 1. Module layout

```
BIM_CAD_Project/
├── configs/default.yaml          # default hyperparameters
├── scripts/
│   └── generate_dataset.py       # writes annotated samples to disk
├── src/
│   ├── data/
│   │   ├── synthetic_floorplan.py   # procedural floorplan generator
│   │   ├── dataset.py               # PyTorch Datasets + collate fns
│   │   └── __init__.py              # public ontology (ROOM_TYPES, ...)
│   ├── models/
│   │   ├── vit.py                   # Vision Transformer
│   │   ├── detr_lite.py             # DETR + Hungarian matcher + loss
│   │   └── mask2former_lite.py      # Mask2Former + loss
│   ├── training/
│   │   ├── train_vit.py
│   │   ├── train_detr.py
│   │   └── train_m2f.py
│   └── inference/
│       ├── visualize.py             # overlays + prediction grids
│       └── run.py                   # run all three models
├── notebooks/walkthrough.ipynb
└── outputs/
    ├── samples/                     # PNG previews & disk dataset
    └── checkpoints/
```

Each file has one responsibility. Imports flow one way: `data → models →
training → inference`.

---

## 2. Core data contracts

### 2.1 Ontology (`src/data/__init__.py`)

```python
ROOM_TYPES      = ["office", "meeting_room", "corridor",
                   "restroom", "lobby", "storage"]         # ViT classes
COMPONENT_TYPES = ["door", "window", "stair",
                   "desk", "toilet"]                       # DETR classes
SEG_CLASSES     = ["background", "floor", "wall",
                   "door", "window"]                       # Mask2Former classes
```

These lists are **the single source of truth** for class indices across
the whole codebase.

### 2.2 Sample dict (`generate_floorplan` return value)

```python
{
    "image":      np.uint8  (H, W, 3),
    "seg_mask":   np.int64  (H, W),
    "boxes":      np.float32 (N, 4),    # xyxy in pixels
    "box_labels": np.int64   (N,),
    "cls_label":  int,
    "rooms":      list[Room],
}
```

### 2.3 PyTorch dataset outputs

| Dataset | `__getitem__` returns |
|---|---|
| `FloorplanClsDataset` | `(image: float32 (3,H,W), label: int)` |
| `FloorplanDetDataset` | `(image, {"labels": int[N], "boxes": cxcywh float[N,4]})` |
| `FloorplanSegDataset` | `(image, seg: int[H,W])` |

Detection uses a custom `collate_detection` to handle variable-length
target lists.

---

## 3. Data flow — training

```
           ┌──────────────────────┐
           │ synthetic_floorplan  │  (procedural, deterministic)
           └──────────┬───────────┘
                      │   sample dict
                      ▼
           ┌──────────────────────┐
           │  dataset.py (3 cls)  │  PyTorch Dataset wrappers
           └──────────┬───────────┘
                      │   (image, target)
                      ▼
           ┌──────────────────────┐
           │   DataLoader batch   │
           └──────────┬───────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   train_vit      train_detr   train_m2f
         │            │            │
         ▼            ▼            ▼
  outputs/checkpoints/{vit,detr,m2f}.pt
```

Each training script:

1. Builds Dataset + DataLoader.
2. Instantiates the model + optimiser (AdamW).
3. Loops epochs, computes task-specific loss, steps.
4. Evaluates a validation split.
5. Saves `{"model": state_dict, "classes": CLASS_LIST}` to disk.

---

## 4. Data flow — inference

```
 image ─┬─► ViTClassifier       ─► room-type label
        │
        ├─► DETRLite             ─► list[(class, box)]
        │      │
        │      └─► Hungarian-free post-processing
        │          (softmax + threshold + cxcywh→xyxy)
        │
        └─► Mask2FormerLite      ─► per-pixel class map
               │
               └─► F.interpolate to full resolution
```

`src/inference/run.py` orchestrates all three and writes a side-by-side
comparison PNG.

---

## 5. How the three models share design DNA

All three use:

- **LayerNorm + residual** transformer blocks.
- **Pre-norm** style (norm → attn → residual).
- **AdamW** with weight decay 0.05 (classifier) or 1e-4 (det/seg).
- **Gradient clipping** to norm 1.0 for detection and segmentation.
- **GroupNorm** in CNN backbones (stable across tiny batch sizes).

Where they diverge:

- ViT uses only an encoder (`[CLS]` gathers the answer).
- DETR uses encoder + decoder with **object queries** matched to GT.
- Mask2Former uses a decoder plus a separate pixel decoder (FPN) and
  computes masks via query-pixel dot products.

---

## 6. Extension points

### 6.1 Swap the dataset

Replace `FloorplanClsDataset` etc. with any `Dataset` that yields the
same contract. Good targets:

- **CubiCasa5K** — 5 000 real floorplans.
- **IFC rasters** — parsed with `ifcopenshell`.

### 6.2 Upgrade the backbone

In `src/models/detr_lite.py` or `mask2former_lite.py`, replace the small
CNN with a pretrained backbone:

```python
from timm import create_model
backbone = create_model("swin_tiny_patch4_window7_224",
                        features_only=True, pretrained=True)
```

Adjust channel counts in the pixel decoder / positional encoding.

### 6.3 Add panoptic matching

`mask2former_loss` currently supervises aggregated per-class masks. To
get instance-level room segmentation, add a Hungarian matcher over
`(class, mask)` pairs following the original Mask2Former paper.

### 6.4 Emit a routable navmesh

After running all three models, run:

1. Morphological skeletonisation on the corridor/floor mask →
   graph nodes.
2. DETR door boxes → graph edges connecting adjacent rooms.
3. ViT room labels → POI metadata for each graph node.

The result is a routable indoor map compatible with wayfinding engines.

---

## 7. Hyperparameter surface

All three trainers accept a minimal CLI:

| Flag | What it controls |
|---|---|
| `--epochs` | Training epochs |
| `--bs` | Batch size |
| `--lr` | Learning rate |
| `--n-train`, `--n-val` | Dataset sizes |
| `--size` | Square image side |
| `--ckpt` | Output checkpoint path |

Model-specific hyperparameters (depth, heads, queries) live in
`configs/default.yaml` and in the model `__init__` defaults.
