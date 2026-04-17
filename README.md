# BIM/CAD Transformer Pipeline (ViT + DETR + Mask2Former)

A compact, fully-runnable reference implementation of a transformer-based
computer-vision pipeline for **object detection, semantic segmentation and
room-type classification on CAD / BIM floorplan data**.

This mirrors the kind of automated building-component extraction that
[MapsPeople](https://www.mapspeople.com/) uses to convert BIM/CAD drawings
into indoor maps for wayfinding and spatial analytics.

---

## 1. Why transformers for BIM/CAD?

MapsPeople turns raw floorplans into structured indoor maps. The pipeline
needs three levels of understanding, all of which transformers handle well:

| Task | Why transformers | Model used here |
|---|---|---|
| Classify room function (office, corridor, restroom…) | Long-range attention captures global layout | **ViT** |
| Detect components (doors, windows, stairs, furniture) | Set-prediction removes NMS heuristics | **DETR-lite** |
| Segment walls, floors, rooms | Unified mask-transformer decoder | **Mask2Former-lite** |

A CNN backbone still underlies each model, but the heads are pure
transformers — the same family MapsPeople describes in their
AI-for-BIM work.

---

## 2. Project layout

```
BIM_CAD_Project/
├── README.md                       <- this file
├── requirements.txt
├── configs/default.yaml
├── scripts/
│   └── generate_dataset.py         <- one-shot dataset creation
├── src/
│   ├── data/
│   │   ├── synthetic_floorplan.py  <- procedural CAD/BIM generator
│   │   └── dataset.py              <- PyTorch Datasets (cls / det / seg)
│   ├── models/
│   │   ├── vit.py                  <- Vision Transformer classifier
│   │   ├── detr_lite.py            <- DETR-style set-prediction detector
│   │   └── mask2former_lite.py     <- Mask2Former-style segmenter
│   ├── training/
│   │   ├── train_vit.py
│   │   ├── train_detr.py
│   │   └── train_m2f.py
│   └── inference/
│       └── visualize.py
├── notebooks/
│   └── walkthrough.ipynb           <- end-to-end demo
└── outputs/
    ├── samples/                    <- example floorplans (pre-generated)
    └── checkpoints/
```

---

## 3. Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset (writes to outputs/samples/)
python scripts/generate_dataset.py --n-train 200 --n-val 40 --size 256

# 3. Train each model (each takes a few minutes on CPU for a demo run)
python -m src.training.train_vit   --epochs 3
python -m src.training.train_detr  --epochs 3
python -m src.training.train_m2f   --epochs 3

# 4. Visualize predictions
python -m src.inference.visualize --split val --n 8
```

Or run `notebooks/walkthrough.ipynb` for the interactive version.

---

## 4. Dataset — synthetic CAD/BIM floorplans

`src/data/synthetic_floorplan.py` procedurally generates building floorplans
with ground-truth annotations for all three tasks. Each sample produces:

* A rendered RGB plan view (PNG).
* **Classification label** — dominant room type.
* **Detection labels** — axis-aligned bounding boxes for *door, window,
  stair, desk, toilet*.
* **Segmentation mask** — per-pixel class (*wall, floor, room, door, window*).

Rooms are laid out with a recursive binary-space-partitioning splitter,
walls are drawn with thick strokes, doors/windows are inserted on shared
walls, and furniture is placed room-type-appropriately (desks in offices,
toilets in restrooms, etc.). Everything is seeded — regenerating is
deterministic.

> Why synthetic? Real IFC/CAD data is licence-restricted. A procedural
> generator is free, scalable, mirrors the component ontology MapsPeople
> cares about, and can be replaced by a real dataset (CubiCasa5K, R2V,
> or parsed IFC) simply by swapping the `Dataset` class.

---

## 5. Models

### 5.1 ViT classifier (`src/models/vit.py`)

Standard patch-embedding Vision Transformer
([Dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929)).
Configurable depth / heads; defaults tuned for 256×256 floorplans
on CPU (6 layers, 4 heads, 16×16 patches → ~2.5 M params).

### 5.2 DETR-lite (`src/models/detr_lite.py`)

[DETR](https://arxiv.org/abs/2005.12872) with a small ResNet-style
backbone, 6 encoder + 6 decoder layers, and 50 object queries.
Uses the Hungarian matcher + generalized-IoU box loss from the
original paper, implemented compactly for clarity.

### 5.3 Mask2Former-lite (`src/models/mask2former_lite.py`)

A minimal variant of
[Mask2Former](https://arxiv.org/abs/2112.01527): pixel decoder
produces multi-scale features, a masked-attention transformer
decoder emits N query embeddings, each query predicts *(class, mask)*.
Ideal for MapsPeople-style unified wall / room / floor segmentation.

---

## 6. How this aligns with MapsPeople

MapsPeople converts BIM/CAD to indoor maps via their MapsIndoors
platform. Their automation pipeline needs to:

1. **Parse a plan** → ingest a floorplan image or IFC slice.
2. **Classify spaces** → determine each room's function for POI tagging.
3. **Detect components** → doors (for routing edges), windows (for visual
   fidelity), stairs/elevators (for level transitions).
4. **Segment the envelope** → separate walls, floors, open space for
   the routable network.

This project mirrors those four steps with open, reproducible code:

| MapsPeople step | This repo |
|---|---|
| Space classification | `ViT` on room crops |
| Component detection | `DETR-lite` on full plan |
| Wall / floor / room segmentation | `Mask2Former-lite` |
| Route graph extraction | *(future work — post-process segmentation + detection into a navmesh)* |

---

## 7. Extending to real BIM data

* Replace `SyntheticFloorplanDataset` with a dataset wrapping **CubiCasa5K**,
  **R2V** or **IFC → raster** conversions (using `ifcopenshell`).
* Swap the tiny backbones for pretrained Swin / ConvNeXt checkpoints
  from `timm`.
* For large plans, tile to 512×512 and stitch segmentation outputs.

---

## 8. Citations

* Dosovitskiy et al., *An Image is Worth 16×16 Words* (ICLR 2021).
* Carion et al., *End-to-End Object Detection with Transformers* (ECCV 2020).
* Cheng et al., *Masked-attention Mask Transformer for Universal Image
  Segmentation* (CVPR 2022).
* Kalervo et al., *CubiCasa5K* (2019) — real floorplan dataset.
