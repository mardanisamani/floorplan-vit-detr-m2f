# Dataset — Synthetic CAD/BIM Floorplans

## 1. Where it comes from

`src/data/synthetic_floorplan.py` generates plans **procedurally** — there
is no downloading involved. Each call to `generate_floorplan(cfg)` produces
one fully-annotated sample, deterministic given `cfg.seed`.

---

## 2. How a plan is built, step by step

1. **Start with a building footprint** (a big rectangle).
2. **Recursively split it** using binary-space-partitioning (BSP).
   Each step picks a random axis and a random cut that respects a
   `min_room` side length.
3. **Label each leaf** with a room type (the biggest becomes the lobby,
   long thin ones become corridors, the rest are sampled).
4. **Paint floors** in room-type-specific colours.
5. **Paint outer walls** around the footprint and **interior walls**
   between neighbouring rooms.
6. **Insert doors** on shared interior walls.
7. **Insert windows** on outer walls.
8. **Place furniture** — desks in offices, toilets in restrooms, a
   staircase in the lobby.

Tiny BSP example — splitting a 4×4 footprint with one cut at `x=2`:

```
Before:                 After a vertical split at x=2:
+-------+               +---+---+
|       |               | A | B |
|       |    ----->     |   |   |
|       |               |   |   |
+-------+               +---+---+
```

The real generator does this recursively up to `max_depth` levels, so a
typical 256×256 plan ends up with 4–10 rooms.

---

## 3. What every sample actually contains

```python
sample = generate_floorplan(FloorplanConfig(size=256, seed=7))

sample["image"]      # ndarray (256, 256, 3) uint8     - the RGB plan view
sample["seg_mask"]   # ndarray (256, 256)    int64     - the seg GT
sample["boxes"]      # ndarray (N, 4)        float32   - xyxy pixel boxes
sample["box_labels"] # ndarray (N,)          int64     - COMPONENT_TYPES idx
sample["cls_label"]  # int                             - ROOM_TYPES idx
sample["rooms"]      # list[Room]                      - debug info
```

---

## 4. Ontology

```python
ROOM_TYPES      = ["office", "meeting_room", "corridor",
                   "restroom", "lobby", "storage"]
COMPONENT_TYPES = ["door", "window", "stair", "desk", "toilet"]
SEG_CLASSES     = ["background", "floor", "wall", "door", "window"]
```

- `ROOM_TYPES` feeds the ViT classifier.
- `COMPONENT_TYPES` feeds the DETR detector.
- `SEG_CLASSES` feeds the Mask2Former segmenter.

---

## 5. Rough distribution (from 100 samples)

Measured on `seed=0..99`, computed in `outputs/samples/dataset_stats.png`:

- Room types: corridors 42%, lobbies 24%, offices 10%, meeting rooms 9%,
  storage 9%, restrooms 6%.
- Components per image (avg): ~9 doors, ~8 windows, ~3 desks, ~2 toilets,
  <1 stair.
- Pixel share: floor ~74%, wall ~16%, background ~6%, door ~3%, window <1%.

**Why it matters:** windows and stairs are *rare classes*. That's
realistic — in real BIM data, structural components vastly outnumber
occasional features — and it's exactly why DETR's `eos_coef`
down-weights the "no-object" class, and why segmentation mIoU is
computed per-class.

---

## 6. Drop-in replacement with real data

The synthetic dataset is just a `torch.utils.data.Dataset` that returns
`(image, target)`. Swap it with:

- **CubiCasa5K** — 5 000 real floorplans with room + icon annotations.
- **R2V (raster-to-vector)** — simpler floorplans.
- **IFC → raster** — use `ifcopenshell` to render plan views from real
  BIM files.

The three models need no changes.

---

## 7. Pre-generated previews

The following PNGs live in `outputs/samples/`:

- `preview_grid.png` — 9 example floorplans
- `annotations_preview.png` — ground-truth segmentation + detection overlay
- `dataset_stats.png` — class distribution bar charts

Regenerate at any scale with:

```bash
python scripts/generate_dataset.py --n-train 200 --n-val 40 --size 256
```
