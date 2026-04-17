# Deep Dive: Ground Truth, Dataset, and the Three Models

This doc explains the *why* and *how* behind the project with small, concrete
examples you can trace through by hand. It's meant to be read alongside the
code in `src/`.

---

## 1. What is "Ground Truth" (GT)?

**Ground truth** is the *correct answer* attached to every training sample.
Models learn by comparing their predictions against the GT and minimising a
loss. At evaluation time, GT is also the yardstick for "did the model get
it right?".

In this project each synthetic floorplan produces **three** kinds of GT
because we're training three different models.

### 1.1 Classification GT — a single integer

For the ViT classifier, GT is just one integer per image: the index of the
dominant room type.

```python
ROOM_TYPES = ["office", "meeting_room", "corridor",
              "restroom", "lobby", "storage"]

# For this floorplan:        GT = 4   (meaning "lobby")
```

Think of it like a single-word label on a photograph.

### 1.2 Detection GT — a list of boxes

For DETR, GT is a variable-length list of `(class, bounding_box)` pairs.
An image with 5 doors and 2 windows has 7 entries.

```python
# Detection GT for one floorplan:
labels = [0, 0, 0, 1, 1, 3, 4]          # door, door, door, window, window, desk, toilet
boxes  = [(30, 12, 44, 22),              # xyxy in pixels (axis-aligned rectangles)
          (80, 12, 94, 22),
          (12, 60, 22, 74),
          (126, 8, 146, 14),
          (220, 120, 240, 126),
          (40, 70, 80, 90),
          (180, 200, 194, 216)]
```

Training code uses the normalised `(cx, cy, w, h)` format — centre-x,
centre-y, width, height, all divided by the image side — because that's
what the DETR loss expects.

### 1.3 Segmentation GT — a per-pixel mask

For Mask2Former, GT is an image-sized array where every pixel carries a
class index.

```python
SEG_CLASSES = ["background", "floor", "wall", "door", "window"]

# A tiny 6x6 toy mask around a single door in a wall:
#
#   0 0 0 0 0 0     <- outside
#   0 1 1 1 1 0     <- floor
#   0 1 3 3 1 0     <- door in middle of a wall
#   0 1 3 3 1 0
#   0 1 1 1 1 0
#   0 0 0 0 0 0
```

For the full 256×256 floorplans, the mask shape is `(256, 256)`.

### 1.4 Why three kinds of GT?

The three tasks answer three different questions:

| Task | Question it answers | GT shape |
|---|---|---|
| Classification | *"What kind of place is this?"* | one int |
| Detection | *"Where are the specific things?"* | list of boxes |
| Segmentation | *"Which pixel belongs to what?"* | (H, W) int map |

Each model has a loss function tailored to the GT shape.

---

## 2. The Dataset

### 2.1 Where it comes from

`src/data/synthetic_floorplan.py` generates plans **procedurally** — there
is no downloading involved. Each call to `generate_floorplan(cfg)` produces
one fully-annotated sample, deterministic given `cfg.seed`.

### 2.2 How a plan is built, step by step

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

### 2.3 What every sample actually contains

```python
sample = generate_floorplan(FloorplanConfig(size=256, seed=7))

sample["image"]      # ndarray (256, 256, 3) uint8     - the RGB plan view
sample["seg_mask"]   # ndarray (256, 256)    int64     - the seg GT
sample["boxes"]      # ndarray (N, 4)        float32   - xyxy pixel boxes
sample["box_labels"] # ndarray (N,)          int64     - COMPONENT_TYPES idx
sample["cls_label"]  # int                             - ROOM_TYPES idx
sample["rooms"]      # list[Room]                      - debug info
```

### 2.4 Rough distribution (from 100 samples)

Measured on `seed=0..99`, computed in `outputs/samples/dataset_stats.png`:

- Room types: corridors 42%, lobbies 24%, offices 10%, meeting rooms 9%,
  storage 9%, restrooms 6%.
- Components per image (avg): ~9 doors, ~8 windows, ~3 desks, ~2 toilets,
  <1 stair.
- Pixel share: floor ~74%, wall ~16%, background ~6%, door ~3%, window <1%.

Why it matters: windows and stairs are *rare classes*. That's realistic —
in real BIM data, structural components vastly outnumber occasional
features — and it's exactly why DETR's `eos_coef` down-weights the
"no-object" class, and why segmentation mIoU is computed per-class.

### 2.5 Drop-in replacement with real data

The synthetic dataset is just a `torch.utils.data.Dataset` that returns
`(image, target)`. Swap it with:

- **CubiCasa5K** — 5 000 real floorplans with room + icon annotations.
- **R2V (raster-to-vector)** — simpler floorplans.
- **IFC → raster** — use `ifcopenshell` to render plan views from real
  BIM files.

The three models need no changes.

---

## 3. The Models — simple explanations + differences

All three use **transformers**, which boil down to one idea:
*every output token attends to every input token and computes a weighted
sum*. The differences are (a) what the tokens represent and (b) what the
final heads do with them.

### 3.1 ViT — "What kind of room is this?"

Input  → one image
Output → one class label

#### Idea

Chop the image into non-overlapping patches, treat each patch as a word,
run a transformer over the "sentence" of patches, and use a special
`[CLS]` token's final embedding to predict the class.

#### Tiny example (4×4 image, 2×2 patches)

```
Image                Patch grid         Tokens
┌─┬─┐  ┌─┬─┐         ┌─┬─┐              [CLS]  P1  P2  P3  P4
│A│B│  │A│B│         │1│2│     ----->    |     |   |   |   |
├─┼─┤  ├─┼─┤         ├─┼─┤               +--- transformer ---+
│C│D│  │C│D│         │3│4│                        |
└─┴─┘  └─┴─┘         └─┴─┘                      class
```

We prepend a learnable `[CLS]` token, add positional embeddings, run N
transformer blocks, and feed the final `[CLS]` embedding into a linear
classifier.

#### In this project (defaults)

- Input 256×256, patch size 16 → **256 patches** + 1 CLS = 257 tokens
- Embedding dim 192, 6 layers, 4 heads → ~2.5 M params
- Loss: cross-entropy over 6 room types

#### When to use ViT

When you only care about one label per image. For a floorplan we use it
to tag the dominant room type; you could also use it to classify the
whole plan ("residential" vs "office") or to predict floor number from
the header region of a CAD drawing.

---

### 3.2 DETR — "Where are all the doors/windows/etc.?"

Input  → one image
Output → a fixed-length set of (class, bounding box) predictions

#### Idea

Replace the "anchors + NMS" machinery of traditional detectors with pure
set prediction: a transformer decoder emits N "object queries", each
responsible for at most one object. The Hungarian algorithm matches
queries to ground-truth boxes during training.

#### Pipeline

```
image ──► CNN backbone ──► feature map ──► transformer encoder
                                                 │
                         learnable queries (Q=50) │
                                   │              │
                                   ▼              ▼
                            transformer decoder (cross-attends)
                                   │
                              per-query heads
                                 │        │
                         ┌───────┘        └───────┐
                         ▼                        ▼
                   class logits (C+1)       box (cx,cy,w,h)
```

The extra class `+1` is **"no object"** — a query that shouldn't predict
anything is supervised to predict this class.

#### Hungarian matching — tiny example

Say the image has 2 doors and we emit 4 queries.

```
Queries:     q1            q2            q3            q4
Predict:     door@(10,20)  window@(60,8) door@(70,50)  door@(100,100)

Real boxes:  door@(12,22)  door@(72,52)
             (target #0)   (target #1)
```

The matcher computes a cost `C[query, target]` combining class
probability, L1 box distance, and GIoU, then runs
`scipy.optimize.linear_sum_assignment` to pick the best 1-to-1 matches:

```
Best matches found:
    q1 ─► target #0  (door at (12,22))
    q3 ─► target #1  (door at (72,52))
    q2 ─► no-object  (supervised as "window" prediction → wrong, penalised)
    q4 ─► no-object  (supervised as "door" prediction → wrong, penalised)
```

Then the loss is:
- Classification CE on all 4 queries
- Box L1 + GIoU only on the matched pair (q1→#0, q3→#1)

#### In this project (defaults)

- Small CNN backbone → feature map at stride /16 (so 256→16×16 tokens)
- 3 encoder + 3 decoder layers, 4 heads, d_model 128
- **50 object queries**
- Output: `{pred_logits: (B, 50, 6), pred_boxes: (B, 50, 4)}`

#### When to use DETR

When you need to localise **individual instances**. For BIM/CAD, DETR
finds every door, window, stair, desk, toilet and returns their
coordinates — exactly what a routing graph needs (one edge per door).

---

### 3.3 Mask2Former — "Which pixel belongs to what?"

Input  → one image
Output → a per-pixel class map (semantic segmentation)

#### Idea

Same query-based spirit as DETR, but each query now predicts a **mask**
over the whole image instead of a box. For semantic segmentation, we
aggregate the per-query masks into per-class masks.

#### Pipeline

```
image ──► CNN backbone (multi-scale)
            │
            ├─► pixel decoder (FPN-like)  ──► feature map F  (D, H/4, W/4)
            │
            ▼
   transformer decoder with Q learnable queries
            │
            ├── class head:   query → (C+1) class logits
            └── mask head:    query → D-dim mask embedding  e_q

Mask for query q:   M_q(h, w) = sigmoid(  e_q · F[:, h, w] )
Semantic seg map:   Seg(c, h, w) = Σ_q  softmax(class_q)[c] · M_q(h, w)
```

The key trick is the dot-product between a query embedding and each
pixel's feature vector — turning a learned query into "where on the
image does this pattern appear?".

#### Tiny example — one query predicting the "wall" mask

Imagine a 2×2 feature map `F` after the pixel decoder:

```
F[:, 0, 0] = [0.1, 0.9, 0.2]     (feature vector at top-left pixel)
F[:, 0, 1] = [0.8, 0.1, 0.1]     (top-right)
F[:, 1, 0] = [0.7, 0.0, 0.2]     (bottom-left)
F[:, 1, 1] = [0.0, 0.9, 0.1]     (bottom-right)
```

One query learns the embedding `e_wall = [1.0, -1.0, 0.0]`. The mask is
the dot-product at every pixel:

```
raw mask  = [[0.1*1 + 0.9*-1,   0.8*1 + 0.1*-1],
             [0.7*1 + 0.0*-1,   0.0*1 + 0.9*-1]]
          = [[-0.8, 0.7],
             [ 0.7, -0.9]]

sigmoid   ≈ [[0.31, 0.67],
             [0.67, 0.29]]
```

So this query "thinks" the top-right and bottom-left pixels are walls.
A different query can learn the "door" pattern, and the final per-class
semantic map sums up all queries weighted by their class probabilities.

#### In this project (defaults)

- Multi-scale backbone with strides /4, /8, /16
- Pixel decoder outputs a feature map at /4 resolution (64×64 for a 256×256 input)
- 3 decoder layers, 4 heads, d_model 128
- **20 object queries**
- Loss: cross-entropy + soft Dice on the aggregated per-class map

#### When to use Mask2Former

When you need **pixel-precise boundaries** — walls, floors, open space.
Bounding boxes aren't enough for wall extraction because walls follow
irregular outlines. Mask2Former's mask head gives you the exact polygon.

---

## 4. Side-by-side comparison

| | **ViT** | **DETR-lite** | **Mask2Former-lite** |
|---|---|---|---|
| Question it answers | "What kind of room?" | "Where are doors/windows/…?" | "Which pixel = wall/floor/door?" |
| Input | image | image | image |
| Output | 1 class | N (class, box) pairs | (H, W) class map |
| Backbone | patch embed only | small CNN | multi-scale CNN |
| Transformer | encoder only | encoder + decoder | decoder only |
| "Queries" concept | one `[CLS]` token | 50 object queries | 20 mask queries |
| Matching | none (single label) | Hungarian | set-prediction (simplified to per-class supervision here) |
| Loss | CE | CE + L1 + GIoU | CE + Dice |
| Typical output | `"office"` | `[(door, box), (door, box), …]` | `array(H, W)` of class IDs |
| Output per-image size | 1 int | up to 50 boxes | 65,536 ints (256×256) |
| Params in this repo | ~2.5 M | ~3.5 M | ~2.8 M |
| Scales with N components | no | yes (via queries) | no (always per-pixel) |

### 4.1 Intuitive analogy

- **ViT** is like a librarian who reads an entire book and says *"this is
  a mystery novel"*.
- **DETR** is like a bouncer with 50 clipboards; each clipboard is told
  to look for one object; most go home empty, a few hand in a name and
  a location.
- **Mask2Former** is like 20 spotlights, each trained on a different
  concept ("wall", "door"…); each spotlight shines on the pixels it
  recognises, then all spotlights combine into a coloured floor plan.

### 4.2 How they complement each other

A full BIM-to-indoor-map pipeline might use:

1. **ViT** to tag each room's function (POI metadata).
2. **DETR** to find doors and stairs (graph edges for routing).
3. **Mask2Former** to get clean wall/floor polygons (the navmesh).

Each model specialises; together they cover the three levels of
understanding needed to automate BIM-to-map conversion.

---

## 5. Concrete numbers you can expect

Running the default configs on CPU for 3 epochs on 200 training samples:

| Metric | Approximate value |
|---|---|
| ViT val accuracy | 60–80% |
| DETR total loss | starts ~12, drops to ~3–4 |
| Mask2Former val mIoU | 0.35–0.55 |

These are "does the pipeline learn anything" numbers — not production
metrics. With a larger backbone (`timm` Swin), real data (CubiCasa5K),
longer training, and proper augmentation, each model reaches
production-grade quality.

---

## 6. FAQ

**Q. Why not use one model for all three tasks?**
Mask2Former can do panoptic segmentation (instances + semantics) and in
principle you could add a classification head on top, but specialised
models are easier to train, debug and evaluate. This repo keeps them
separate so each is small and pedagogically clear.

**Q. Why is DETR's "no object" class weighted down (`eos_coef = 0.1`)?**
Because most of the 50 queries should predict "no object" — without
down-weighting, the model would collapse to always predicting "no
object" since that's the cheapest answer.

**Q. Why is the pixel decoder at /4 resolution instead of full?**
Memory. Full-resolution attention would be quadratic in the pixel count.
Mask2Former up-samples back to full resolution at the end.

**Q. Can I train these on a GPU?**
Yes — the `device` check in every script automatically uses CUDA if
available. Default batch sizes are sized for CPU; bump `--bs` to 64+ on
GPU.
