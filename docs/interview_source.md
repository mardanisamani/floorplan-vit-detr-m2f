---
title: "BIM/CAD Transformer Pipeline — Interview Preparation"
subtitle: "Mapping the project to MapsPeople's Job Description"
author: "Sara"
date: "April 2026"
---

# 1. Executive Summary

This document maps a personal research project — a transformer-based
computer-vision pipeline that performs **object detection, semantic
segmentation, and classification on CAD / BIM floorplans** — onto the
MapsPeople Machine-Learning / Computer-Vision Engineer role.

The project demonstrates the core MapsPeople workflow end-to-end:
automated extraction of building components from BIM/CAD drawings so
they can be turned into routable indoor maps.

Three transformer-based models are trained on a procedurally-generated
synthetic floorplan dataset:

- **ViT** — classifies the dominant room type (office, corridor, lobby…).
- **DETR-lite** — detects building components (doors, windows, stairs,
  desks, toilets) as bounding boxes.
- **Mask2Former-lite** — segments the building envelope pixel-by-pixel
  (walls, floors, doors, windows).

Combined, the three outputs produce all the information a MapsIndoors-style
navigation engine needs: room labels for POIs, door positions for graph
edges, and wall polygons for the navmesh.

---

# 2. Why Three Models?

The question "why not one big model?" comes up constantly in interviews.
The short answer: **the three tasks answer three different questions and
have three different output shapes, and specialised models train faster,
debug easier, and deploy more flexibly.**

| Question | Output shape | Best-fit model family |
|---|---|---|
| What kind of place is this? | one integer | classification (ViT) |
| Where are the specific things? | list of (class, box) | detection (DETR) |
| Which pixel belongs to what? | (H, W) integer map | segmentation (Mask2Former) |

## 2.1 Could one model do it all?

Yes in principle — Mask2Former can even do **panoptic** segmentation
(instances + semantics). But in practice:

- **Training signal is cleaner when each model focuses on one loss.**
  Joint training requires careful loss balancing; a regression in one
  task can drag the others down.
- **Deployment is more flexible.** A customer who only needs room
  classification on pre-existing BIM doesn't pay for a full detector.
- **Teams can iterate independently.** At a company like MapsPeople,
  one engineer can improve the door detector while another swaps in a
  better segmentation backbone.

## 2.2 When would you merge them?

When the models share significant compute (e.g. a heavy Swin backbone)
and the deployment constraint is total latency, a unified
multi-task head on a shared backbone is worth considering. That is a
natural "phase 2" extension of this project.

---

# 3. The Dataset — Synthetic CAD/BIM Floorplans

## 3.1 Ontology

- `ROOM_TYPES`: office, meeting_room, corridor, restroom, lobby, storage
- `COMPONENT_TYPES`: door, window, stair, desk, toilet
- `SEG_CLASSES`: background, floor, wall, door, window

## 3.2 What a single sample contains

Every call to `generate_floorplan(cfg)` produces one sample:

    sample["image"]      -> (256, 256, 3) uint8   rendered plan view
    sample["seg_mask"]   -> (256, 256)    int64   per-pixel class
    sample["boxes"]      -> (N, 4)        float32 xyxy pixel boxes
    sample["box_labels"] -> (N,)          int64   component class IDs
    sample["cls_label"]  -> int                   dominant room type

Everything is seeded — regenerating with the same `cfg.seed` reproduces
the same floorplan exactly.

## 3.3 Why synthetic data?

- Real IFC/CAD floorplans are licence-restricted or proprietary.
- A procedural generator gives unlimited free samples with perfect
  ground truth, so the models can be stress-tested under known conditions.
- The dataset is a drop-in replacement interface — the same PyTorch
  `Dataset` subclass can later wrap CubiCasa5K or parsed IFC files,
  without touching the models.

---

# 4. Model 1 — ViT Classifier

## 4.1 What it does

Reads a floorplan image and predicts the **dominant room type** (one of
six classes).

## 4.2 Sample training example

Suppose the generator produces a floorplan that is mostly a lobby with
two small offices. The ground-truth label is:

    cls_label = ROOM_TYPES.index("lobby")   # = 4

Forward pass:

    image (1, 3, 256, 256)
       │  patch embed 16x16 → 256 patch tokens
       │  prepend [CLS] token            → 257 tokens, dim 192
       │  + positional embeddings
       │  6 × transformer blocks (self-attn + MLP)
       ▼
    [CLS] embedding (1, 192)
       │  linear head
       ▼
    logits (1, 6)   → softmax → probability per room type

Loss:

    loss = cross_entropy(logits, cls_label)   # scalar

One gradient step updates all 2.5 M parameters.

## 4.3 Advantages of the ViT approach

- **Global reasoning** — every patch attends to every other patch, so the
  classifier sees the whole layout in one step. CNNs need many layers to
  reach the same receptive field.
- **Scales with data** — ViTs are famously hungry for data but rewarding
  when it's available. MapsPeople owns a large proprietary BIM corpus,
  which is the regime ViTs shine in.
- **Easy to fine-tune** — swap the head, keep the body, and a pretrained
  ImageNet-21k ViT transfers surprisingly well to floorplans.

---

# 5. Model 2 — DETR-lite Detector

## 5.1 What it does

Emits up to 50 predictions per image, each of the form `(class, box)`.
Classes are the five component types; the special "no object" class is
used when a query shouldn't fire.

## 5.2 Sample training example

Suppose the image has three real objects:

    Real targets:
      t0: door   at (x=30, y=12, w=14, h=10)
      t1: door   at (x=80, y=12, w=14, h=10)
      t2: window at (x=12, y=60, w=10, h=14)

The model emits 50 query predictions. After the Hungarian matcher runs
(using a cost of class probability + L1 box distance + GIoU), the best
matches might be:

    q7  ↔ t0   (door, IoU 0.81)
    q22 ↔ t1   (door, IoU 0.74)
    q41 ↔ t2   (window, IoU 0.65)
    all other 47 queries → "no object" (supervised accordingly)

The total loss is:

    loss_cls  = weighted_CE(all 50 logits, matched labels or "no-obj")
    loss_bbox = L1(q7 box, t0 box) + L1(q22, t1) + L1(q41, t2)
    loss_giou = (1 - GIoU) summed over the same three matched pairs
    total     = 1·loss_cls + 5·loss_bbox + 2·loss_giou

The 47 unmatched queries only contribute through the classification
loss (down-weighted by `eos_coef = 0.1` so the model doesn't collapse).

## 5.3 Advantages of the DETR approach

- **End-to-end set prediction — no NMS.** Traditional detectors produce
  thousands of anchors and need non-maximum suppression with hand-tuned
  IoU thresholds. DETR's Hungarian matcher enforces 1-to-1 correspondence
  directly in the loss.
- **Cleaner training signal on sparse scenes.** Floorplans are sparse
  (tens of objects on a 65K-pixel image); set prediction is ideal.
- **Flexible head.** Adding a new component class (e.g. *elevator*) means
  bumping `n_classes` by one — no anchor redesign.

---

# 6. Model 3 — Mask2Former-lite Segmenter

## 6.1 What it does

Produces a full-resolution per-pixel class map where each pixel is
labelled with one of `background, floor, wall, door, window`.

## 6.2 Sample training example

Input image is 256×256. The ground-truth `seg_mask` has the same shape
and stores integers 0..4.

Forward pass:

    image (1, 3, 256, 256)
       │ multi-scale backbone → features at strides /4, /8, /16
       │ pixel decoder (FPN-like) → F (1, 128, 64, 64)
       │ 20 learnable queries → 3-layer transformer decoder → Q (1, 20, 128)
       │ class head:  Q → logits (1, 20, 6)
       │ mask  head:  Q → mask embeddings (1, 20, 128)
       │ per-query masks: einsum("bqd,bdhw->bqhw", mask_emb, F)
       ▼
    seg_logits (1, 5, 64, 64) = Σ_q softmax(class_q) · sigmoid(mask_q)

The loss compares `seg_logits` to the down-sampled GT mask:

    target_resized = interpolate(seg_mask, size=(64,64), mode='nearest')
    loss = cross_entropy(seg_logits, target_resized)
         + dice_loss(softmax(seg_logits), one_hot(target_resized))

## 6.3 Advantages of the Mask2Former approach

- **Pixel-precise polygons.** Walls curve, corridors bend, rooms are
  L-shaped. Bounding boxes can't express any of that; masks can.
- **Unified head for semantic / instance / panoptic.** The same
  query-based mask head can do all three tasks by changing only the
  matching rule. Future work: instance-level room segmentation.
- **Natural fit for the MapsPeople navmesh stage.** Once walls are
  masked out, skeletonisation of the free-space mask yields the
  corridor graph used for routing.

---

# 7. How the Three Models Fit Together

    ┌───────────────┐
    │  BIM / CAD    │  (IFC, DWG, PDF → raster)
    │  floorplan    │
    └──────┬────────┘
           │
   ┌───────┼────────────────────────────┐
   ▼       ▼                            ▼
  ViT    DETR                      Mask2Former
  (what)  (where)                   (which pixel)
   │       │                            │
   ▼       ▼                            ▼
  room    doors/windows/stairs       walls/floors/openings
  labels  with pixel coords          pixel masks
   │       │                            │
   └───┬───┘                            │
       │                                │
       ▼                                ▼
   POI metadata           ─┐     navmesh polygons ─┐
                           │                        │
   Doors as graph edges ───┘                        │
                                                    │
                                          ┌─────────┘
                                          ▼
                              Routable indoor map
                              (MapsIndoors-style)

Each specialised model contributes one pillar of the final indoor map;
none of them alone is sufficient.

---

# 8. Mapping to the MapsPeople JD

Below is how the project lines up against the typical bullet points of
a MapsPeople ML / CV Engineer job description. The exact wording
varies by posting, but the responsibilities recur.

| Typical JD bullet | How this project demonstrates it |
|---|---|
| Develop computer-vision models to extract information from BIM / CAD drawings | Three from-scratch CV models (ViT, DETR-lite, Mask2Former-lite) targeting exactly this problem |
| Work with transformer-based architectures | All three heads are transformers — self-attention, set-prediction, masked cross-attention |
| Build and maintain data pipelines for BIM / floorplan data | Procedural synthetic generator + PyTorch `Dataset` wrappers with a clear swap-in interface for CubiCasa5K / IFC |
| Implement object detection, semantic segmentation, classification | One model per task, each trained end-to-end |
| Prototype on small datasets and scale | Pipeline runs on CPU in minutes on 200 samples; all hyperparameters are exposed for scaling |
| Integrate outputs into downstream mapping products | Architecture doc explicitly maps outputs to navmesh / POI / routing-graph stages |
| Write clean, maintainable Python / PyTorch | ~1,200 LOC across `src/`, unit-testable functions, typed docstrings |
| Collaborate with product and engineering | Project ships with README, per-topic docs, a Jupyter walkthrough, and an architecture design doc |

## 8.1 Concrete talking points for the interview

- "I built a synthetic floorplan generator that produces every kind of
  annotation — classification, detection, and segmentation — so I could
  stress-test three specialised transformer models against known ground
  truth before moving to real IFC data."
- "I chose DETR over YOLO specifically because floorplan components are
  sparse and the Hungarian matcher removes anchor-tuning. That matters
  at MapsPeople where new component types (elevators, ramps) get added
  as customers ship new building types."
- "The project deliberately separates the three models even though
  Mask2Former could technically do all three. I did this for training
  stability and deployment flexibility — exactly the trade-off I'd
  expect to discuss during design review at MapsPeople."
- "The architecture doc shows how the three outputs feed into the
  downstream navmesh and POI graph — the step *after* CV, which is
  where MapsIndoors actually earns its value."

---

# 9. Anticipated Interview Questions & Answers

**Q1. Why transformers instead of CNNs on floorplans?**

Floorplan understanding depends on *global* structure: whether a room is
a corridor depends on the whole layout, not local textures. Self-attention
gives every patch immediate access to every other patch — a CNN needs a
deep stack to achieve the same receptive field. ViTs also scale well with
proprietary data, which MapsPeople has in abundance.

**Q2. How does Hungarian matching actually work?**

It's the standard `O(N³)` linear-sum assignment. We build a cost matrix
`C[query, target]` combining negative class probability, L1 box distance,
and negative GIoU, then `scipy.optimize.linear_sum_assignment` picks the
best 1-to-1 mapping. Queries that are not matched get supervised toward
the "no-object" class. This replaces the NMS heuristic entirely.

**Q3. Why is Mask2Former's decoder at /4 resolution?**

Memory. Self-attention is `O(N²)` in the number of tokens, and full-res
tokens would be 256×256 = 65 K per image. The pixel decoder upsamples
features to /4 (64×64 = 4 K tokens), which keeps memory bounded while
the final `F.interpolate` restores full resolution for evaluation.

**Q4. How would you extend this to real IFC/BIM?**

Three steps. (1) Use `ifcopenshell` to parse IFC files and render plan
views per storey. (2) Swap the synthetic `Dataset` for an IFC-backed one
that emits the same `(image, target)` contract. (3) Upgrade the
backbones to pretrained Swin / ConvNeXt from `timm` — the transformer
heads need no change.

**Q5. What's the weakest link in your current system?**

Segmentation of thin structures. Windows are <1 % of pixels and wall
outlines are one pixel thick at /4 resolution; plain cross-entropy is
dominated by floor pixels. Remedies: Dice loss (already included),
per-class weighted CE, a higher-resolution pixel decoder, or a dedicated
edge-loss term. This is exactly the kind of trade-off I'd expect to own
at MapsPeople.

**Q6. How do you handle the sim-to-real gap when moving from synthetic
to real floorplans?**

Three levers. (a) **Domain randomisation** — vary wall thickness, line
styles, furniture colours, fonts, noise, rotations during synthesis so
the model sees an over-broad distribution. (b) **Mixed training** — once
some labelled real data exists, train on both at a small ratio to anchor
the model. (c) **Self-training / pseudo-labels** — run the
synthetically-trained model on unlabelled real plans, keep the confident
outputs, and fine-tune.

**Q7. How would you evaluate this in production?**

Per-class IoU for segmentation, mAP for detection, top-1 accuracy for
classification — but these are not the product metrics. The product
metric is "does the generated indoor map route correctly?", which
requires end-to-end evaluation: generate POIs → build the graph →
simulate routing queries → compare against curated ground-truth paths.
I'd own both layers.

---

# 10. Personal Reflection

Building this project taught me three things I want to carry into the
MapsPeople role:

1. **Specialised models beat monolithic ones** when the tasks have
   different output shapes and different failure modes. Keeping them
   small and separate shortens the debug cycle.
2. **Data pipelines deserve the same care as models.** The procedural
   generator took longer to build than any individual model, and it is
   the component that most directly translates to real BIM work.
3. **Computer vision is upstream of the actual product.** The navmesh
   and POI graph are what MapsPeople sells; CV is the enabling layer.
   Designing the CV outputs with the downstream consumer in mind is how
   you ship something useful, not just something accurate.

---

# Appendix A — File Map

    BIM_CAD_Project/
    ├── src/
    │   ├── data/          synthetic_floorplan.py, dataset.py
    │   ├── models/        vit.py, detr_lite.py, mask2former_lite.py
    │   ├── training/      train_vit.py, train_detr.py, train_m2f.py
    │   └── inference/     visualize.py, run.py
    ├── docs/              per-topic concept and engineering guides
    ├── notebooks/         walkthrough.ipynb
    ├── scripts/           generate_dataset.py
    └── outputs/samples/   preview_grid.png, dataset_stats.png, …

# Appendix B — Key References

- Dosovitskiy et al., *An Image is Worth 16×16 Words* (ICLR 2021).
- Carion et al., *End-to-End Object Detection with Transformers*
  (ECCV 2020).
- Cheng et al., *Masked-attention Mask Transformer for Universal Image
  Segmentation* (CVPR 2022).
- Kalervo et al., *CubiCasa5K* (2019) — real floorplan dataset.
