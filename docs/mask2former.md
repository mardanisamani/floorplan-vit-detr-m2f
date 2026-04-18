# Mask2Former-lite — Universal Mask Transformer

> *"Which pixel belongs to what?"*

**Input** → one image
**Output** → a per-pixel class map (semantic segmentation)

---

## 1. Idea

Same query-based spirit as DETR, but each query now predicts a **mask**
over the whole image instead of a box. For semantic segmentation, we
aggregate the per-query masks into per-class masks.

Why it works on floorplans: walls, floors and open space are defined by
**irregular polygons**, not rectangles. Bounding boxes can't express the
L-shape of a corridor — masks can.

---

## 2. Pipeline

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

The key trick is the **dot-product** between a query embedding and each
pixel's feature vector — turning a learned query into "where on the
image does this pattern appear?".

---

## 3. Tiny example — one query predicting the "wall" mask

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

---

## 4. In this project (defaults)

| Hyperparameter | Value |
|---|---|
| Backbone strides | /4, /8, /16 |
| Pixel decoder resolution | /4 (64 × 64 for 256 input) |
| Decoder layers | 3 |
| Heads | 4 |
| `d_model` | 128 |
| Mask queries | **20** |
| Classes | 5 semantic classes + 1 no-object |
| Parameters | ~2.8 M |

Code: `src/models/mask2former_lite.py`.

### 4.1 Simplification vs. the original

The reference implementation in this repo **skips the set-prediction
matcher for masks** and instead supervises the aggregated per-class
`seg_logits` directly with:

```python
loss = cross_entropy(seg_logits, target_seg) + dice_loss(prob, one_hot_target)
```

This is a pragmatic simplification for CPU-scale demos. To go full
panoptic, add Hungarian matching on `(class, mask)` pairs (see the
original Mask2Former paper, Sec. 3.2).

---

## 5. Output format

```python
out = m2f(image)

out["mask_logits"]   # (B, Q,        Hp, Wp)  per-query raw masks
out["class_logits"]  # (B, Q, C+1)             per-query class probs
out["seg_logits"]    # (B, C,        Hp, Wp)  aggregated per-class scores
out["pixel_features"]# (B, D,        Hp, Wp)  for debugging
```

For a full-resolution segmentation:

```python
pred = F.interpolate(out["seg_logits"], size=(256, 256),
                     mode="bilinear", align_corners=False).argmax(1)
```

---

## 6. When to use Mask2Former

When you need **pixel-precise boundaries** — walls, floors, open space.
Bounding boxes aren't enough for wall extraction because walls follow
irregular outlines. Mask2Former's mask head gives you the exact polygon,
which is what a routable indoor-map navmesh ultimately consumes.

---

## 7. Common pitfalls

- **Class imbalance.** Windows occupy <1% of pixels — plain cross-entropy
  will happily ignore them. Dice loss (already included) helps; you can
  also weight per-class CE.
- **Low-resolution artefacts.** Masks are computed at /4 resolution then
  up-sampled. For very thin structures (wall outlines), upgrade to /2 or
  full resolution at the cost of more memory.
- **Query collapse.** If many queries learn the same thing, add mutual
  diversity losses or reduce `n_queries`.
