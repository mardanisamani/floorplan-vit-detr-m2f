# DETR-lite — Detection Transformer

> *"Where are all the doors / windows / stairs / desks / toilets?"*

**Input** → one image
**Output** → a fixed-length set of `(class, bounding_box)` predictions

---

## 1. Idea

Replace the "anchors + NMS" machinery of traditional detectors with pure
**set prediction**: a transformer decoder emits N "object queries", each
responsible for at most one object. The Hungarian algorithm matches
queries to ground-truth boxes during training.

Why it works on floorplans: components like doors and windows are
**sparse, geometrically simple, and non-overlapping**. Set prediction is
a natural fit — no need for anchor tuning or IoU thresholds.

---

## 2. Pipeline

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

---

## 3. Hungarian matching — tiny example

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
    q4 ─► no-object  (supervised as "door"   prediction → wrong, penalised)
```

Then the loss is:

- **Classification CE** on all 4 queries (q2, q4 pushed toward "no-object").
- **Box L1 + GIoU** only on the matched pair (q1→#0, q3→#1).

---

## 4. The three loss components

```python
loss_cls  = weighted_cross_entropy(pred_logits, target_labels)
loss_bbox = L1(pred_boxes_matched, gt_boxes_matched)
loss_giou = 1 - GIoU(pred_boxes_matched, gt_boxes_matched)
total     = w_cls * loss_cls + w_bbox * loss_bbox + w_giou * loss_giou
```

- **`eos_coef = 0.1`** on the weighted cross-entropy — down-weights the
  "no object" class so the model doesn't collapse to always predicting it.
- **GIoU** generalises IoU so it stays informative even when predicted
  and GT boxes don't overlap.

---

## 5. In this project (defaults)

| Hyperparameter | Value |
|---|---|
| Backbone | Small CNN → stride /16 |
| Feature map | 16 × 16 tokens |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Heads | 4 |
| `d_model` | 128 |
| Object queries | **50** |
| Classes | 5 components + 1 no-object = 6 |
| Parameters | ~3.5 M |
| Loss weights | `w_cls=1, w_bbox=5, w_giou=2, eos_coef=0.1` |

Code: `src/models/detr_lite.py`.

---

## 6. Output format

```python
out = detr(image)   # one image, batch_size=1

out["pred_logits"]  # (1, 50, 6)   last class = "no object"
out["pred_boxes"]   # (1, 50, 4)   cxcywh in [0, 1]
```

Post-processing to usable predictions:

```python
prob   = out["pred_logits"].softmax(-1)[0]            # (50, 6)
scores = prob[:, :-1].max(-1).values                  # drop no-obj
labels = prob[:, :-1].argmax(-1)
keep   = scores > 0.5
final_boxes_px = cxcywh_to_xyxy(out["pred_boxes"][0][keep]) * image_size
```

No NMS needed — Hungarian matching already enforces 1-to-1 correspondence.

---

## 7. When to use DETR

When you need to **localise individual instances**. For BIM/CAD, DETR
finds every door, window, stair, desk, toilet and returns their
coordinates — exactly what a routing graph needs (one edge per door).

---

## 8. Common pitfalls

- **Slow convergence.** DETR is famously slow to train. Give it more
  epochs or use Deformable-DETR / DN-DETR if you need faster convergence.
- **Too few queries.** If `n_queries < max_objects_per_image`, the model
  literally cannot predict them all. Default 50 is generous for our
  synthetic data.
- **Uneven classes.** Rare classes (e.g. stairs) need either upsampling
  or class-weighted CE if you care about them.
