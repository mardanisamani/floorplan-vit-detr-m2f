# Model Comparison — ViT vs DETR vs Mask2Former

All three use **transformers**, which boil down to one idea: *every output
token attends to every input token and computes a weighted sum*. The
differences are (a) what the tokens represent and (b) what the final
heads do with them.

---

## 1. At a glance

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

---

## 2. Intuitive analogy

- **ViT** is like a librarian who reads an entire book and says *"this is
  a mystery novel"*.
- **DETR** is like a bouncer with 50 clipboards; each clipboard is told
  to look for one object; most go home empty, a few hand in a name and
  a location.
- **Mask2Former** is like 20 spotlights, each trained on a different
  concept ("wall", "door"…); each spotlight shines on the pixels it
  recognises, then all spotlights combine into a coloured floor plan.

---

## 3. How they complement each other

A full BIM-to-indoor-map pipeline might use:

1. **ViT** to tag each room's function (POI metadata).
2. **DETR** to find doors and stairs (graph edges for routing).
3. **Mask2Former** to get clean wall/floor polygons (the navmesh).

Each model specialises; together they cover the three levels of
understanding needed to automate BIM-to-map conversion.

```
┌─────────┐      ┌─────────┐      ┌──────────────┐
│  ViT    │      │  DETR   │      │ Mask2Former  │
│ (what?) │      │ (where?)│      │ (which px?)  │
└────┬────┘      └────┬────┘      └──────┬───────┘
     │                │                  │
     ▼                ▼                  ▼
  room name      door & stair         wall/floor
   metadata      coordinates          polygons
     │                │                  │
     └────────────────┴──────────────────┘
                      │
                      ▼
             routable indoor map
```

---

## 4. Concrete numbers you can expect

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

## 5. FAQ

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
