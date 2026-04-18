# Ground Truth (GT)

**Ground truth** is the *correct answer* attached to every training sample.
Models learn by comparing their predictions against the GT and minimising a
loss. At evaluation time, GT is also the yardstick for "did the model get
it right?".

In this project each synthetic floorplan produces **three** kinds of GT
because we're training three different models.

---

## 1. Classification GT — a single integer

For the ViT classifier, GT is just one integer per image: the index of the
dominant room type.

```python
ROOM_TYPES = ["office", "meeting_room", "corridor",
              "restroom", "lobby", "storage"]

# For this floorplan:        GT = 4   (meaning "lobby")
```

Think of it like a single-word label on a photograph.

---

## 2. Detection GT — a list of boxes

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

---

## 3. Segmentation GT — a per-pixel mask

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

---

## 4. Why three kinds of GT?

The three tasks answer three different questions:

| Task | Question it answers | GT shape |
|---|---|---|
| Classification | *"What kind of place is this?"* | one int |
| Detection | *"Where are the specific things?"* | list of boxes |
| Segmentation | *"Which pixel belongs to what?"* | (H, W) int map |

Each model has a loss function tailored to the GT shape:

- Classification → **cross-entropy** on one logits vector
- Detection → **Hungarian matching + CE + L1 + GIoU** on matched boxes
- Segmentation → **per-pixel cross-entropy + Dice** on the full map

See the individual model docs for how each loss is computed.
