# ViT — Vision Transformer Classifier

> *"What kind of room is this?"*

**Input** → one image
**Output** → one class label (one of `ROOM_TYPES`)

---

## 1. Idea

Chop the image into non-overlapping patches, treat each patch as a "word",
run a transformer over the "sentence" of patches, and use a special
`[CLS]` token's final embedding to predict the class.

Why it works on floorplans: room function is a **global** property.
Whether a plan is "office-heavy" or "corridor-heavy" depends on the layout
across the whole image, and self-attention lets every patch inform every
other patch in one step.

---

## 2. Tiny example (4×4 image, 2×2 patches)

```
Image                Patch grid         Tokens
┌─┬─┐  ┌─┬─┐         ┌─┬─┐              [CLS]  P1  P2  P3  P4
│A│B│  │A│B│         │1│2│     ----->    |     |   |   |   |
├─┼─┤  ├─┼─┤         ├─┼─┤               +--- transformer ---+
│C│D│  │C│D│         │3│4│                        |
└─┴─┘  └─┴─┘         └─┴─┘                      class
```

1. Prepend a learnable `[CLS]` token.
2. Add positional embeddings (so tokens know their grid location).
3. Run N transformer blocks (multi-head self-attention + MLP).
4. Feed the final `[CLS]` embedding through a linear classifier.

---

## 3. In this project (defaults)

| Hyperparameter | Value |
|---|---|
| Input size | 256 × 256 |
| Patch size | 16 × 16 |
| Tokens per image | 256 patches + 1 CLS = **257** |
| Embedding dim | 192 |
| Layers | 6 |
| Heads | 4 |
| Parameters | ~2.5 M |
| Loss | Cross-entropy over 6 room types |

Code: `src/models/vit.py`.

---

## 4. Walk-through of `forward`

```python
x = patch_embed(x)                          # (B, 256, 192)
cls = cls_token.expand(B, -1, -1)           # (B, 1, 192)
x = torch.cat([cls, x], dim=1)              # (B, 257, 192)
x = drop(x + pos_embed)                     # add positions
for blk in blocks:
    x = blk(x)                              # self-attention + MLP
x = norm(x)
logits = head(x[:, 0])                      # use [CLS] only
```

The attention inside each block is:

```python
q = k = v = x                                # same sequence
attn = softmax((q @ k.T) / sqrt(d))          # (B, heads, 257, 257)
out  = attn @ v
```

Every one of the 257 tokens can attend to every other — that's how the
`[CLS]` token "reads" the whole image.

---

## 5. When to use ViT

When you only care about **one label per image**. For a floorplan we use
it to tag the dominant room type. You could also use it to:

- Classify a whole drawing ("residential" vs "commercial").
- Predict floor number from the title block of a CAD sheet.
- Tag drawing style or CAD software (for format-specific post-processing).

---

## 6. What GT looks like here

```python
# For a sample whose dominant room is a lobby:
cls_label = ROOM_TYPES.index("lobby")   # 4
```

The loss is standard cross-entropy:

```python
loss = F.cross_entropy(model(image), cls_label)
```

No matching, no set prediction — simple by design.

---

## 7. Common pitfalls

- **Under-fitting on small data.** With only 200 train samples, expect
  60–80% val accuracy. Increase `--n-train` or `--epochs`.
- **Over-fitting.** Drop out aggressively (`dropout=0.1 → 0.3`) or shrink
  the model (`depth=6 → 3`).
- **Position embeddings at new resolutions.** If you change `img_size`,
  the number of patches changes and `pos_embed` must be reinitialised or
  interpolated.
