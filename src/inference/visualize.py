"""Visualisation helpers shared by the notebook and CLI tools."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data import ROOM_TYPES, COMPONENT_TYPES, SEG_CLASSES


# Colour map for segmentation overlay (RGB 0..255)
_SEG_COLORS = np.array([
    [255, 255, 255],   # background
    [200, 200, 100],   # floor
    [  0,   0,   0],   # wall
    [230, 100,  50],   # door
    [ 60, 140, 220],   # window
], dtype=np.uint8)


# Colour map per component class for bbox rendering
_BOX_COLORS = {
    "door":   (230, 100,  50),
    "window": ( 60, 140, 220),
    "stair":  ( 90,  90,  90),
    "desk":   (150, 110,  70),
    "toilet": (170, 170, 200),
}


def overlay_segmentation(image: np.ndarray, seg: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Blend an H x W integer seg mask on top of an RGB image."""
    colors = _SEG_COLORS[seg]
    out = (image.astype(np.float32) * (1 - alpha)
           + colors.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boxes(image: np.ndarray,
               boxes: np.ndarray,
               labels: Sequence[int],
               scores: Sequence[float] | None = None,
               class_names: Sequence[str] = COMPONENT_TYPES) -> np.ndarray:
    """Return a copy of `image` with labelled bounding boxes drawn on it."""
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        x0, y0, x1, y1 = [float(v) for v in box]
        cname = class_names[int(lbl)]
        color = _BOX_COLORS.get(cname, (255, 0, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        tag = cname if scores is None else f"{cname}:{scores[i]:.2f}"
        draw.text((x0 + 2, max(0, y0 - 10)), tag, fill=color, font=font)
    return np.array(img)


def make_prediction_figure(sample: dict,
                           pred_boxes: np.ndarray | None = None,
                           pred_labels: Sequence[int] | None = None,
                           pred_seg: np.ndarray | None = None,
                           save_path: str | None = None):
    """Build a 1x4 matplotlib figure: input | GT seg | GT boxes | prediction."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(sample["image"])
    axes[0].set_title(f"Input\ncls = {ROOM_TYPES[sample['cls_label']]}")

    gt_seg_rgb = overlay_segmentation(sample["image"], sample["seg_mask"])
    axes[1].imshow(gt_seg_rgb)
    axes[1].set_title("GT segmentation")

    gt_box_img = draw_boxes(sample["image"], sample["boxes"],
                            sample["box_labels"])
    axes[2].imshow(gt_box_img)
    axes[2].set_title("GT detection")

    if pred_seg is not None:
        pred_rgb = overlay_segmentation(sample["image"], pred_seg)
        axes[3].imshow(pred_rgb)
        axes[3].set_title("Prediction - seg")
    elif pred_boxes is not None and pred_labels is not None:
        axes[3].imshow(draw_boxes(sample["image"], pred_boxes, pred_labels))
        axes[3].set_title("Prediction - det")
    else:
        axes[3].imshow(sample["image"])
        axes[3].set_title("(no prediction)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig
