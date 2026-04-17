"""
Synthetic CAD/BIM floorplan generator.

Produces plan-view floorplans that mimic BIM exports typically used
for indoor mapping. Each sample comes with:

* RGB image   - drawn plan view (walls, floors, doors, windows, furniture)
* cls_label   - dominant room type (int)
* boxes       - Nx4 ndarray of (x0, y0, x1, y1) component boxes
* box_labels  - N ints, index into COMPONENT_TYPES
* seg_mask    - HxW int mask, index into SEG_CLASSES

Uses only numpy + PIL so it runs without PyTorch and is fully deterministic
under `seed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

ROOM_TYPES: List[str] = [
    "office",
    "meeting_room",
    "corridor",
    "restroom",
    "lobby",
    "storage",
]

COMPONENT_TYPES: List[str] = [
    "door",
    "window",
    "stair",
    "desk",
    "toilet",
]

SEG_CLASSES: List[str] = [
    "background",  # 0  - outside the building footprint
    "floor",       # 1  - room interior floor
    "wall",        # 2  - any wall
    "door",        # 3  - door opening region
    "window",      # 4  - window frame
]


# ---------------------------------------------------------------------------
# Colour palette for the rendered plan view (values in 0..255 RGB)
# ---------------------------------------------------------------------------

_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "background":    (245, 245, 240),
    "floor_office":  (235, 225, 200),
    "floor_meeting": (220, 235, 220),
    "floor_corridor":(235, 235, 235),
    "floor_restroom":(210, 225, 240),
    "floor_lobby":   (240, 220, 210),
    "floor_storage": (230, 225, 215),
    "wall":          (40, 40, 40),
    "door":          (180, 120, 60),
    "window":        (90, 140, 200),
    "desk":          (150, 110, 70),
    "toilet":        (220, 220, 230),
    "stair":         (90, 90, 90),
}


_ROOM_FLOOR_COLOR = {
    "office": _PALETTE["floor_office"],
    "meeting_room": _PALETTE["floor_meeting"],
    "corridor": _PALETTE["floor_corridor"],
    "restroom": _PALETTE["floor_restroom"],
    "lobby": _PALETTE["floor_lobby"],
    "storage": _PALETTE["floor_storage"],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FloorplanConfig:
    size: int = 256              # square image side
    min_room: int = 40           # min room dimension in px
    max_depth: int = 3           # BSP split depth (2 -> up to 4 rooms, 3 -> up to 8)
    wall_thickness: int = 4
    door_width: int = 14
    window_width: int = 20
    margin: int = 8              # outer building padding
    seed: int = 0


# ---------------------------------------------------------------------------
# BSP room layout
# ---------------------------------------------------------------------------

@dataclass
class Room:
    x0: int
    y0: int
    x1: int
    y1: int
    room_type: str = "office"

    @property
    def w(self) -> int: return self.x1 - self.x0
    @property
    def h(self) -> int: return self.y1 - self.y0
    @property
    def area(self) -> int: return self.w * self.h
    @property
    def cx(self) -> float: return 0.5 * (self.x0 + self.x1)
    @property
    def cy(self) -> float: return 0.5 * (self.y0 + self.y1)


def _bsp_split(rng: np.random.Generator, room: Room, depth: int,
               min_room: int) -> List[Room]:
    if depth == 0 or (room.w < 2 * min_room and room.h < 2 * min_room):
        return [room]

    # Pick split axis biased toward the longer side
    can_v = room.w >= 2 * min_room
    can_h = room.h >= 2 * min_room
    if can_v and can_h:
        vertical = rng.random() < (room.w / (room.w + room.h))
    elif can_v:
        vertical = True
    elif can_h:
        vertical = False
    else:
        return [room]

    if vertical:
        split = rng.integers(room.x0 + min_room, room.x1 - min_room + 1)
        left = Room(room.x0, room.y0, int(split), room.y1)
        right = Room(int(split), room.y0, room.x1, room.y1)
        return _bsp_split(rng, left, depth - 1, min_room) + \
               _bsp_split(rng, right, depth - 1, min_room)
    else:
        split = rng.integers(room.y0 + min_room, room.y1 - min_room + 1)
        top = Room(room.x0, room.y0, room.x1, int(split))
        bot = Room(room.x0, int(split), room.x1, room.y1)
        return _bsp_split(rng, top, depth - 1, min_room) + \
               _bsp_split(rng, bot, depth - 1, min_room)


def _assign_room_types(rng: np.random.Generator, rooms: List[Room]) -> None:
    """Assign room types with loose heuristics. The biggest becomes lobby,
    thin ones become corridors, rest are sampled."""
    if not rooms:
        return
    rooms_sorted_area = sorted(rooms, key=lambda r: -r.area)
    rooms_sorted_area[0].room_type = "lobby"

    for r in rooms:
        if r.room_type == "lobby":
            continue
        ratio = max(r.w, r.h) / max(min(r.w, r.h), 1)
        if ratio > 2.2:
            r.room_type = "corridor"
        else:
            r.room_type = rng.choice(
                ["office", "meeting_room", "restroom", "storage"]
            )


# ---------------------------------------------------------------------------
# Wall / door / window detection between neighbouring rooms
# ---------------------------------------------------------------------------

def _shared_wall(a: Room, b: Room) -> Tuple[str, int, int, int] | None:
    """Return ('v'|'h', fixed_coord, lo, hi) if rooms share a wall segment."""
    # Vertical wall: a.x1 == b.x0 or a.x0 == b.x1
    if a.x1 == b.x0 or b.x1 == a.x0:
        x = a.x1 if a.x1 == b.x0 else a.x0
        lo = max(a.y0, b.y0)
        hi = min(a.y1, b.y1)
        if hi - lo > 20:
            return "v", x, lo, hi
    if a.y1 == b.y0 or b.y1 == a.y0:
        y = a.y1 if a.y1 == b.y0 else a.y0
        lo = max(a.x0, b.x0)
        hi = min(a.x1, b.x1)
        if hi - lo > 20:
            return "h", y, lo, hi
    return None


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_floorplan(cfg: FloorplanConfig) -> Dict:
    """Generate a single floorplan sample.

    Returns a dict with keys:
      image         - (H, W, 3) uint8
      seg_mask      - (H, W)    int64, values in [0, len(SEG_CLASSES))
      boxes         - (N, 4)    float32, xyxy in pixel coordinates
      box_labels    - (N,)      int64, index into COMPONENT_TYPES
      cls_label     - int, index into ROOM_TYPES (dominant room)
      rooms         - list of Room dataclasses (for debugging)
    """
    rng = np.random.default_rng(cfg.seed)
    S = cfg.size
    m = cfg.margin

    # --- layout
    footprint = Room(m, m, S - m, S - m)
    rooms = _bsp_split(rng, footprint, cfg.max_depth, cfg.min_room)
    _assign_room_types(rng, rooms)

    # --- initialise arrays
    seg = np.zeros((S, S), dtype=np.int64)  # background = 0
    img = Image.new("RGB", (S, S), _PALETTE["background"])
    draw = ImageDraw.Draw(img)

    # --- paint floors
    for r in rooms:
        color = _ROOM_FLOOR_COLOR[r.room_type]
        draw.rectangle([r.x0, r.y0, r.x1 - 1, r.y1 - 1], fill=color)
        seg[r.y0:r.y1, r.x0:r.x1] = SEG_CLASSES.index("floor")

    # --- paint walls (outer + between rooms)
    t = cfg.wall_thickness
    wall_idx = SEG_CLASSES.index("wall")

    def paint_wall(x0, y0, x1, y1):
        # Rectangle wall segment (inclusive). Update both image + seg.
        x0c, x1c = max(0, x0), min(S, x1 + 1)
        y0c, y1c = max(0, y0), min(S, y1 + 1)
        seg[y0c:y1c, x0c:x1c] = wall_idx
        draw.rectangle([x0, y0, x1, y1], fill=_PALETTE["wall"])

    # Outer perimeter
    paint_wall(m - t, m - t, S - m + t - 1, m + t - 1)          # top
    paint_wall(m - t, S - m - t, S - m + t - 1, S - m + t - 1)  # bottom
    paint_wall(m - t, m - t, m + t - 1, S - m + t - 1)          # left
    paint_wall(S - m - t, m - t, S - m + t - 1, S - m + t - 1)  # right

    # Interior walls between neighbouring rooms
    components: List[Tuple[Tuple[int, int, int, int], str]] = []
    half = t // 2

    placed_shared = set()
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            sw = _shared_wall(a, b)
            if sw is None:
                continue
            orient, coord, lo, hi = sw
            key = (orient, coord, lo, hi)
            if key in placed_shared:
                continue
            placed_shared.add(key)

            if orient == "v":
                paint_wall(coord - half, lo, coord + half, hi)
            else:
                paint_wall(lo, coord - half, hi, coord + half)

            # Door placement
            if rng.random() < 0.9:
                length = hi - lo
                door_margin = cfg.door_width + 4
                if length > 2 * door_margin + 4:
                    center = int(rng.integers(lo + door_margin,
                                              hi - door_margin))
                    if orient == "v":
                        box = (coord - cfg.door_width // 2,
                               center - cfg.door_width // 2,
                               coord + cfg.door_width // 2,
                               center + cfg.door_width // 2)
                    else:
                        box = (center - cfg.door_width // 2,
                               coord - cfg.door_width // 2,
                               center + cfg.door_width // 2,
                               coord + cfg.door_width // 2)
                    x0, y0, x1, y1 = box
                    draw.rectangle([x0, y0, x1, y1], fill=_PALETTE["door"])
                    seg[y0:y1, x0:x1] = SEG_CLASSES.index("door")
                    components.append((box, "door"))

    # --- Windows on the outer walls
    for side in ("top", "bottom", "left", "right"):
        n_win = rng.integers(1, 4)
        for _ in range(n_win):
            if side in ("top", "bottom"):
                cx = rng.integers(m + 15, S - m - 15)
                y = m if side == "top" else S - m
                box = (cx - cfg.window_width // 2, y - t // 2,
                       cx + cfg.window_width // 2, y + t // 2)
            else:
                cy = rng.integers(m + 15, S - m - 15)
                x = m if side == "left" else S - m
                box = (x - t // 2, cy - cfg.window_width // 2,
                       x + t // 2, cy + cfg.window_width // 2)
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], fill=_PALETTE["window"])
            seg[y0:y1, x0:x1] = SEG_CLASSES.index("window")
            components.append((box, "window"))

    # --- Room-type-appropriate furniture
    for r in rooms:
        if r.room_type == "office":
            # 1-3 desks
            for _ in range(int(rng.integers(1, 4))):
                dw = int(rng.integers(18, 32))
                dh = int(rng.integers(12, 18))
                if r.w - dw - 12 <= 8 or r.h - dh - 12 <= 8:
                    continue
                x = int(rng.integers(r.x0 + 8, r.x1 - dw - 8))
                y = int(rng.integers(r.y0 + 8, r.y1 - dh - 8))
                draw.rectangle([x, y, x + dw, y + dh], fill=_PALETTE["desk"],
                               outline=(60, 40, 20))
                components.append(((x, y, x + dw, y + dh), "desk"))
        elif r.room_type == "meeting_room":
            # 1 big desk in centre
            dw = max(24, int(r.w * 0.5))
            dh = max(14, int(r.h * 0.3))
            x = int(r.cx - dw / 2); y = int(r.cy - dh / 2)
            draw.rectangle([x, y, x + dw, y + dh], fill=_PALETTE["desk"],
                           outline=(60, 40, 20))
            components.append(((x, y, x + dw, y + dh), "desk"))
        elif r.room_type == "restroom":
            # toilets along a wall
            for _ in range(int(rng.integers(1, 3))):
                tw, th = 14, 16
                if r.w - tw - 12 <= 8 or r.h - th - 12 <= 8:
                    continue
                x = int(rng.integers(r.x0 + 8, r.x1 - tw - 8))
                y = int(rng.integers(r.y0 + 8, r.y1 - th - 8))
                draw.rectangle([x, y, x + tw, y + th], fill=_PALETTE["toilet"],
                               outline=(120, 120, 140))
                components.append(((x, y, x + tw, y + th), "toilet"))
        elif r.room_type == "lobby":
            # a staircase glyph
            if r.w > 40 and r.h > 40 and rng.random() < 0.7:
                sw = 28; sh = 20
                x = int(r.cx - sw / 2); y = int(r.cy - sh / 2)
                draw.rectangle([x, y, x + sw, y + sh], fill=_PALETTE["stair"],
                               outline=(30, 30, 30))
                # tread lines
                for i in range(1, 6):
                    yy = y + int(i * sh / 6)
                    draw.line([(x, yy), (x + sw, yy)], fill=(200, 200, 200))
                components.append(((x, y, x + sw, y + sh), "stair"))

    # --- Pack into arrays
    if components:
        boxes = np.array([c[0] for c in components], dtype=np.float32)
        box_labels = np.array([COMPONENT_TYPES.index(c[1]) for c in components],
                              dtype=np.int64)
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        box_labels = np.zeros((0,), dtype=np.int64)

    # Dominant room type by area
    area_by_type: Dict[str, int] = {}
    for r in rooms:
        area_by_type[r.room_type] = area_by_type.get(r.room_type, 0) + r.area
    dominant = max(area_by_type, key=area_by_type.get)
    cls_label = ROOM_TYPES.index(dominant)

    image = np.array(img, dtype=np.uint8)

    return {
        "image": image,
        "seg_mask": seg,
        "boxes": boxes,
        "box_labels": box_labels,
        "cls_label": cls_label,
        "rooms": rooms,
    }


# ---------------------------------------------------------------------------
# Convenience: build an in-memory dataset for a notebook
# ---------------------------------------------------------------------------

def build_dataset(n: int, size: int = 256, seed: int = 0) -> List[Dict]:
    samples = []
    for i in range(n):
        cfg = FloorplanConfig(size=size, seed=seed + i)
        samples.append(generate_floorplan(cfg))
    return samples
