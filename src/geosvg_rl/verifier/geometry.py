from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

EPS = 1e-8


@dataclass(slots=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def contains(self, other: "BBox", padding: float = 0.0) -> bool:
        return (
            other.x >= self.x + padding
            and other.y >= self.y + padding
            and other.x2 <= self.x2 - padding
            and other.y2 <= self.y2 - padding
        )

    def in_canvas(self, width: float, height: float, padding: float = 0.0) -> bool:
        return self.x >= padding and self.y >= padding and self.x2 <= width - padding and self.y2 <= height - padding

    def min_inner_margin(self, other: "BBox") -> float:
        return min(other.x - self.x, other.y - self.y, self.x2 - other.x2, self.y2 - other.y2)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def union_bbox(boxes: Iterable[BBox]) -> BBox | None:
    boxes = [b for b in boxes if b.width >= 0 and b.height >= 0]
    if not boxes:
        return None
    x1 = min(b.x for b in boxes)
    y1 = min(b.y for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    return BBox(x1, y1, x2 - x1, y2 - y1)


def overflow_area_ratio(box: BBox | None, canvas_w: float, canvas_h: float) -> float:
    if box is None or box.area <= EPS:
        return 0.0
    ix1 = max(box.x, 0)
    iy1 = max(box.y, 0)
    ix2 = min(box.x2, canvas_w)
    iy2 = min(box.y2, canvas_h)
    inside_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    outside = max(0.0, box.area - inside_area)
    return outside / (box.area + EPS)


def distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    return hypot(p[0] - q[0], p[1] - q[1])


def anchor_of_bbox(box: BBox, name: str) -> tuple[float, float]:
    if name == "left":
        return (box.x, box.y + box.height / 2)
    if name == "right":
        return (box.x2, box.y + box.height / 2)
    if name == "top":
        return (box.x + box.width / 2, box.y)
    if name == "bottom":
        return (box.x + box.width / 2, box.y2)
    return box.center


def nearest_anchor(point: tuple[float, float], boxes: dict[str, BBox]) -> tuple[str | None, str | None, float]:
    best = (None, None, float("inf"))
    for node_id, box in boxes.items():
        for name in ["left", "right", "top", "bottom"]:
            d = distance(point, anchor_of_bbox(box, name))
            if d < best[2]:
                best = (node_id, name, d)
    return best
