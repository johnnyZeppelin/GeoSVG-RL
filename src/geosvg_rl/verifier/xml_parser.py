from __future__ import annotations

import re
import html
from dataclasses import dataclass, field
from typing import Any

try:
    from lxml import etree
except Exception:  # pragma: no cover
    import xml.etree.ElementTree as etree  # type: ignore

from .geometry import BBox

FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


@dataclass(slots=True)
class SVGElement:
    tag: str
    id: str | None = None
    data: dict[str, str] = field(default_factory=dict)
    bbox: BBox | None = None
    text: str = ""
    points: list[tuple[float, float]] = field(default_factory=list)
    raw_attrs: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedSVG:
    valid: bool
    width: float
    height: float
    elements: list[SVGElement] = field(default_factory=list)
    error: str | None = None

    @property
    def rects(self) -> list[SVGElement]:
        return [e for e in self.elements if e.tag == "rect"]

    @property
    def texts(self) -> list[SVGElement]:
        return [e for e in self.elements if e.tag == "text"]

    @property
    def connectors(self) -> list[SVGElement]:
        return [e for e in self.elements if e.tag in {"line", "polyline", "path"} and len(e.points) >= 2]


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip().replace("px", "")
    m = re.search(FLOAT_RE, text)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def _parse_viewbox(root) -> tuple[float, float]:
    width = _float(root.attrib.get("width"), 0)
    height = _float(root.attrib.get("height"), 0)
    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if (width <= 0 or height <= 0) and view_box:
        vals = [float(v) for v in re.findall(FLOAT_RE, view_box)]
        if len(vals) == 4:
            width, height = vals[2], vals[3]
    return width or 800.0, height or 600.0


def _parse_style(style: str) -> dict[str, str]:
    out = {}
    for piece in style.split(";"):
        if ":" in piece:
            k, v = piece.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _attrs(node) -> dict[str, str]:
    attrs = {str(k): str(v) for k, v in node.attrib.items()}
    if "style" in attrs:
        attrs.update({k: v for k, v in _parse_style(attrs["style"]).items() if k not in attrs})
    return attrs


def _parse_transform(transform: str | None) -> tuple[float, float, float, float]:
    """Return approximate sx, sy, tx, ty. Handles translate and scale; ignores rotation."""
    sx = sy = 1.0
    tx = ty = 0.0
    if not transform:
        return sx, sy, tx, ty
    for name, args in re.findall(r"(translate|scale)\(([^)]*)\)", transform):
        vals = [float(v) for v in re.findall(FLOAT_RE, args)]
        if name == "translate" and vals:
            tx += vals[0]
            ty += vals[1] if len(vals) > 1 else 0.0
        elif name == "scale" and vals:
            sx *= vals[0]
            sy *= vals[1] if len(vals) > 1 else vals[0]
    return sx, sy, tx, ty


def _apply(point: tuple[float, float], tr: tuple[float, float, float, float]) -> tuple[float, float]:
    sx, sy, tx, ty = tr
    return point[0] * sx + tx, point[1] * sy + ty


def _merge_transform(parent, current):
    psx, psy, ptx, pty = parent
    sx, sy, tx, ty = current
    return psx * sx, psy * sy, ptx + tx * psx, pty + ty * psy


def _text_content(node) -> str:
    try:
        txt = "".join(node.itertext())
    except Exception:
        txt = node.text or ""
    return html.unescape(" ".join(txt.split()))


def _text_bbox(attrs: dict[str, str], text: str, tr) -> BBox:
    x = _float(attrs.get("x"), 0)
    y = _float(attrs.get("y"), 0)
    fs = _float(attrs.get("font-size"), 14)
    width = max(fs * 0.55 * len(text), fs * 0.8)
    height = fs * 1.25
    anchor = attrs.get("text-anchor", "start")
    baseline = attrs.get("dominant-baseline", attrs.get("alignment-baseline", "baseline"))
    if anchor == "middle":
        x -= width / 2
    elif anchor == "end":
        x -= width
    if baseline in {"middle", "central"}:
        y -= height / 2
    else:
        y -= height * 0.8
    x, y = _apply((x, y), tr)
    sx, sy, _, _ = tr
    return BBox(x, y, width * sx, height * sy)


def _line_points(attrs: dict[str, str], tr) -> list[tuple[float, float]]:
    return [
        _apply((_float(attrs.get("x1")), _float(attrs.get("y1"))), tr),
        _apply((_float(attrs.get("x2")), _float(attrs.get("y2"))), tr),
    ]


def _polyline_points(attrs: dict[str, str], tr) -> list[tuple[float, float]]:
    nums = [float(v) for v in re.findall(FLOAT_RE, attrs.get("points", ""))]
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append(_apply((nums[i], nums[i + 1]), tr))
    return pts


def _path_points(attrs: dict[str, str], tr) -> list[tuple[float, float]]:
    d = attrs.get("d", "")
    # Approximate with the first and last coordinate pairs. This is intentionally conservative.
    nums = [float(v) for v in re.findall(FLOAT_RE, d)]
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append(_apply((nums[i], nums[i + 1]), tr))
    if len(pts) >= 2:
        return [pts[0], pts[-1]]
    return pts


def _bbox_from_points(points: list[tuple[float, float]]) -> BBox | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def parse_svg(svg: str) -> ParsedSVG:
    try:
        parser = etree.XMLParser(recover=True, resolve_entities=False) if hasattr(etree, "XMLParser") else None
        root = etree.fromstring(svg.encode("utf-8"), parser=parser) if parser is not None else etree.fromstring(svg.encode("utf-8"))
    except Exception as e:
        return ParsedSVG(valid=False, width=800, height=600, error=f"XML parse failed: {e}")
    if _strip_ns(str(root.tag)) != "svg":
        return ParsedSVG(valid=False, width=800, height=600, error="root element is not svg")
    width, height = _parse_viewbox(root)
    elements: list[SVGElement] = []

    def walk(node, parent_tr=(1.0, 1.0, 0.0, 0.0)):
        attrs = _attrs(node)
        tr = _merge_transform(parent_tr, _parse_transform(attrs.get("transform")))
        tag = _strip_ns(str(node.tag))
        data = {k: v for k, v in attrs.items() if k.startswith("data-")}
        el: SVGElement | None = None
        if tag == "rect":
            x = _float(attrs.get("x"), 0)
            y = _float(attrs.get("y"), 0)
            w = _float(attrs.get("width"), 0)
            h = _float(attrs.get("height"), 0)
            x, y = _apply((x, y), tr)
            sx, sy, _, _ = tr
            el = SVGElement(tag=tag, id=attrs.get("id"), data=data, bbox=BBox(x, y, w * sx, h * sy), raw_attrs=attrs)
        elif tag == "text":
            text = _text_content(node)
            el = SVGElement(tag=tag, id=attrs.get("id"), data=data, bbox=_text_bbox(attrs, text, tr), text=text, raw_attrs=attrs)
        elif tag == "line":
            pts = _line_points(attrs, tr)
            el = SVGElement(tag=tag, id=attrs.get("id"), data=data, bbox=_bbox_from_points(pts), points=pts, raw_attrs=attrs)
        elif tag == "polyline":
            pts = _polyline_points(attrs, tr)
            el = SVGElement(tag=tag, id=attrs.get("id"), data=data, bbox=_bbox_from_points(pts), points=pts, raw_attrs=attrs)
        elif tag == "path":
            # Ignore marker arrowhead paths inside defs by checking if no data/edge and parent is defs is hard here.
            pts = _path_points(attrs, tr)
            el = SVGElement(tag=tag, id=attrs.get("id"), data=data, bbox=_bbox_from_points(pts), points=pts, raw_attrs=attrs)
        if el is not None:
            # Skip invisible defs marker paths if they have tiny local coordinates and no edge metadata.
            if not (tag == "path" and not data and attrs.get("fill") and len(el.points) <= 2 and el.bbox and el.bbox.area < 100):
                elements.append(el)
        for child in list(node):
            walk(child, tr)

    walk(root)
    return ParsedSVG(valid=True, width=width, height=height, elements=elements)
