from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from geosvg_rl.data.schema import LayoutPlan

from .geometry import BBox, anchor_of_bbox, distance, nearest_anchor, overflow_area_ratio, union_bbox
from .xml_parser import ParsedSVG, SVGElement

EPS = 1e-8


@dataclass(slots=True)
class MetricResult:
    RSR: float = 0.0
    GFR: float = 0.0
    OAR: float = 0.0
    EICR: float = 0.0
    AAcc: float = 0.0
    AEE: float = 0.0
    TBR: float = 0.0
    TPVR: float = 0.0
    EF1: float = 0.0
    Clean: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "RSR": self.RSR,
            "GFR": self.GFR,
            "OAR": self.OAR,
            "EICR": self.EICR,
            "AAcc": self.AAcc,
            "AEE": self.AEE,
            "TBR": self.TBR,
            "TPVR": self.TPVR,
            "E-F1": self.EF1,
            "Clean": self.Clean,
            "details": self.details,
        }


def rects_by_node(parsed: ParsedSVG, plan: LayoutPlan) -> dict[str, BBox]:
    out: dict[str, BBox] = {}
    plan_ids = {n.id for n in plan.nodes}
    for el in parsed.rects:
        node_id = el.data.get("data-node-id") or el.id
        if node_id in plan_ids and el.bbox is not None:
            out[node_id] = el.bbox
    # Fallback: match sorted rects to nodes by nearest planned center, ignoring background/group boxes.
    if len(out) < len(plan.nodes):
        candidates = [r for r in parsed.rects if r.bbox and r.bbox.width > 20 and r.bbox.height > 20 and r.bbox.area < parsed.width * parsed.height * 0.8]
        used: set[int] = set()
        for node in plan.nodes:
            if node.id in out:
                continue
            target = BBox(node.x, node.y, node.width, node.height).center
            best_i, best_d = None, float("inf")
            for i, cand in enumerate(candidates):
                if i in used or cand.bbox is None:
                    continue
                d = distance(target, cand.bbox.center)
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None and candidates[best_i].bbox is not None:
                out[node.id] = candidates[best_i].bbox
                used.add(best_i)
    return out


def text_boxes_by_node(parsed: ParsedSVG, plan: LayoutPlan) -> dict[str, BBox]:
    out: dict[str, BBox] = {}
    plan_ids = {n.id for n in plan.nodes}
    # Prefer explicit attributes.
    for el in parsed.texts:
        node_id = el.data.get("data-node-id")
        if node_id in plan_ids and el.bbox is not None:
            out[node_id] = el.bbox
    if len(out) == len(plan.nodes):
        return out
    # Fallback by label text.
    label_to_id = {n.label.strip().lower(): n.id for n in plan.nodes}
    for el in parsed.texts:
        if el.bbox is None:
            continue
        key = el.text.strip().lower()
        if key in label_to_id and label_to_id[key] not in out:
            out[label_to_id[key]] = el.bbox
    # Fallback by containment in planned/parsed node box.
    node_boxes = rects_by_node(parsed, plan)
    for el in parsed.texts:
        if el.bbox is None:
            continue
        if any(existing is el.bbox for existing in out.values()):
            continue
        cx, cy = el.bbox.center
        for node_id, nb in node_boxes.items():
            if node_id not in out and nb.x <= cx <= nb.x2 and nb.y <= cy <= nb.y2:
                out[node_id] = el.bbox
                break
    return out


def extract_pred_edges(parsed: ParsedSVG, plan: LayoutPlan, node_boxes: dict[str, BBox], threshold: float = 40.0) -> dict[tuple[str, str], SVGElement]:
    pred: dict[tuple[str, str], SVGElement] = {}
    for el in parsed.connectors:
        if len(el.points) < 2:
            continue
        src = el.data.get("data-src")
        dst = el.data.get("data-dst")
        if src and dst:
            pred[(src, dst)] = el
            continue
        p1, p2 = el.points[0], el.points[-1]
        n1, _a1, d1 = nearest_anchor(p1, node_boxes)
        n2, _a2, d2 = nearest_anchor(p2, node_boxes)
        if n1 and n2 and n1 != n2 and d1 <= threshold and d2 <= threshold:
            pred[(n1, n2)] = el
    return pred


def compute_metrics(parsed: ParsedSVG, plan: LayoutPlan, *, anchor_threshold_px: float = 12.0, text_padding_px: float = 6.0) -> MetricResult:
    if not parsed.valid:
        return MetricResult(details={"error": parsed.error})

    visible_boxes = [el.bbox for el in parsed.elements if el.bbox is not None and el.bbox.area >= 0]
    all_box = union_bbox(visible_boxes)
    gfr = 1.0 if all_box is None or all_box.in_canvas(parsed.width, parsed.height) else 0.0
    oar = overflow_area_ratio(all_box, parsed.width, parsed.height)
    eicr = sum(1 for b in visible_boxes if b.in_canvas(parsed.width, parsed.height)) / max(1, len(visible_boxes))

    node_boxes = rects_by_node(parsed, plan)
    text_boxes = text_boxes_by_node(parsed, plan)
    pred_edges = extract_pred_edges(parsed, plan, node_boxes)

    # Text metrics.
    tbr_vals = []
    tpvr_vals = []
    for node in plan.nodes:
        container = node_boxes.get(node.id, BBox(node.x, node.y, node.width, node.height))
        tbox = text_boxes.get(node.id)
        if tbox is None:
            tbr_vals.append(0.0)
            tpvr_vals.append(1.0)
            continue
        tbr_vals.append(1.0 if container.contains(tbox, padding=0.0) else 0.0)
        margin = container.min_inner_margin(tbox)
        tpvr_vals.append(1.0 if margin < text_padding_px else 0.0)
    tbr = sum(tbr_vals) / max(1, len(tbr_vals))
    tpvr = sum(tpvr_vals) / max(1, len(tpvr_vals))

    # Edge F1.
    gold_edges = {(e.src, e.dst) for e in plan.edges}
    pred_edge_set = set(pred_edges)
    tp = len(gold_edges & pred_edge_set)
    precision = tp / max(1, len(pred_edge_set))
    recall = tp / max(1, len(gold_edges))
    ef1 = 2 * precision * recall / (precision + recall + EPS)

    # Anchor metrics: compare predicted edge endpoints to planned anchors.
    acc = []
    errs = []
    node_map = plan.node_map()
    for edge in plan.edges:
        el = pred_edges.get((edge.src, edge.dst))
        src_node = node_map[edge.src]
        dst_node = node_map[edge.dst]
        gold = [src_node.anchor(edge.src_anchor), dst_node.anchor(edge.dst_anchor)]
        diags = [src_node.diagonal, dst_node.diagonal]
        if el is None or len(el.points) < 2:
            acc.extend([0.0, 0.0])
            errs.extend([1.0, 1.0])
            continue
        pred_pts = [el.points[0], el.points[-1]]
        # If an extracted connector is reversed, choose the orientation with lower total error.
        direct = distance(pred_pts[0], gold[0]) + distance(pred_pts[1], gold[1])
        reverse = distance(pred_pts[1], gold[0]) + distance(pred_pts[0], gold[1])
        if reverse < direct:
            pred_pts = [pred_pts[1], pred_pts[0]]
        for p, g, dnorm in zip(pred_pts, gold, diags):
            d = distance(p, g)
            acc.append(1.0 if d <= anchor_threshold_px else 0.0)
            errs.append(min(d / max(EPS, dnorm), 1.0))
    aacc = sum(acc) / max(1, len(acc))
    aee = sum(errs) / max(1, len(errs))

    meaningful = {"rect", "text", "line", "polyline"}
    geometric = [e for e in parsed.elements if e.tag in {"rect", "text", "line", "polyline", "path", "circle", "ellipse"}]
    clean = sum(1 for e in geometric if e.tag in meaningful) / max(1, len(geometric))

    return MetricResult(
        RSR=1.0,
        GFR=gfr,
        OAR=oar,
        EICR=eicr,
        AAcc=aacc,
        AEE=aee,
        TBR=tbr,
        TPVR=tpvr,
        EF1=ef1,
        Clean=clean,
        details={
            "num_elements": len(parsed.elements),
            "num_visible_boxes": len(visible_boxes),
            "num_pred_edges": len(pred_edge_set),
            "num_gold_edges": len(gold_edges),
            "edge_precision": precision,
            "edge_recall": recall,
        },
    )
