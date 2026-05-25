from __future__ import annotations

import json
import re
from typing import Any


def format_plan_prompt(prompt: str) -> str:
    return (
        "You are a layout planner for editable SVG technical diagrams.\n"
        "Given a natural-language diagram request, output only a compact JSON layout plan with canvas, nodes, edges, and groups.\n"
        "Every node must have id, type, x, y, width, height, and label. Every edge must have src, dst, src_anchor, and dst_anchor.\n\n"
        f"Request:\n{prompt}\n\nJSON layout plan:\n"
    )


def format_svg_prompt(prompt: str, plan: dict[str, Any]) -> str:
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
    return (
        "You are an expert SVG code generator for clean, editable box-arrow-text diagrams.\n"
        "Generate only one complete SVG program. Do not include Markdown fences or explanations.\n"
        "Use semantic primitives such as rect, text, line, and polyline when possible.\n"
        "Keep all text inside its assigned box with safe padding. Align connector endpoints to valid node anchors.\n"
        "Preserve data-node-id on node rectangles/text and data-edge-id/data-src/data-dst on connectors.\n\n"
        f"Request:\n{prompt}\n\nLayout plan JSON:\n{plan_text}\n\nSVG code:\n"
    )


def extract_svg(text: str) -> str:
    if "```" in text:
        blocks = re.findall(r"```(?:svg|xml)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            text = max(blocks, key=len)
    m = re.search(r"<svg[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    return m.group(0).strip() if m else text.strip()


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            text = blocks[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None
