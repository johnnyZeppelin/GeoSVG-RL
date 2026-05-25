# GeoSVG-RL Metric Definitions

This document defines the metrics implemented in `geosvg_rl.verifier`.

## Execution and global layout

- **RSR, Render Success Rate:** `1` if SVG parses and renders, otherwise `0`.
- **GFR, Global Fit Rate:** `1` if the union bounding box of all visible elements lies inside the canvas.
- **OAR, Overflow Area Ratio:** area outside canvas divided by the union bounding-box area. Lower is better.
- **EICR, Element-In-Canvas Rate:** fraction of extracted visible elements fully inside the canvas.

## Local geometric precision

- **AAcc, Arrow Anchor Accuracy:** fraction of connector endpoints within `tau=12px` of their target anchor.
- **AEE, Anchor Endpoint Error:** mean normalized endpoint distance, using the node-box diagonal as normalization.
- **TBR, Text-In-Box Rate:** fraction of text boxes fully inside their assigned node boxes.
- **TPVR, Text Padding Violation Rate:** fraction of text boxes with less than `p=6px` safe margin.

## Structural and code quality

- **E-F1, Edge Connectivity F1:** edge-level F1 between recovered and target directed graph edges.
- **Clean, SVG Cleanliness:** ratio of semantic primitives (`rect`, `text`, `line`, `polyline`) among geometric elements.

## Reward

The total reward is a normalized weighted sum:

```text
R = w_exec*RSR + w_fit*GFR - w_overflow*OAR
  + w_anchor*(AAcc - AEE) + w_text*TBR - w_padding*TPVR
  + w_graph*E-F1 + w_clean*Clean
```

Default weights:

| Component | Weight |
|---|---:|
| exec | 1.00 |
| fit | 0.60 |
| overflow | 0.50 |
| anchor | 1.20 |
| text | 1.10 |
| padding | 0.50 |
| graph | 0.90 |
| clean | 0.30 |
