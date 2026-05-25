# Synthetic Dataset

The synthetic data generator creates aligned prompt-plan-SVG examples for layout-constrained technical diagrams.

## Families

1. horizontal pipelines;
2. vertical stacks;
3. branching flows;
4. grouped containers;
5. retrieval-oriented architectures;
6. multi-stage ML workflows.

## JSONL schema

Each row contains:

```json
{
  "id": "train_000001",
  "split": "train",
  "family": "pipeline",
  "prompt": "Draw a clean horizontal pipeline diagram ...",
  "plan": {
    "canvas": {"width": 800, "height": 600},
    "nodes": [
      {"id": "N1", "type": "rect", "x": 60, "y": 268, "width": 130, "height": 64, "label": "Input"}
    ],
    "edges": [
      {"id": "E1", "src": "N1", "dst": "N2", "src_anchor": "right", "dst_anchor": "left", "label": ""}
    ],
    "groups": []
  },
  "svg": "<svg ...>...</svg>",
  "metadata": {
    "anchors": {"N1": {"left": [60, 300], "right": [190, 300]}},
    "edges": []
  }
}
```

## Recommended split sizes

| Split | Size |
|---|---:|
| train | 48,000 |
| validation | 4,000 |
| IID test | 4,000 |
| template-held-out test | 2,000 |
| complexity-held-out test | 2,000 |
| real-prompt-style test | 200 |

## Generate

```bash
python -m geosvg_rl.data.generate --out data/geosvg_synth --n-train 48000 --n-val 4000 --n-iid-test 4000 --n-template-test 2000 --n-complexity-test 2000 --n-real-prompt-test 200 --seed 13 --render-png
```
