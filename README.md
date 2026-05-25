# GeoSVG-RL

A codebase for **Geometry-Aware Reinforcement Learning for Layout-Constrained Text-to-SVG Diagram Generation**.

This repository implements a plan-first SVG diagram generator, a procedural synthetic data engine, a browser/XML-backed SVG verifier, geometry-aware metrics and rewards, supervised warm start training, verifier reranking, and a practical custom GRPO training loop.

> Scope: the repository includes a CPU smoke path that runs without downloading a 7B model, plus full training scripts for Qwen2.5-Coder-7B-Instruct with LoRA on GPU.

---

## 1. Repository layout

```text
geosvg-rl/
├── configs/
│   ├── smoke.yaml               # small CPU-friendly pipeline
│   └── full_qwen7b.yaml          # paper-scale/default hyperparameters
├── docs/
│   ├── DATASET.md
│   ├── METRICS.md
│   └── REPRODUCIBILITY.md
├── examples/
│   ├── example_prompt.json
│   └── run_inference.py
├── scripts/
│   ├── smoke_test.sh
│   ├── generate_dataset.sh
│   ├── train_sft.sh
│   ├── train_grpo.sh
│   └── evaluate.sh
├── src/geosvg_rl/
│   ├── data/                    # schema, procedural generator, SVG template renderer
│   ├── verifier/                # browser/XML extraction, metrics, rewards
│   ├── models/                  # prompt formatting, HF loading, generation utilities
│   ├── training/                # SFT, planner SFT, GRPO
│   ├── eval/                    # evaluation CLI and aggregation
│   └── utils/
└── tests/
```

---

## 2. Installation

### 2.1 Minimal CPU install for data generation and verification

```bash
conda create -n geosvg python=3.10 -y
conda activate geosvg
pip install -e .
```

This minimal install supports:

```bash
python -m geosvg_rl.data.generate --out data/smoke --n-train 20 --n-val 5 --n-test 5 --seed 13 --render-png
python -m geosvg_rl.eval.evaluate --jsonl data/smoke/test.jsonl --pred-field svg --out outputs/smoke_metrics.json
```

The verifier uses **Playwright/Chromium** when installed. If it is not available, it automatically falls back to a deterministic XML/text-width approximation so the repository remains runnable on a fresh machine.

### 2.2 Browser-backed verifier install

```bash
pip install -e '.[browser]'
python -m playwright install chromium
```

### 2.3 Full training install

```bash
pip install -e '.[train,browser]'
python -m playwright install chromium
```

For Qwen2.5-Coder-7B-Instruct LoRA training, use a recent CUDA PyTorch build and at least one A100 80GB GPU for comfortable reproduction. Multi-GPU training can be launched through `accelerate` or `torchrun` after adjusting the config.

---

## 3. Quick start

### 3.1 End-to-end smoke test

```bash
bash scripts/smoke_test.sh
```

This performs:

1. synthetic data generation;
2. XML/browser verifier evaluation of reference SVGs;
3. geometry metric aggregation;
4. a rule-based inference example.

Expected outputs:

```text
outputs/smoke_metrics.json
outputs/example.svg
outputs/example_plan.json
```

### 3.2 Generate synthetic corpus

Small development set:

```bash
python -m geosvg_rl.data.generate \
  --out data/dev \
  --n-train 1000 \
  --n-val 100 \
  --n-test 200 \
  --seed 13 \
  --render-png
```

Paper-scale split:

```bash
python -m geosvg_rl.data.generate \
  --out data/geosvg_synth \
  --n-train 48000 \
  --n-val 4000 \
  --n-iid-test 4000 \
  --n-template-test 2000 \
  --n-complexity-test 2000 \
  --n-real-prompt-test 200 \
  --seed 13 \
  --render-png
```

Each JSONL row contains:

```json
{
  "id": "train_000000",
  "split": "train",
  "family": "pipeline",
  "prompt": "Draw a clean horizontal pipeline diagram ...",
  "plan": {"canvas": ..., "nodes": [...], "edges": [...]},
  "svg": "<svg ...>...</svg>",
  "metadata": {"nodes": ..., "anchors": ..., "edges": ...}
}
```

### 3.3 Evaluate SVG predictions

If a JSONL file already has predictions in `pred_svg`:

```bash
python -m geosvg_rl.eval.evaluate \
  --jsonl data/my_predictions.jsonl \
  --pred-field pred_svg \
  --out outputs/eval.json \
  --num-workers 8
```

For reference SVG sanity check:

```bash
python -m geosvg_rl.eval.evaluate \
  --jsonl data/dev/test.jsonl \
  --pred-field svg \
  --out outputs/ref_eval.json
```

---

## 4. Training pipeline

### 4.1 Supervised warm start

The paper uses a plan-first factorization. This repository supports two supervised tasks:

- **Planner SFT:** prompt -> JSON layout plan
- **Generator SFT:** prompt + layout plan -> SVG code

Train generator with LoRA:

```bash
python -m geosvg_rl.training.train_sft \
  --train-jsonl data/geosvg_synth/train.jsonl \
  --val-jsonl data/geosvg_synth/val.jsonl \
  --task generator \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir checkpoints/geosvg_sft_generator \
  --max-seq-length 4096 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 128 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

Train planner:

```bash
python -m geosvg_rl.training.train_sft \
  --train-jsonl data/geosvg_synth/train.jsonl \
  --val-jsonl data/geosvg_synth/val.jsonl \
  --task planner \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir checkpoints/geosvg_sft_planner \
  --max-seq-length 2048 \
  --learning-rate 2e-5 \
  --epochs 3
```

### 4.2 Verifier reranking baseline

```bash
python -m geosvg_rl.models.generate \
  --jsonl data/geosvg_synth/iid_test.jsonl \
  --model checkpoints/geosvg_sft_generator \
  --out outputs/sft_reranked.jsonl \
  --num-candidates 4 \
  --temperature 0.6 \
  --top-p 0.90 \
  --rerank-with-verifier
```

### 4.3 GRPO refinement

```bash
python -m geosvg_rl.training.train_grpo \
  --train-jsonl data/geosvg_synth/train.jsonl \
  --val-jsonl data/geosvg_synth/val.jsonl \
  --model checkpoints/geosvg_sft_generator \
  --output-dir checkpoints/geosvg_grpo \
  --group-size 4 \
  --updates 1500 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --clip-range 0.2 \
  --kl-coef 0.02 \
  --temperature 0.6 \
  --top-p 0.90 \
  --max-new-tokens 2048 \
  --num-workers 8
```

The custom GRPO loop:

1. samples `G` SVG candidates per prompt;
2. evaluates candidates using `GeoSVGVerifier`;
3. computes group-relative advantages;
4. performs a clipped token-level policy update with a reference-policy KL term.

---

## 5. Method details implemented in code

### 5.1 Plan schema

The plan is a compact JSON contract:

```json
{
  "canvas": {"width": 800, "height": 600},
  "nodes": [
    {"id": "N1", "type": "rect", "x": 80, "y": 220, "width": 140, "height": 64, "label": "Input"}
  ],
  "edges": [
    {"id": "E1", "src": "N1", "dst": "N2", "src_anchor": "right", "dst_anchor": "left", "label": ""}
  ],
  "groups": []
}
```

All generated SVG uses semantic primitives and stable `data-node-id`, `data-edge-id`, `data-src`, and `data-dst` attributes. These attributes are not required by the verifier, but they improve extraction when available and preserve editability.

### 5.2 Synthetic diagram families

Implemented families:

- horizontal pipelines;
- vertical stacks;
- branching flows;
- grouped containers;
- retrieval-oriented architectures;
- multi-stage ML workflows.

The generator supports IID, template-held-out, complexity-held-out, and real-prompt style splits. It samples node labels, graph structures, dimensions, spacing, edge routes, and prompt paraphrases.

### 5.3 Verifier and metrics

The verifier computes:

| Metric | Meaning |
|---|---|
| `RSR` | render/parse success |
| `GFR` | union bounding box inside canvas |
| `OAR` | overflow area ratio |
| `EICR` | per-element in-canvas rate |
| `AAcc` | arrow endpoints within anchor threshold |
| `AEE` | normalized endpoint distance |
| `TBR` | rendered text box inside container |
| `TPVR` | text padding violation rate |
| `E-F1` | graph edge connectivity F1 |
| `Clean` | semantic primitive ratio |

Default thresholds and reward weights match the paper-oriented settings:

```yaml
anchor_threshold_px: 12
text_padding_px: 6
weights:
  exec: 1.00
  fit: 0.60
  overflow: 0.50
  anchor: 1.20
  text: 1.10
  padding: 0.50
  graph: 0.90
  clean: 0.30
```

---

## 6. Reproducibility notes

### Recommended full run

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/generate_dataset.sh
bash scripts/train_sft.sh
bash scripts/train_grpo.sh
bash scripts/evaluate.sh
```

### Default paper-scale hyperparameters

| Item | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| SFT epochs | 3 |
| SFT learning rate | 2e-5 |
| Max sequence length | 4096 |
| GRPO group size | 4 |
| GRPO clip range | 0.2 |
| GRPO learning rate | 5e-6 |
| GRPO updates | 1500 |
| KL coefficient | 0.02 |
| Sampling | temperature 0.6, top-p 0.90 |
| Max generation tokens | 2048 |

### Hardware

The full 7B LoRA run is intended for A100 80GB-class GPUs. For development, use a smaller code model such as `Qwen/Qwen2.5-Coder-0.5B-Instruct`, reduce `max_seq_length`, and use a small generated dataset.

---

## 7. Practical tips for best performance

1. **Keep plan conditioning enabled.** It stabilizes coordinates and graph recovery.
2. **Use the browser verifier for final training/evaluation.** XML-only text estimates are useful for smoke tests but less accurate for TBR/TPVR.
3. **Start GRPO from a strong SFT checkpoint.** Random or weak SVG policies create too many malformed rollouts.
4. **Use moderate sampling.** `temperature=0.6`, `top_p=0.90`, and `G=4` provide a good stability/diversity balance.
5. **Ramp global layout terms.** The default curriculum first emphasizes anchor/text/graph correctness, then ramps canvas fit and overflow.
6. **Prefer semantic primitives.** The SVG template and reward both discourage unnecessary path fragmentation.
7. **Evaluate every model through the same verifier.** This is essential when comparing reranking, SFT, GRPO, and external baselines.

---

## 8. Known limitations

- This repository cannot include Qwen model weights. Download them through Hugging Face under the model license.
- The default synthetic generator intentionally targets box-arrow-text diagrams, not arbitrary artistic SVG.
- Full GRPO training is computationally expensive because every rollout must be rendered and verified.
- The XML fallback verifier is deterministic and useful for debugging, but browser-backed measurement should be used for final claims.

---

## 9. Citation

```bibtex
@misc{geosvg_rl_2026,
  title  = {GeoSVG-RL: Geometry-Aware Reinforcement Learning for Layout-Constrained Text-to-SVG Diagram Generation},
  year   = {2026},
  note   = {Anonymous NeurIPS submission draft}
}
```

---

## 10. License and third-party assets

This repository skeleton is released under the MIT License. You must separately respect the licenses of:

- Qwen/Qwen2.5-Coder models;
- PyTorch, Transformers, PEFT, Accelerate, Playwright, Chromium, lxml, svgpathtools, CairoSVG;
- any external baseline implementation or dataset you add.
