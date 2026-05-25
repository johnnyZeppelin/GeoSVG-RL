#!/usr/bin/env bash
set -euo pipefail
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
  --use-browser true
