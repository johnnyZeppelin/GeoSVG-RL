#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
python -m geosvg_rl.training.train_sft \
  --train-jsonl data/geosvg_synth/train.jsonl \
  --val-jsonl data/geosvg_synth/val.jsonl \
  --task generator \
  --model-name "$MODEL" \
  --output-dir checkpoints/geosvg_sft_generator \
  --max-seq-length 4096 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 128 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
