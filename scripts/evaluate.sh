#!/usr/bin/env bash
set -euo pipefail
python -m geosvg_rl.models.generate \
  --jsonl data/geosvg_synth/iid_test.jsonl \
  --model checkpoints/geosvg_grpo \
  --out outputs/grpo_iid_predictions.jsonl \
  --num-candidates 1 \
  --temperature 0.6 \
  --top-p 0.90
python -m geosvg_rl.eval.evaluate \
  --jsonl outputs/grpo_iid_predictions.jsonl \
  --pred-field pred_svg \
  --out outputs/grpo_iid_eval.json \
  --details-out outputs/grpo_iid_eval_details.jsonl \
  --num-workers 8 \
  --use-browser true
