#!/usr/bin/env bash
set -euo pipefail
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
