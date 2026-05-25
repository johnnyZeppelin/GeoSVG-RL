#!/usr/bin/env bash
set -euo pipefail

mkdir -p data outputs
python -m geosvg_rl.data.generate --out data/smoke --n-train 20 --n-val 5 --n-test 5 --n-template-test 0 --n-complexity-test 0 --n-real-prompt-test 0 --seed 13
python -m geosvg_rl.eval.evaluate --jsonl data/smoke/test.jsonl --pred-field svg --out outputs/smoke_metrics.json --use-browser auto
python examples/run_inference.py --prompt-json examples/example_prompt.json --out-dir outputs
cat outputs/smoke_metrics.json
