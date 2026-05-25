# Reproducibility Checklist

## Environment

```bash
conda create -n geosvg python=3.10 -y
conda activate geosvg
pip install -e '.[train,browser]'
python -m playwright install chromium
```

## Seeds

Use seeds `13`, `21`, and `42` for repeated runs.

## Data

Generate all synthetic splits with disjoint random seeds:

```bash
bash scripts/generate_dataset.sh
```

## SFT

```bash
bash scripts/train_sft.sh
```

## GRPO

```bash
bash scripts/train_grpo.sh
```

## Evaluation

```bash
bash scripts/evaluate.sh
```

## Notes

- Use the same verifier for all models and baselines.
- Disable self-consistency for final single-sample evaluation.
- Strip non-SVG explanatory text before verification.
- Report point estimates and, when possible, seed-wise confidence intervals.
