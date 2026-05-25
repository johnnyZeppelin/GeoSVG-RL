from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from geosvg_rl.data.svg_template import render_plan_to_svg
from geosvg_rl.models.hf_io import ModelLoadConfig, load_causal_lm, load_tokenizer
from geosvg_rl.models.prompts import extract_svg, format_svg_prompt
from geosvg_rl.utils.jsonl import iter_jsonl, write_jsonl
from geosvg_rl.verifier import GeoSVGVerifier


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return extract_svg(gen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SVG predictions from an SFT/GRPO model.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--model", default=None, help="HF model/checkpoint. If omitted, use rule-based renderer.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--rerank-with-verifier", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    rows = list(iter_jsonl(args.jsonl))
    model = tokenizer = None
    if args.model:
        tokenizer = load_tokenizer(args.model)
        model = load_causal_lm(ModelLoadConfig(args.model, load_in_4bit=args.load_in_4bit), for_training=False)
        model.eval()
    verifier = GeoSVGVerifier(use_browser="auto") if args.rerank_with_verifier else None
    out_rows: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="generate"):
        if model is None:
            pred = render_plan_to_svg(__import__("geosvg_rl.data.schema", fromlist=["LayoutPlan"]).LayoutPlan.from_dict(row["plan"]), title=row["prompt"])
            row["pred_svg"] = pred
            out_rows.append(row)
            continue
        input_text = format_svg_prompt(row["prompt"], row["plan"])
        candidates = [generate_one(model, tokenizer, input_text, args.max_new_tokens, args.temperature, args.top_p) for _ in range(args.num_candidates)]
        if verifier is not None and len(candidates) > 1:
            scored = [(verifier.verify(svg, row["plan"]).reward.reward, svg) for svg in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            row["pred_svg"] = scored[0][1]
            row["candidate_rewards"] = [s for s, _ in scored]
        else:
            row["pred_svg"] = candidates[0]
        out_rows.append(row)

    write_jsonl(args.out, out_rows)
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
