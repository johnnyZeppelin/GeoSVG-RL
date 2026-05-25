from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geosvg_rl.models.hf_io import ModelLoadConfig, load_causal_lm, load_tokenizer
from geosvg_rl.models.prompts import extract_svg, format_svg_prompt
from geosvg_rl.training.grpo_utils import grpo_loss, selective_logprobs
from geosvg_rl.utils.jsonl import iter_jsonl
from geosvg_rl.utils.seed import seed_everything
from geosvg_rl.verifier import GeoSVGVerifier, group_relative_advantages


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def generate_candidates(model, tokenizer, prompts: list[str], *, group_size: int, max_new_tokens: int, temperature: float, top_p: float) -> list[list[str]]:
    all_candidates: list[list[str]] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        candidates = []
        for _ in range(group_size):
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
            candidates.append(extract_svg(gen))
        all_candidates.append(candidates)
    return all_candidates


def build_policy_batch(tokenizer, prompt_texts: list[str], svg_texts: list[str], max_length: int, device) -> dict[str, torch.Tensor]:
    rows = []
    for prompt, svg in zip(prompt_texts, svg_texts):
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        y_ids = tokenizer(svg + "\n", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = (p_ids + y_ids)[:max_length]
        labels = ([-100] * len(p_ids) + y_ids)[:max_length]
        rows.append((input_ids, labels))
    max_len = max(len(x[0]) for x in rows)
    input_ids, labels, attention_mask = [], [], []
    for ids, lab in rows:
        pad = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom verifier-backed GRPO training for GeoSVG-RL.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--model", required=True, help="SFT checkpoint or HF model name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--use-browser", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    use_browser: str | bool = args.use_browser
    if use_browser == "true":
        use_browser = True
    elif use_browser == "false":
        use_browser = False

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(ModelLoadConfig(args.model, load_in_4bit=args.load_in_4bit), for_training=True)
    model.train()
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    verifier = GeoSVGVerifier(use_browser=use_browser)
    rows = list(iter_jsonl(args.train_jsonl))
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate_rows, drop_last=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    pbar = tqdm(total=args.updates, desc="grpo")
    while global_step < args.updates:
        for batch_rows in loader:
            if global_step >= args.updates:
                break
            prompt_texts_base = [format_svg_prompt(r["prompt"], r["plan"]) for r in batch_rows]
            candidate_groups = generate_candidates(
                model,
                tokenizer,
                prompt_texts_base,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            flat_prompts: list[str] = []
            flat_svgs: list[str] = []
            flat_advantages: list[float] = []
            flat_rewards: list[float] = []
            for row, prompt_text, candidates in zip(batch_rows, prompt_texts_base, candidate_groups):
                rewards = [verifier.verify(svg, row["plan"], update=global_step).reward.reward for svg in candidates]
                adv = group_relative_advantages(rewards)
                flat_prompts.extend([prompt_text] * len(candidates))
                flat_svgs.extend(candidates)
                flat_advantages.extend(adv)
                flat_rewards.extend(rewards)

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            n_items = len(flat_svgs)
            micro = max(1, args.microbatch_size)
            for start in range(0, n_items, micro):
                end = min(n_items, start + micro)
                policy_batch = build_policy_batch(
                    tokenizer,
                    flat_prompts[start:end],
                    flat_svgs[start:end],
                    args.max_seq_length,
                    model.device,
                )
                with torch.no_grad():
                    old_logp = selective_logprobs(model, policy_batch["input_ids"], policy_batch["attention_mask"], policy_batch["labels"]).detach()
                    ref_logp = selective_logprobs(ref_model, policy_batch["input_ids"], policy_batch["attention_mask"], policy_batch["labels"]).detach()
                new_logp = selective_logprobs(model, policy_batch["input_ids"], policy_batch["attention_mask"], policy_batch["labels"])
                adv = torch.tensor(flat_advantages[start:end], dtype=new_logp.dtype, device=new_logp.device)
                loss = grpo_loss(new_logp, old_logp, ref_logp, adv, args.clip_range, args.kl_coef)
                (loss / max(1, args.gradient_accumulation_steps)).backward()
                total_loss += float(loss.detach().cpu())
                if ((start // micro) + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            # Flush remaining grads.
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            log = {
                "step": global_step,
                "loss": total_loss / max(1, n_items),
                "reward_mean": sum(flat_rewards) / max(1, len(flat_rewards)),
                "reward_max": max(flat_rewards) if flat_rewards else 0.0,
                "reward_min": min(flat_rewards) if flat_rewards else 0.0,
            }
            with (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")
            if global_step % args.save_every == 0 and global_step > 0:
                model.save_pretrained(str(args.output_dir / f"step_{global_step}"))
                tokenizer.save_pretrained(str(args.output_dir / f"step_{global_step}"))
            global_step += 1
            pbar.update(1)
    pbar.close()
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
