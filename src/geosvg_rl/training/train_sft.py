from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geosvg_rl.models.hf_io import ModelLoadConfig, apply_lora, load_causal_lm, load_tokenizer
from geosvg_rl.models.prompts import format_plan_prompt, format_svg_prompt
from geosvg_rl.utils.jsonl import iter_jsonl
from geosvg_rl.utils.seed import seed_everything


@dataclass
class Example:
    prompt_text: str
    target_text: str


def build_examples(path: Path, task: str) -> list[Example]:
    examples: list[Example] = []
    for row in iter_jsonl(path):
        if task == "planner":
            x = format_plan_prompt(row["prompt"])
            y = json.dumps(row["plan"], ensure_ascii=False) + "\n"
        elif task == "generator":
            x = format_svg_prompt(row["prompt"], row["plan"])
            y = row["svg"] + "\n"
        else:
            raise ValueError(f"unknown task: {task}")
        examples.append(Example(x, y))
    return examples


class CausalSFTDataset:
    def __init__(self, examples: list[Example], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer(ex.prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(ex.target_text, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class DataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised warm-start training for planner or SVG generator.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--task", choices=["planner", "generator"], default="generator")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    seed_everything(args.seed)
    tokenizer = load_tokenizer(args.model_name)
    model = load_causal_lm(ModelLoadConfig(args.model_name, load_in_4bit=args.load_in_4bit), for_training=True)
    if not args.no_lora:
        model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    train_dataset = CausalSFTDataset(build_examples(args.train_jsonl, args.task), tokenizer, args.max_seq_length)
    eval_dataset = CausalSFTDataset(build_examples(args.val_jsonl, args.task), tokenizer, args.max_seq_length) if args.val_jsonl else None

    from transformers import Trainer, TrainingArguments

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        report_to=[],
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
