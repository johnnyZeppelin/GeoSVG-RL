from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelLoadConfig:
    model_name: str
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(config: ModelLoadConfig, *, for_training: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = getattr(torch, config.torch_dtype) if hasattr(torch, config.torch_dtype) else torch.bfloat16
    kwargs = {"torch_dtype": dtype, "trust_remote_code": config.trust_remote_code}
    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "auto" if not for_training else None
    return AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)


def apply_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules: list[str] | None = None):
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
