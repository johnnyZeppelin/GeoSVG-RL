from __future__ import annotations

import torch
import torch.nn.functional as F


def selective_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return per-sequence mean log-prob over non-ignored label positions.

    labels must contain -100 for prompt/padding tokens and target token ids for completion tokens.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    logp = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    logp = logp * mask
    return logp.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def grpo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, ref_logp: torch.Tensor, advantages: torch.Tensor, clip_range: float, kl_coef: float) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    pg_loss = -torch.minimum(unclipped, clipped).mean()
    approx_kl = (new_logp - ref_logp).mean()
    # Non-negative smooth KL approximation used in several PPO-style LLM implementations.
    kl = (torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1.0).mean()
    return pg_loss + kl_coef * kl + 0.0 * approx_kl
