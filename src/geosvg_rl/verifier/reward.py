from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metrics import MetricResult


@dataclass(slots=True)
class RewardWeights:
    exec: float = 1.0
    fit: float = 0.6
    overflow: float = 0.5
    anchor: float = 1.2
    text: float = 1.1
    padding: float = 0.5
    graph: float = 0.9
    clean: float = 0.3

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RewardWeights":
        if not d:
            return cls()
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def curriculum(self, update: int | None, local_updates: int = 500, ramp_updates: int = 500) -> "RewardWeights":
        if update is None:
            return self
        if update < local_updates:
            # Initially de-emphasize global packing, as described in the paper.
            return RewardWeights(
                exec=self.exec,
                fit=self.fit * 0.25,
                overflow=self.overflow * 0.25,
                anchor=self.anchor,
                text=self.text,
                padding=self.padding,
                graph=self.graph,
                clean=self.clean,
            )
        if update >= local_updates + ramp_updates:
            return self
        alpha = (update - local_updates) / max(1, ramp_updates)
        return RewardWeights(
            exec=self.exec,
            fit=self.fit * (0.25 + 0.75 * alpha),
            overflow=self.overflow * (0.25 + 0.75 * alpha),
            anchor=self.anchor,
            text=self.text,
            padding=self.padding,
            graph=self.graph,
            clean=self.clean,
        )


@dataclass(slots=True)
class RewardResult:
    reward: float
    components: dict[str, float] = field(default_factory=dict)


def compute_reward(metrics: MetricResult, weights: RewardWeights | None = None) -> RewardResult:
    w = weights or RewardWeights()
    components = {
        "exec": metrics.RSR,
        "fit": metrics.GFR,
        "overflow": -metrics.OAR,
        "anchor": metrics.AAcc - metrics.AEE,
        "text": metrics.TBR,
        "padding": -metrics.TPVR,
        "graph": metrics.EF1,
        "clean": metrics.Clean,
    }
    if metrics.RSR <= 0:
        # Binary gate: malformed/non-renderable SVG should not receive positive geometry reward.
        return RewardResult(reward=-1.0, components=components)
    reward = (
        w.exec * components["exec"]
        + w.fit * components["fit"]
        + w.overflow * components["overflow"]
        + w.anchor * components["anchor"]
        + w.text * components["text"]
        + w.padding * components["padding"]
        + w.graph * components["graph"]
        + w.clean * components["clean"]
    )
    denom = w.exec + w.fit + w.overflow + w.anchor + w.text + w.padding + w.graph + w.clean
    return RewardResult(reward=reward / max(1e-8, denom), components=components)


def group_relative_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var**0.5
    return [(r - mean) / (std + eps) for r in rewards]
