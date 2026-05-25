from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geosvg_rl.data.schema import LayoutPlan

from .browser import BrowserSVGMeasurer, BrowserUnavailable
from .metrics import MetricResult, compute_metrics
from .reward import RewardResult, RewardWeights, compute_reward
from .xml_parser import parse_svg


@dataclass(slots=True)
class VerificationResult:
    metrics: MetricResult
    reward: RewardResult

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics.to_dict(), "reward": {"score": self.reward.reward, "components": self.reward.components}}


class GeoSVGVerifier:
    def __init__(
        self,
        *,
        use_browser: bool | str = "auto",
        timeout_ms: int = 5000,
        anchor_threshold_px: float = 12.0,
        text_padding_px: float = 6.0,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        self.use_browser = use_browser
        self.timeout_ms = timeout_ms
        self.anchor_threshold_px = anchor_threshold_px
        self.text_padding_px = text_padding_px
        self.reward_weights = reward_weights or RewardWeights()

    def verify(self, svg: str, plan: LayoutPlan | dict[str, Any], *, update: int | None = None) -> VerificationResult:
        if isinstance(plan, dict):
            plan = LayoutPlan.from_dict(plan)
        parsed = parse_svg(svg)
        if parsed.valid and self.use_browser is not False:
            parsed = self._maybe_patch_text_boxes_with_browser(svg, parsed)
        metrics = compute_metrics(
            parsed,
            plan,
            anchor_threshold_px=self.anchor_threshold_px,
            text_padding_px=self.text_padding_px,
        )
        weights = self.reward_weights.curriculum(update)
        reward = compute_reward(metrics, weights)
        return VerificationResult(metrics=metrics, reward=reward)

    def _maybe_patch_text_boxes_with_browser(self, svg: str, parsed):
        should_try = self.use_browser is True or self.use_browser == "auto"
        if not should_try:
            return parsed
        try:
            with BrowserSVGMeasurer(timeout_ms=self.timeout_ms) as measurer:
                boxes = measurer.measure_text(svg)
        except BrowserUnavailable:
            if self.use_browser is True:
                raise
            return parsed
        except Exception:
            if self.use_browser is True:
                raise
            return parsed
        text_elements = parsed.texts
        for i, box in enumerate(boxes):
            if i < len(text_elements):
                text_elements[i].bbox = box.as_bbox()
        return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one SVG against one layout plan JSON.")
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--use-browser", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args()
    svg = args.svg.read_text(encoding="utf-8")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    use_browser: str | bool = args.use_browser
    if use_browser == "true":
        use_browser = True
    elif use_browser == "false":
        use_browser = False
    result = GeoSVGVerifier(use_browser=use_browser).verify(svg, plan)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
