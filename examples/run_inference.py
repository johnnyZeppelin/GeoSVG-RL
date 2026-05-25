from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geosvg_rl.data.generator import FAMILY_BUILDERS, make_prompt  # noqa: E402
from geosvg_rl.data.schema import metadata_from_plan  # noqa: E402
from geosvg_rl.data.svg_template import render_plan_to_svg  # noqa: E402
from geosvg_rl.verifier import GeoSVGVerifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based plan+SVG inference demo.")
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.prompt_json.read_text(encoding="utf-8"))
    rng = random.Random(13)
    family = payload.get("family", "pipeline")
    builder = FAMILY_BUILDERS.get(family, FAMILY_BUILDERS["pipeline"])
    plan = builder(rng)
    prompt = payload.get("prompt") or make_prompt(plan, rng)
    svg = render_plan_to_svg(plan, title=prompt)
    metrics = GeoSVGVerifier(use_browser="auto").verify(svg, plan).to_dict()

    (args.out_dir / "example.svg").write_text(svg, encoding="utf-8")
    (args.out_dir / "example_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    (args.out_dir / "example_metadata.json").write_text(json.dumps(metadata_from_plan(plan), indent=2), encoding="utf-8")
    (args.out_dir / "example_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'example.svg'}")
    print(json.dumps(metrics["metrics"], indent=2))


if __name__ == "__main__":
    main()
