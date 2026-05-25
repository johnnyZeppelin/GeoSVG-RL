from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from geosvg_rl.utils.jsonl import iter_jsonl, write_jsonl
from geosvg_rl.verifier import GeoSVGVerifier

METRIC_KEYS = ["RSR", "GFR", "OAR", "EICR", "AAcc", "AEE", "TBR", "TPVR", "E-F1", "Clean"]


def verify_row(row: dict[str, Any], pred_field: str, use_browser: str | bool) -> dict[str, Any]:
    verifier = GeoSVGVerifier(use_browser=use_browser)
    svg = row.get(pred_field) or row.get("svg") or ""
    result = verifier.verify(svg, row["plan"])
    return {"id": row.get("id"), **result.to_dict()}


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {k: [] for k in METRIC_KEYS}
    rewards = []
    for r in results:
        md = r["metrics"]
        for k in METRIC_KEYS:
            metrics[k].append(float(md.get(k, 0.0)))
        rewards.append(float(r["reward"]["score"]))
    out = {k: mean(v) if v else 0.0 for k, v in metrics.items()}
    out["reward"] = mean(rewards) if rewards else 0.0
    out["n"] = len(results)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SVG predictions with GeoSVG verifier.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--pred-field", default="pred_svg")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details-out", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--use-browser", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args()
    use_browser: str | bool = args.use_browser
    if use_browser == "true":
        use_browser = True
    elif use_browser == "false":
        use_browser = False

    rows = list(iter_jsonl(args.jsonl))
    results: list[dict[str, Any]] = []
    if args.num_workers <= 1:
        for row in tqdm(rows, desc="verify"):
            results.append(verify_row(row, args.pred_field, use_browser))
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(verify_row, row, args.pred_field, use_browser) for row in rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="verify"):
                results.append(fut.result())
    summary = aggregate(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.details_out:
        write_jsonl(args.details_out, results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
