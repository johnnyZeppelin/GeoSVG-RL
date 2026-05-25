from geosvg_rl.data.generator import make_plan
from geosvg_rl.data.svg_template import render_plan_to_svg
from geosvg_rl.verifier import GeoSVGVerifier
import random


def test_reference_svg_scores_high():
    plan = make_plan(random.Random(13), family="pipeline")
    svg = render_plan_to_svg(plan)
    res = GeoSVGVerifier(use_browser=False).verify(svg, plan)
    assert res.metrics.RSR == 1.0
    assert res.metrics.GFR == 1.0
    assert res.metrics.AAcc > 0.9
    assert res.metrics.EF1 > 0.9
