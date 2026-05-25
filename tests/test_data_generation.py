from geosvg_rl.data.generator import make_plan
from geosvg_rl.data.svg_template import render_plan_to_svg
import random


def test_make_plan_and_svg():
    plan = make_plan(random.Random(13), family="pipeline")
    svg = render_plan_to_svg(plan)
    assert "<svg" in svg
    assert len(plan.nodes) >= 3
    assert len(plan.edges) >= 2
