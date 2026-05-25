from geosvg_rl.verifier.reward import group_relative_advantages


def test_group_relative_advantages_centered():
    adv = group_relative_advantages([1.0, 2.0, 3.0])
    assert abs(sum(adv)) < 1e-6
    assert adv[-1] > adv[0]
