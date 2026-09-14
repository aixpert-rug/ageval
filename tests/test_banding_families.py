import pytest

from contracts.banding import (
    higher_is_better_rate_band,
    lower_is_better_continuous_band,
    lower_is_better_rate_band,
    ratio_to_budget_band,
    target_band_band,
)


def test_h_band_exact_boundaries():
    assert higher_is_better_rate_band(0.0).level == 0
    assert higher_is_better_rate_band(0.19).level == 0
    assert higher_is_better_rate_band(0.20).level == 1
    assert higher_is_better_rate_band(0.39).level == 1
    assert higher_is_better_rate_band(0.40).level == 2
    assert higher_is_better_rate_band(0.59).level == 2
    assert higher_is_better_rate_band(0.60).level == 3
    assert higher_is_better_rate_band(0.79).level == 3
    assert higher_is_better_rate_band(0.80).level == 4
    assert higher_is_better_rate_band(1.0).level == 4


def test_h_band_rejects_out_of_range():
    with pytest.raises(ValueError):
        higher_is_better_rate_band(1.5)


def test_l_band_exact_boundaries():
    assert lower_is_better_rate_band(0.0).level == 4
    assert lower_is_better_rate_band(0.049).level == 4
    assert lower_is_better_rate_band(0.05).level == 3
    assert lower_is_better_rate_band(0.099).level == 3
    assert lower_is_better_rate_band(0.10).level == 2
    assert lower_is_better_rate_band(0.199).level == 2
    assert lower_is_better_rate_band(0.20).level == 1
    assert lower_is_better_rate_band(0.399).level == 1
    assert lower_is_better_rate_band(0.40).level == 0
    assert lower_is_better_rate_band(1.0).level == 0


def test_l_band_differs_from_ratio_to_budget_at_same_input():
    """The exact confusion worth guarding against: a value of 0.10
    means something very different under each family."""
    l_result = lower_is_better_rate_band(0.10)
    r_result = ratio_to_budget_band(0.10, budget=1.0)
    assert l_result.level == 2   # L family: 0.10 is in [0.10, 0.20) -> band 2
    assert r_result.level == 4   # R family: ratio 0.10 <= 0.25 -> band 4
    assert l_result.level != r_result.level


def test_c_band_ece_example_from_table_5():
    assert lower_is_better_continuous_band(0.0, budget_ceiling=0.20).level == 4
    assert lower_is_better_continuous_band(0.05, budget_ceiling=0.20).level == 3
    assert lower_is_better_continuous_band(0.10, budget_ceiling=0.20).level == 2
    assert lower_is_better_continuous_band(0.15, budget_ceiling=0.20).level == 1
    assert lower_is_better_continuous_band(0.20, budget_ceiling=0.20).level == 0
    assert lower_is_better_continuous_band(1.0, budget_ceiling=0.20).level == 0


def test_c_band_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        lower_is_better_continuous_band(-1.0, budget_ceiling=0.20)
    with pytest.raises(ValueError):
        lower_is_better_continuous_band(0.1, budget_ceiling=0.0)


def test_w_band_inside_window_is_band_4():
    assert target_band_band(0.90, lo=0.85, hi=0.95).level == 4
    assert target_band_band(0.85, lo=0.85, hi=0.95).level == 4
    assert target_band_band(0.95, lo=0.85, hi=0.95).level == 4


def test_w_band_degrades_with_distance_from_window():
    just_below = target_band_band(0.80, lo=0.85, hi=0.95)
    far_below = target_band_band(0.10, lo=0.85, hi=0.95)
    assert just_below.level > far_below.level


def test_w_band_rejects_invalid_window():
    with pytest.raises(ValueError):
        target_band_band(0.5, lo=0.9, hi=0.1)
