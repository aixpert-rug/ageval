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
    assert target_band_band(0.20, lo=0.10, hi=0.30).level == 4
    assert target_band_band(0.10, lo=0.10, hi=0.30).level == 4
    assert target_band_band(0.30, lo=0.10, hi=0.30).level == 4


def test_w_band_worked_examples():
    # band = max(0, 4 - ceil(distance / 0.10)), window [0.10, 0.30]
    assert target_band_band(0.35, lo=0.10, hi=0.30).level == 3
    assert target_band_band(0.45, lo=0.10, hi=0.30).level == 2
    assert target_band_band(0.75, lo=0.10, hi=0.30).level == 0


def test_w_band_drops_one_band_per_step_on_both_sides():
    lo, hi = 0.10, 0.30
    assert [target_band_band(v, lo, hi).level for v in (0.31, 0.40, 0.50, 0.60, 0.70)] == [3, 3, 2, 1, 0]
    assert [target_band_band(v, lo, hi).level for v in (0.09, 0.0)] == [3, 3]


def test_w_band_exact_multiples_are_not_pushed_down_by_float_error():
    # 0.40 - 0.30 == 0.10000000000000003 in floating point, still exactly one step.
    assert target_band_band(0.40, lo=0.10, hi=0.30).level == 3
    assert target_band_band(0.50, lo=0.10, hi=0.30).level == 2
    assert target_band_band(0.60, lo=0.10, hi=0.30).level == 1
    assert target_band_band(0.70, lo=0.10, hi=0.30).level == 0
    # (0.4 - 0.1) / 0.10 == 3.0000000000000004 -- the guard must snap it to 3.
    assert target_band_band(0.4, lo=0.0, hi=0.1).level == 1
    assert target_band_band(0.1, lo=0.4, hi=0.5).level == 1
    

def test_w_band_rejects_invalid_window():
    with pytest.raises(ValueError):
        target_band_band(0.5, lo=0.9, hi=0.1)
