import pytest

from contracts.banding import ratio_to_budget_band
from metrics.efficiency.cost_per_query import CostPerQueryEvaluator
from metrics.efficiency.latency import LatencyEvaluator


def test_band_exact_boundaries_lower_is_better():
    assert ratio_to_budget_band(0.0, 100).level == 4
    assert ratio_to_budget_band(25.0, 100).level == 4
    assert ratio_to_budget_band(25.01, 100).level == 3
    assert ratio_to_budget_band(50.0, 100).level == 3
    assert ratio_to_budget_band(75.0, 100).level == 2
    assert ratio_to_budget_band(100.0, 100).level == 1
    assert ratio_to_budget_band(100.01, 100).level == 0


def test_band_labels_match_levels():
    assert ratio_to_budget_band(0.0, 100).label == "Excellent"
    assert ratio_to_budget_band(200.0, 100).label == "Unacceptable"


def test_band_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        ratio_to_budget_band(10.0, 0)


def test_band_higher_is_better_mirrors_correctly():
    assert ratio_to_budget_band(100.0, 100, lower_is_better=False).level == 4
    assert ratio_to_budget_band(0.0, 100, lower_is_better=False).level == 0
    assert ratio_to_budget_band(75.0, 100, lower_is_better=False).level == 3


def test_latency_within_budget_scores_well():
    result = LatencyEvaluator().evaluate_measurement(elapsed_seconds=10.0, budget_seconds=100.0)
    assert result.metadata["band_level"] == 4
    assert result.is_success is True


def test_latency_over_budget_fails():
    result = LatencyEvaluator().evaluate_measurement(elapsed_seconds=150.0, budget_seconds=100.0)
    assert result.metadata["band_level"] == 0
    assert result.is_success is False


def test_latency_rejects_negative_elapsed_time():
    with pytest.raises(ValueError):
        LatencyEvaluator().evaluate_measurement(elapsed_seconds=-1.0, budget_seconds=100.0)


def test_cost_within_budget_scores_well():
    result = CostPerQueryEvaluator().evaluate_measurement(cost=0.01, budget=0.05)
    assert result.metadata["band_level"] == 4
    assert result.is_success is True


def test_cost_over_budget_fails():
    result = CostPerQueryEvaluator().evaluate_measurement(cost=0.10, budget=0.05)
    assert result.metadata["band_level"] == 0
    assert result.is_success is False
