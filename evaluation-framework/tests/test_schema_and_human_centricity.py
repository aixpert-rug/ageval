import pytest

from metrics.common.schema_violation_rate import SchemaViolationRateEvaluator
from metrics.human_centricity.cognitive_load import CognitiveLoadEvaluator
from metrics.human_centricity.perceived_usability import PerceivedUsabilityEvaluator


# --- schema_violation_rate ---

def test_schema_violation_rate_all_valid():
    responses = [{"answer": "42"}, {"answer": "7"}, {"answer": "13"}]
    result = SchemaViolationRateEvaluator().evaluate_batch(responses)
    assert result.score == 1.0
    assert result.is_success is True
    assert result.metadata["violation_count"] == 0


def test_schema_violation_rate_catches_none_entries():
    """None entries simulate ReactAgent._respond's own failure mode: it
    catches the schema-validation exception and returns None rather than
    raising, so None is the correct thing to check for."""
    responses = [{"answer": "42"}, None, {"answer": "13"}, None]
    result = SchemaViolationRateEvaluator().evaluate_batch(responses)
    assert result.score == 0.5
    assert result.is_success is False
    assert result.metadata["violation_count"] == 2


def test_schema_violation_rate_requires_nonempty_batch():
    with pytest.raises(ValueError):
        SchemaViolationRateEvaluator().evaluate_batch([])


# --- perceived_usability (SUS) ---

def test_sus_perfect_score():
    """All positively-worded items rated 5 (strongly agree), all
    negatively-worded items rated 1 (strongly disagree) -- the
    best-possible response pattern. Known result: SUS = 100."""
    responses = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
    result = PerceivedUsabilityEvaluator().evaluate_responses(responses)
    assert result.metadata["sus_score_0_100"] == 100.0
    assert result.metadata["band"] == "excellent"


def test_sus_worst_score():
    """Inverse of the perfect pattern. Known result: SUS = 0."""
    responses = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
    result = PerceivedUsabilityEvaluator().evaluate_responses(responses)
    assert result.metadata["sus_score_0_100"] == 0.0
    assert result.metadata["band"] == "poor"


def test_sus_rejects_wrong_item_count():
    with pytest.raises(ValueError):
        PerceivedUsabilityEvaluator().evaluate_responses([3, 3, 3])


def test_sus_rejects_out_of_range_response():
    with pytest.raises(ValueError):
        PerceivedUsabilityEvaluator().evaluate_responses([3, 3, 3, 3, 3, 3, 3, 3, 3, 7])


def test_sus_batch_averages_participants():
    perfect = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
    worst = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
    result = PerceivedUsabilityEvaluator().evaluate_batch([perfect, worst])
    assert result.metadata["mean_sus_score_0_100"] == 50.0
    assert result.metadata["n_participants"] == 2


# --- cognitive_load (NASA-TLX) ---

def test_nasa_tlx_zero_workload():
    ratings = {"mental_demand": 0, "physical_demand": 0, "temporal_demand": 0, "performance": 0, "effort": 0, "frustration": 0}
    result = CognitiveLoadEvaluator().evaluate_ratings(ratings)
    assert result.metadata["workload_0_100"] == 0.0
    assert result.score == 1.0
    assert result.is_success is True


def test_nasa_tlx_max_workload():
    ratings = {"mental_demand": 100, "physical_demand": 100, "temporal_demand": 100, "performance": 100, "effort": 100, "frustration": 100}
    result = CognitiveLoadEvaluator().evaluate_ratings(ratings)
    assert result.metadata["workload_0_100"] == 100.0
    assert result.score == 0.0
    assert result.is_success is False


def test_nasa_tlx_weighted_vs_unweighted_can_differ():
    ratings = {"mental_demand": 100, "physical_demand": 0, "temporal_demand": 0, "performance": 0, "effort": 0, "frustration": 0}
    unweighted = CognitiveLoadEvaluator().evaluate_ratings(ratings)
    weights = {"mental_demand": 1.0, "physical_demand": 0.0, "temporal_demand": 0.0, "performance": 0.0, "effort": 0.0, "frustration": 0.0}
    weighted = CognitiveLoadEvaluator().evaluate_ratings(ratings, weights=weights)
    assert weighted.metadata["workload_0_100"] == 100.0
    assert unweighted.metadata["workload_0_100"] == pytest.approx(16.67, abs=0.01)
    assert weighted.metadata["workload_0_100"] > unweighted.metadata["workload_0_100"]


def test_nasa_tlx_rejects_missing_subscale():
    with pytest.raises(ValueError):
        CognitiveLoadEvaluator().evaluate_ratings({"mental_demand": 50})


def test_nasa_tlx_rejects_weights_not_summing_to_one():
    ratings = {"mental_demand": 50, "physical_demand": 50, "temporal_demand": 50, "performance": 50, "effort": 50, "frustration": 50}
    bad_weights = {"mental_demand": 0.5, "physical_demand": 0.5, "temporal_demand": 0.5, "performance": 0.0, "effort": 0.0, "frustration": 0.0}
    with pytest.raises(ValueError):
        CognitiveLoadEvaluator().evaluate_ratings(ratings, weights=bad_weights)
