import json
from pathlib import Path

from metrics.reflective.convergence_rate import ConvergenceRateEvaluator
from metrics.reflective.score_monotonicity import ScoreMonotonicityEvaluator

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_reflexion_fixture():
    return json.loads((FIXTURES / "mock_reflexion_trajectory.json").read_text())


# --- convergence_rate ---

def test_convergence_rate_on_reflexion_fixture():
    """Fixture converges on trial 3 of (assumed) a 3-trial budget -- late
    convergence, so score should be low-but-nonzero, not 0 (it DID converge)
    and not 1.0 (it took every available trial)."""
    data = _load_reflexion_fixture()
    is_success_per_trial = [t["eval_result"]["is_success"] for t in data["reflections"]]
    result = ConvergenceRateEvaluator().evaluate_trials(is_success_per_trial, max_iterations=3)
    assert result.is_success is True
    assert result.metadata["converged_at"] == 3
    # Converging on the very last trial of the budget -> minimum positive score.
    assert result.score == 1 / 3


def test_convergence_rate_scores_early_success_higher():
    result = ConvergenceRateEvaluator().evaluate_trials([True], max_iterations=3)
    assert result.score == 1.0
    assert result.metadata["converged_at"] == 1


def test_convergence_rate_zero_when_never_converged():
    result = ConvergenceRateEvaluator().evaluate_trials([False, False, False], max_iterations=3)
    assert result.score == 0.0
    assert result.is_success is False
    assert result.metadata["converged_at"] is None


def test_convergence_rate_rejects_inconsistent_budget():
    import pytest
    with pytest.raises(ValueError):
        ConvergenceRateEvaluator().evaluate_trials([False, False, True, True], max_iterations=2)


# --- score_monotonicity ---

def test_score_monotonicity_on_reflexion_fixture():
    """Fixture scores rise 0.2 -> 0.6 -> 0.95: genuinely monotonic improvement."""
    data = _load_reflexion_fixture()
    result = ScoreMonotonicityEvaluator().evaluate_trials(data["trial_scores"])
    assert result.score == 1.0
    assert result.is_success is True
    assert result.metadata["fully_monotonic"] is True


def test_score_monotonicity_catches_thrashing():
    """Negative control: scores go 0.9 -> 0.3 -> 0.7 -- higher final score
    than consistency.py's own thrashing test, but NOT monotonic (regresses
    on the middle trial). This is exactly the case consistency's stdev-based
    check can't reliably distinguish from genuine convergence."""
    result = ScoreMonotonicityEvaluator().evaluate_trials([0.9, 0.3, 0.7])
    assert result.is_success is False
    assert result.score == 0.5  # 1 of 2 transitions non-decreasing
    assert result.metadata["fully_monotonic"] is False


def test_score_monotonicity_vacuous_with_single_trial():
    result = ScoreMonotonicityEvaluator().evaluate_trials([0.8])
    assert result.score == 1.0
    assert result.metadata["num_trials"] == 1
