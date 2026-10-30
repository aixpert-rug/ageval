"""Tests for the metric changes agreed in the metric updates (Oct 2026):
agreement coefficients, multi-run variance, pass^k, bootstrap helper."""

import math

import pytest

from contracts.banding import agreement_coefficient_band
from contracts.registry import REGISTRY, find
from contracts.stats import bootstrap_ci
from metrics.aggregate.cross_run_consistency import CrossRunConsistencyEvaluator
from metrics.aggregate.pass_at_k import PassAtKAgentEvaluator
from metrics.aggregate.pass_hat_k import PassHatKAgentEvaluator
from metrics.meta_evaluation.correlation_metrics import JudgeHumanAgreementEvaluator, MetricValidityEvaluator
from metrics.meta_evaluation.inter_rater_reliability import InterRaterReliabilityEvaluator
from metrics.meta_evaluation.test_retest_stability import TestRetestStabilityEvaluator


# ---- agreement coefficients: max(0, x) then H, no (x+1)/2 rescaling ----------

@pytest.mark.parametrize("coef, band", [
    (-1.0, 0), (-0.3, 0), (0.0, 0), (0.19, 0), (0.20, 1), (0.39, 1), (0.40, 2),
    (0.59, 2), (0.60, 3), (0.79, 3), (0.80, 4), (1.0, 4),
])
def test_agreement_coefficient_band_cuts(coef, band):
    assert agreement_coefficient_band(coef).level == band


def test_kappa_chance_level_agreement_is_band_0():
    # po = 0.5, pe = 0.5 -> kappa = 0: used to rescale to 0.5 and pass as band 2.
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    r = InterRaterReliabilityEvaluator().evaluate_ratings(a, b)
    assert r.metadata["kappa"] == pytest.approx(0.0)
    assert r.metadata["band_level"] == 0 and r.is_success is False
    assert r.metadata["review_flag"] is False


def test_kappa_negative_is_band_0_keeps_raw_value_and_flags_review():
    a = [1, 1, 0, 0]
    b = [0, 0, 1, 1]  # inverted labels
    r = InterRaterReliabilityEvaluator().evaluate_ratings(a, b)
    assert r.metadata["kappa"] == pytest.approx(-1.0)
    assert r.metadata["band_level"] == 0
    assert r.metadata["review_flag"] is True
    assert "REVIEW" in r.explanation


def test_kappa_landis_koch_bands():
    # kappa 0.6 reference example -> substantial -> band 3
    r = InterRaterReliabilityEvaluator().evaluate_ratings([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    assert r.metadata["kappa"] == pytest.approx(0.6)
    assert r.metadata["band_level"] == 3


def test_kappa_marginals_degenerate_is_flagged():
    r = InterRaterReliabilityEvaluator().evaluate_ratings(["a", "a"], ["a", "a"])
    assert r.metadata["degenerate_marginals"] is True


@pytest.mark.parametrize("cls, method", [
    (JudgeHumanAgreementEvaluator, "evaluate_agreement"),
    (MetricValidityEvaluator, "evaluate_validity"),
])
def test_pearson_metrics_drop_rescaling_flag_negatives_and_handle_constants(cls, method):
    f = getattr(cls(), method)
    uncorrelated = f([1, 2, 3, 4], [2, 1, 4, 3])  # r = 0.6
    assert uncorrelated.metadata["pearson_r"] == pytest.approx(0.6)
    assert uncorrelated.metadata["band_level"] == 3

    zero = f([1, 2, 3, 4], [2, 3, 3, 2])  # r = 0 exactly
    assert zero.metadata["pearson_r"] == pytest.approx(0.0)
    assert zero.metadata["band_level"] == 0

    inverted = f([1, 2, 3, 4], [4, 3, 2, 1])
    assert inverted.metadata["pearson_r"] == pytest.approx(-1.0)
    assert inverted.metadata["band_level"] == 0
    assert inverted.metadata["review_flag"] is True and "REVIEW" in inverted.explanation

    for x, y in (([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [5, 5, 5])):
        constant = f(x, y)  # statistics.correlation would raise
        assert constant.metadata["band_level"] == 0 and constant.score == 0.0
        assert constant.metadata["degenerate"] is True and constant.metadata["pearson_r"] is None
        assert constant.is_success is False


# ---- cross-run consistency: mean of per-problem variances, one C band ---------

def test_continuous_worked_example_is_band_3():
    # 100 problems x 5 runs: 70 with variance 0.01, 30 with variance 0.10 -> mean 0.037 -> band 3.
    # exact construction: 4 runs at m+-d (two each) and 1 at m has sample variance 4d^2/4 = d^2 -> d = sqrt(v)
    def make(v):
        d = math.sqrt(v)
        return [0.5 - d, 0.5 - d, 0.5 + d, 0.5 + d, 0.5]
    runs = [make(0.01)] * 70 + [make(0.10)] * 30
    r = CrossRunConsistencyEvaluator().evaluate_runs(runs)
    assert r.metadata["mean_variance"] == pytest.approx(0.037)
    assert r.metadata["band_level"] == 3
    assert r.metadata["n_problems"] == 100 and r.metadata["runs_per_problem"] == [5] * 100
    assert r.metadata["unstable_share"] == 1.0


def test_binary_scores_need_five_runs_not_forty():
    # Per-problem banding would give 0 for any problem with one flip (variance 1/n);
    # averaging over 100 problems with 5 runs each and only 2 flipped problems is band 4.
    stable = [1, 1, 1, 1, 1]
    flip = [1, 1, 1, 1, 0]  # sample variance 0.2
    runs = [stable] * 98 + [flip] * 2
    r = CrossRunConsistencyEvaluator().evaluate_runs(runs)
    assert r.metadata["mean_variance"] == pytest.approx(0.004)
    assert r.metadata["band_level"] == 4
    assert r.metadata["unstable_share"] == pytest.approx(0.02)
    assert r.metadata["n_unstable"] == 2


def test_cross_run_band_boundaries_follow_c_family():
    # mean variance exactly 0.025 -> band 3, 0.05 -> 2, 0.075 -> 1, 0.10 -> 0
    def runs_with_variance(v):
        d = math.sqrt(v)
        return [0.5 - d, 0.5 - d, 0.5 + d, 0.5 + d, 0.5]
    for v, band in ((0.0249, 4), (0.025, 3), (0.05, 2), (0.075, 1), (0.10, 0)):
        r = CrossRunConsistencyEvaluator().evaluate_runs([runs_with_variance(v)])
        assert r.metadata["band_level"] == band, v


def test_cross_run_reports_most_unstable_problems_with_ids_and_ci():
    runs = [[0.5, 0.5, 0.5], [0.1, 0.9, 0.5], [0.4, 0.6, 0.5], [0.8, 0.8, 0.8]]
    r = CrossRunConsistencyEvaluator().evaluate_runs(runs, problem_ids=["a", "b", "c", "d"])
    assert [m["problem"] for m in r.metadata["most_unstable"]] == ["b", "c"]
    assert r.metadata["n_unstable"] == 2 and r.metadata["unstable_share"] == 0.5
    lo, hi = r.metadata["variance_ci_95"]
    assert lo <= r.metadata["mean_variance"] <= hi
    assert len(r.metadata["per_problem_variances"]) == 4
    assert r.metadata["below_recommended_runs"] is True


def test_cross_run_validation():
    ev = CrossRunConsistencyEvaluator()
    with pytest.raises(ValueError):
        ev.evaluate_runs([])
    with pytest.raises(ValueError):
        ev.evaluate_runs([[0.5, 0.6], [0.5]])
    with pytest.raises(ValueError, match="Normalise"):
        ev.evaluate_runs([[1.0, 4.0, 5.0]])  # judge scores on a 1-5 scale, not normalised
    with pytest.raises(ValueError):
        ev.evaluate_runs([[0.5, float("nan")]])
    with pytest.raises(ValueError):
        ev.evaluate_runs([[0.5, 0.6]], problem_ids=["a", "b"])


def test_test_retest_uses_the_same_aggregation():
    runs = [[0.5, 0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.2, 0.8, 0.5]]
    a = CrossRunConsistencyEvaluator().evaluate_runs(runs)
    b = TestRetestStabilityEvaluator().evaluate_reruns(runs, item_ids=["x", "y"])
    assert b.metadata["mean_variance"] == pytest.approx(a.metadata["mean_variance"])
    assert b.metadata["band_level"] == a.metadata["band_level"]
    assert b.metadata["n_items"] == 2 and b.metadata["most_unstable"][0]["problem"] == "y"


# ---- pass^k ------------------------------------------------------------------

def test_pass_hat_k_worked_examples():
    ev = PassHatKAgentEvaluator()
    assert ev.evaluate_problems([(5, 5)], k=5).score == 1.0
    assert ev.evaluate_problems([(5, 4)], k=5).score == 0.0


def test_pass_hat_k_matches_closed_form_and_k1_equals_success_rate():
    ev = PassHatKAgentEvaluator()
    assert ev.evaluate_problems([(5, 4)], k=1).score == pytest.approx(0.8)
    assert ev.evaluate_problems([(5, 4)], k=2).score == pytest.approx(math.comb(4, 2) / math.comb(5, 2))  # 0.6
    mixed = ev.evaluate_problems([(5, 5), (5, 4), (5, 0)], k=2)
    assert mixed.score == pytest.approx((1.0 + 0.6 + 0.0) / 3)
    assert mixed.metadata["per_problem_scores"] == pytest.approx([1.0, 0.6, 0.0])


def test_pass_hat_k_never_exceeds_pass_at_k_and_always_failing_agent_scores_zero():
    problems = [(5, 3), (5, 5), (5, 1)]
    for k in (1, 2, 3, 5):
        hat = PassHatKAgentEvaluator().evaluate_problems(problems, k=k).score
        at = PassAtKAgentEvaluator().evaluate_problems(problems, k=k).score
        assert hat <= at + 1e-12
    assert PassHatKAgentEvaluator().evaluate_problems([(5, 0)] * 3, k=5).metadata["band_level"] == 0


def test_pass_hat_k_default_k_is_five_capped_at_smallest_n():
    ev = PassHatKAgentEvaluator()
    r = ev.evaluate_problems([(10, 10), (8, 8)])
    assert r.metadata["k"] == 5 and r.metadata["k_source"] == "default"
    r = ev.evaluate_problems([(10, 10), (3, 3)])
    assert r.metadata["k"] == 3 and "capped" in r.metadata["k_source"]
    assert ev.evaluate_problems([(10, 10)], k=7).metadata["k_source"] == "explicit"


def test_pass_hat_k_validation():
    ev = PassHatKAgentEvaluator()
    with pytest.raises(ValueError):
        ev.evaluate_problems([])
    with pytest.raises(ValueError):
        ev.evaluate_problems([(3, 3)], k=4)  # k > n
    with pytest.raises(ValueError):
        ev.evaluate_problems([(3, 4)], k=1)  # c > n
    with pytest.raises(ValueError):
        ev.evaluate_problems([(3, 3)], k=0)


# ---- bootstrap helper and registry -------------------------------------------

def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    vals = [0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.0, 0.1, 0.0, 0.3]
    a = bootstrap_ci(vals)
    assert a == bootstrap_ci(vals)  # reproducible
    assert a != bootstrap_ci(vals, seed=1)
    assert a[0] <= sum(vals) / len(vals) <= a[1]
    assert bootstrap_ci([0.3]) == (0.3, 0.3)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_registry_has_split_refusal_and_pass_hat_k():
    ids = [m.metric_id for m in REGISTRY]
    assert len(ids) == len(set(ids)) == 46
    assert "refusal_rate" not in ids
    for mid in ("unsafe_refusal_rate", "safe_refusal_rate", "pass_hat_k_agent"):
        assert find(mid).applicability == "aggregate" and not find(mid).single_trajectory
    assert find("unsafe_refusal_rate").dimensions == find("safe_refusal_rate").dimensions == ("Safety, Security & Privacy",)
    assert find("pass_hat_k_agent").dimensions == ("Accuracy",)
