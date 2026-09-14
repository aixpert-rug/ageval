import pytest

from metrics.meta_evaluation.contamination import ContaminationEvaluator
from metrics.meta_evaluation.correlation_metrics import JudgeHumanAgreementEvaluator, MetricValidityEvaluator
from metrics.meta_evaluation.discriminative_power import DiscriminativePowerEvaluator
from metrics.meta_evaluation.inter_rater_reliability import InterRaterReliabilityEvaluator
from metrics.meta_evaluation.judge_bias import JudgeBiasEvaluator
from metrics.meta_evaluation.sensitivity import SensitivityEvaluator
from metrics.meta_evaluation.test_retest_stability import TestRetestStabilityEvaluator


def test_judge_human_agreement_perfect_correlation():
    result = JudgeHumanAgreementEvaluator().evaluate_agreement(
        judge_scores=[1, 2, 3, 4, 5], human_scores=[2, 4, 6, 8, 10]
    )
    assert result.metadata["pearson_r"] == pytest.approx(1.0)
    assert result.metadata["band_level"] == 4


def test_judge_human_agreement_no_correlation():
    result = JudgeHumanAgreementEvaluator().evaluate_agreement(
        judge_scores=[1, 2, 3, 4, 5], human_scores=[3, 1, 4, 1, 5]
    )
    assert -0.5 < result.metadata["pearson_r"] < 0.5


def test_judge_human_agreement_requires_matched_lengths():
    with pytest.raises(ValueError):
        JudgeHumanAgreementEvaluator().evaluate_agreement([1, 2], [1, 2, 3])


def test_metric_validity_uses_same_correlation_math():
    result = MetricValidityEvaluator().evaluate_validity(
        metric_scores=[1, 2, 3, 4, 5], criterion_scores=[2, 4, 6, 8, 10]
    )
    assert result.metadata["pearson_r"] == pytest.approx(1.0)


def test_inter_rater_kappa_matches_hand_verified_reference():
    rater_a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    rater_b = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    result = InterRaterReliabilityEvaluator().evaluate_ratings(rater_a, rater_b)
    assert result.metadata["observed_agreement"] == pytest.approx(0.8)
    assert result.metadata["expected_agreement"] == pytest.approx(0.5)
    assert result.metadata["kappa"] == pytest.approx(0.6)


def test_inter_rater_perfect_agreement():
    result = InterRaterReliabilityEvaluator().evaluate_ratings(["a", "b", "a"], ["a", "b", "a"])
    assert result.metadata["kappa"] == pytest.approx(1.0)
    assert result.metadata["band_level"] == 4


def test_inter_rater_requires_matched_lengths():
    with pytest.raises(ValueError):
        InterRaterReliabilityEvaluator().evaluate_ratings([1, 2], [1])


def test_discriminative_power_clearly_separated_systems():
    scores = {"sys_a": [0.9, 0.85, 0.95], "sys_b": [0.5, 0.45, 0.55], "sys_c": [0.1, 0.15, 0.05]}
    result = DiscriminativePowerEvaluator().evaluate_groups(scores)
    assert result.metadata["eta_squared"] > 0.9
    assert result.metadata["band_level"] == 4


def test_discriminative_power_indistinguishable_systems():
    scores = {"sys_a": [0.5, 0.51, 0.49], "sys_b": [0.5, 0.52, 0.48], "sys_c": [0.5, 0.49, 0.51]}
    result = DiscriminativePowerEvaluator().evaluate_groups(scores)
    assert result.metadata["eta_squared"] == pytest.approx(0.0)
    assert result.metadata["band_level"] == 0


def test_discriminative_power_requires_at_least_two_systems():
    with pytest.raises(ValueError):
        DiscriminativePowerEvaluator().evaluate_groups({"sys_a": [0.5]})


def test_test_retest_stability_zero_variance():
    result = TestRetestStabilityEvaluator().evaluate_reruns([0.8, 0.8, 0.8])
    assert result.metadata["variance"] == 0.0
    assert result.metadata["band_level"] == 4


def test_test_retest_stability_vacuous_single_rerun():
    result = TestRetestStabilityEvaluator().evaluate_reruns([0.8])
    assert result.score == 1.0


def test_sensitivity_detects_all_known_gaps():
    cases = [(0.9, 0.3, True), (0.2, 0.8, False), (0.95, 0.1, True)]
    result = SensitivityEvaluator().evaluate_known_gaps(cases)
    assert result.metadata["sensitivity_rate"] == 1.0
    assert result.metadata["band_level"] == 4


def test_sensitivity_misses_known_gaps():
    cases = [(0.3, 0.9, True), (0.8, 0.2, False)]
    result = SensitivityEvaluator().evaluate_known_gaps(cases)
    assert result.metadata["sensitivity_rate"] == 0.0
    assert result.is_success is False


def test_sensitivity_requires_nonempty_cases():
    with pytest.raises(ValueError):
        SensitivityEvaluator().evaluate_known_gaps([])


def test_contamination_no_overlap_passes():
    result = ContaminationEvaluator().evaluate_overlap(train_items={"q1", "q2"}, test_items={"q3", "q4"})
    assert result.metadata["contamination_rate"] == 0.0
    assert result.metadata["is_vetoed"] is False
    assert result.metadata["band_level"] == 4


def test_contamination_at_veto_threshold_triggers_veto():
    train = {"q1", "q2"}
    test = {"q1", "q2", "q3", "q4", "q5"}
    result = ContaminationEvaluator().evaluate_overlap(train_items=train, test_items=test)
    assert result.metadata["contamination_rate"] == pytest.approx(0.4)
    assert result.metadata["is_vetoed"] is True
    assert "VETO TRIGGERED" in result.explanation


def test_contamination_evaluate_measurement_accepts_precomputed_rate():
    result = ContaminationEvaluator().evaluate_measurement(0.5)
    assert result.metadata["is_vetoed"] is True


def test_contamination_requires_nonempty_test_set():
    with pytest.raises(ValueError):
        ContaminationEvaluator().evaluate_overlap(train_items={"q1"}, test_items=set())


def test_judge_bias_no_shift_is_unbiased():
    pairs = [(0.8, 0.8), (0.5, 0.5), (0.3, 0.3)]
    result = JudgeBiasEvaluator().evaluate_controlled_pairs(pairs)
    assert result.metadata["mean_shift"] == 0.0
    assert result.metadata["band_level"] == 4


def test_judge_bias_large_shift_flagged():
    pairs = [(0.9, 0.3), (0.8, 0.2)]
    result = JudgeBiasEvaluator().evaluate_controlled_pairs(pairs)
    assert result.metadata["mean_shift"] == pytest.approx(0.6)
    assert result.is_success is False


def test_judge_bias_requires_nonempty_pairs():
    with pytest.raises(ValueError):
        JudgeBiasEvaluator().evaluate_controlled_pairs([])
