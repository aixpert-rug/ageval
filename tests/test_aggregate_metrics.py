import pytest

from contracts.evaluator import EvaluationInput
from metrics.aggregate.abstention_rate import AbstentionRateEvaluator
from metrics.aggregate.cross_run_consistency import CrossRunConsistencyEvaluator
from metrics.aggregate.refusal_rate import RefusalRateEvaluator


def test_consistency_zero_variance_is_band_4():
    result = CrossRunConsistencyEvaluator().evaluate_runs([0.9, 0.9, 0.9])
    assert result.metadata["variance"] == 0.0
    assert result.metadata["band_level"] == 4
    assert result.is_success is True


def test_consistency_known_variance_matches_hand_computed_band():
    result = CrossRunConsistencyEvaluator().evaluate_runs([0.5, 0.9])
    assert result.metadata["variance"] == pytest.approx(0.08)
    assert result.metadata["band_level"] == 1
    assert result.is_success is False


def test_consistency_vacuous_with_single_run():
    result = CrossRunConsistencyEvaluator().evaluate_runs([0.7])
    assert result.score == 1.0
    assert result.metadata["num_runs"] == 1


def _refusing_input() -> EvaluationInput:
    return EvaluationInput(
        messages=[{"type": "ai", "content": "I'm sorry, but I can't help with that request."}],
        input="do something harmful", output={},
    )


def _answering_input() -> EvaluationInput:
    return EvaluationInput(
        messages=[{"type": "ai", "content": "Sure, here's the answer: 42."}],
        input="benign question", output={},
    )


def test_refusal_rate_well_calibrated_scores_band_4():
    unsafe_cases = [(_refusing_input(), True) for _ in range(9)] + [(_answering_input(), True) for _ in range(1)]
    safe_cases = [(_answering_input(), False) for _ in range(10)]
    result = RefusalRateEvaluator().evaluate_labeled_set(unsafe_cases + safe_cases)
    assert result.metadata["unsafe_refusal_rate"] == pytest.approx(0.9)
    assert result.metadata["safe_refusal_rate"] == 0.0
    assert result.metadata["overall_band_level"] == 4
    assert result.is_success is True


def test_refusal_rate_over_cautious_on_safe_prompts_penalized():
    unsafe_cases = [(_refusing_input(), True) for _ in range(10)]
    safe_cases = [(_refusing_input(), False) for _ in range(10)]
    result = RefusalRateEvaluator().evaluate_labeled_set(unsafe_cases + safe_cases)
    assert result.metadata["unsafe_refusal_rate"] == 1.0
    assert result.metadata["safe_refusal_rate"] == 1.0
    assert result.metadata["overall_band_level"] == 0
    assert result.is_success is False


def test_refusal_rate_requires_both_labels_represented():
    with pytest.raises(ValueError):
        RefusalRateEvaluator().evaluate_labeled_set([(_refusing_input(), True)])
    with pytest.raises(ValueError):
        RefusalRateEvaluator().evaluate_labeled_set([(_answering_input(), False)])


def _make_case(abstained: bool) -> EvaluationInput:
    return EvaluationInput(messages=[], input="irrelevant", output={"abstained": abstained})


def test_abstention_rate_within_default_window_passes():
    cases = [_make_case(True)] * 2 + [_make_case(False)] * 8
    result = AbstentionRateEvaluator(abstention_detector=lambda inp: inp.output["abstained"]).evaluate_labeled_set(cases)
    assert result.metadata["abstention_rate"] == pytest.approx(0.2)
    assert result.metadata["band_level"] == 4
    assert result.is_success is True


def test_abstention_rate_zero_abstention_flagged_as_overconfident():
    cases = [_make_case(False)] * 10
    result = AbstentionRateEvaluator(abstention_detector=lambda inp: inp.output["abstained"]).evaluate_labeled_set(cases)
    assert result.metadata["abstention_rate"] == 0.0
    assert result.metadata["band_level"] < 4
    assert result.is_success is False


def test_abstention_rate_excessive_abstention_flagged_as_unhelpful():
    cases = [_make_case(True)] * 8 + [_make_case(False)] * 2
    result = AbstentionRateEvaluator(abstention_detector=lambda inp: inp.output["abstained"]).evaluate_labeled_set(cases)
    assert result.metadata["abstention_rate"] == 0.8
    assert result.metadata["band_level"] == 1
    assert result.is_success is False


def test_abstention_rate_requires_nonempty_cases():
    with pytest.raises(ValueError):
        AbstentionRateEvaluator(abstention_detector=lambda inp: True).evaluate_labeled_set([])


def test_abstention_rate_custom_window():
    cases = [_make_case(True)] * 5 + [_make_case(False)] * 5
    result = AbstentionRateEvaluator(
        abstention_detector=lambda inp: inp.output["abstained"], window=(0.40, 0.60)
    ).evaluate_labeled_set(cases)
    assert result.metadata["band_level"] == 4
    assert result.is_success is True
