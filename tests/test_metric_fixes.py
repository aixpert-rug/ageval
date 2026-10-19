import pytest

from contracts.evaluator import EvaluationInput
from contracts.governance import EscalationEvent, GovernanceTestCase
from metrics.common.math_accuracy import MathAccuracyEvaluator, is_numeric_target
from metrics.governance.prompt_injection_resistance import PromptInjectionResistanceEvaluator
from metrics.human_centricity.cognitive_load import CognitiveLoadEvaluator
from metrics.trajectory.tool_call_accuracy import ToolCallAccuracyEvaluator


def _calls(*calls):
    return EvaluationInput(
        messages=[{"type": "ai", "content": "", "tool_calls": [{"name": n, "args": a, "id": str(i)} for i, (n, a) in enumerate(calls)]}],
        input="", output=None)


# ---- tool_call_accuracy
def test_tool_calls_one_to_one_and_precision():
    ev = ToolCallAccuracyEvaluator()
    assert ev.evaluate_with_expected_calls(_calls(("add", {})), [{"name": "add"}, {"name": "add"}]).metadata["recall"] == 0.5
    r = ev.evaluate_with_expected_calls(_calls(("add", {}), ("sub", {}), ("mul", {}), ("div", {})), [{"name": "add"}])
    assert r.metadata["recall"] == 1.0 and r.metadata["precision"] == 0.25 and r.score == pytest.approx(0.4)
    r = ToolCallAccuracyEvaluator(penalize_extra=False).evaluate_with_expected_calls(
        _calls(("add", {}), ("sub", {})), [{"name": "add"}])
    assert r.score == 1.0


def test_tool_calls_args_order_and_ignored_terminal_tool():
    t17 = _calls(("add", {"a": 17, "b": 25}), ("finish_task", {"answer": "42"}))
    assert ToolCallAccuracyEvaluator().evaluate_with_expected_calls(t17, [{"name": "add", "args": {"a": 17, "b": 25}}]).score == 1.0
    assert ToolCallAccuracyEvaluator().evaluate_with_expected_calls(t17, [{"name": "add", "args": {"a": 1}}]).score == 0.0
    assert ToolCallAccuracyEvaluator(check_args=False).evaluate_with_expected_calls(t17, [{"name": "add", "args": {"a": 1}}]).score == 1.0
    swapped = _calls(("b", {}), ("a", {}))
    exp = [{"name": "a"}, {"name": "b"}]
    assert ToolCallAccuracyEvaluator().evaluate_with_expected_calls(swapped, exp).score == 1.0
    assert ToolCallAccuracyEvaluator(ordered=True).evaluate_with_expected_calls(swapped, exp).score == 0.5
    assert ToolCallAccuracyEvaluator().evaluate_with_expected_calls(_calls(), exp).score == 0.0


# ---- math_accuracy
@pytest.mark.parametrize("pred,target,ok", [
    ("50%", "0.5", True), ("0.5", "1/2", True), ("1/2", "50%", True), ("The answer is 50 %", "0.5", True),
    ("5%", "0.05", True), ("42", "42", True), ("43", "42", False), ("$1,000", "1000", True), ("-3", "-3", True),
    ("5-3", "-3", False),
])
def test_math_accuracy_equivalences(pred, target, ok):
    r = MathAccuracyEvaluator().evaluate_with_target(EvaluationInput(messages=[], input="", output=pred), target)
    assert r.is_success is ok


def test_is_numeric_target():
    class T:  # minimal TaskDefinition stand-in
        def __init__(self, target): self.target = target
    assert is_numeric_target(T("42")) and is_numeric_target(T("50%")) and is_numeric_target(T("1/2"))
    assert not is_numeric_target(T("Paris")) and not is_numeric_target(T("14 degrees, 80 percent")) and not is_numeric_target(T(None))


# ---- prompt injection
def _case(malicious, flagged, category=""):
    ev = [EscalationEvent("prompt_injection", "input", "r")] if flagged else []
    return GovernanceTestCase("x", malicious, ev, category)


def test_prompt_injection_flag_everything_is_not_a_pass():
    r = PromptInjectionResistanceEvaluator().evaluate_test_set([_case(True, True)] * 10 + [_case(False, True)] * 10)
    assert r.metadata["recall"] == 1.0 and r.metadata["false_positive_rate"] == 1.0
    assert r.is_success is False and r.score == 0.0


def test_prompt_injection_good_detector_ci_and_categories():
    cases = [_case(True, True, "direct")] * 9 + [_case(True, False, "indirect")] + [_case(False, False)] * 19 + [_case(False, True)]
    r = PromptInjectionResistanceEvaluator().evaluate_test_set(cases)
    assert r.is_success is True
    lo, hi = r.metadata["recall_ci95"]
    assert lo < 0.9 < hi
    assert r.metadata["recall_by_category"] == {"direct": 1.0, "indirect": 0.0}


def test_prompt_injection_needs_both_classes():
    with pytest.raises(ValueError):
        PromptInjectionResistanceEvaluator().evaluate_test_set([_case(True, True)])
    with pytest.raises(ValueError):
        PromptInjectionResistanceEvaluator().evaluate_test_set([_case(False, False)])


# ---- cognitive load
def test_cognitive_load_low_is_good_on_all_six_scales_no_reversal():
    ratings = {"mental_demand": 60, "physical_demand": 10, "temporal_demand": 40, "performance": 30, "effort": 50, "frustration": 20}
    r = CognitiveLoadEvaluator().evaluate_ratings(ratings)
    assert r.metadata["workload_0_100"] == pytest.approx(35.0)
    assert r.score == pytest.approx(0.65) and r.is_success is True
