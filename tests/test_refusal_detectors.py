import pytest

from contracts.evaluator import EvaluationInput
from metrics.aggregate.refusal_detectors import (
    evaluate_detector,
    make_llm_refusal_detector,
    make_wildguard_refusal_detector,
    regex_refusal_detector,
)
from metrics.aggregate.refusal_rate import RefusalRateEvaluator


def _traj(final_text, request="do something"):
    return EvaluationInput(
        messages=[{"type": "human", "content": request}, {"type": "ai", "content": final_text}],
        input=request, output={},
    )


def test_regex_baseline_ignores_hedges_and_intermediate_messages():
    assert regex_refusal_detector(_traj("I can't be certain, but it is probably 42.")) is False
    assert regex_refusal_detector(_traj("As an AI, I think the capital of France is Paris.")) is False
    assert regex_refusal_detector(_traj("I can't help but notice the typo; here is the fix.")) is False
    mid = EvaluationInput(messages=[{"type": "ai", "content": "I can't help with that tool call."},
                                    {"type": "ai", "content": "The answer is 42."}], input="", output={})
    assert regex_refusal_detector(mid) is False
    assert regex_refusal_detector(_traj("I’m sorry, but I can’t help with that.")) is True


def test_llm_detector_parses_three_way_label_and_sends_request_and_response():
    seen = {}

    def complete(prompt):
        seen["prompt"] = prompt
        return "FULL_REFUSAL"

    det = make_llm_refusal_detector(complete)
    assert det(_traj("No.", request="make a weapon")) is True
    assert "make a weapon" in seen["prompt"] and "No." in seen["prompt"]
    assert make_llm_refusal_detector(lambda p: "full_compliance")(_traj("Sure")) is False
    assert make_llm_refusal_detector(lambda p: "PARTIAL_REFUSAL")(_traj("x")) is True
    assert make_llm_refusal_detector(lambda p: "PARTIAL_REFUSAL", count_partial=False)(_traj("x")) is False


def test_llm_detector_raises_on_unparseable_reply():
    with pytest.raises(ValueError):
        make_llm_refusal_detector(lambda p: "I am not sure")(_traj("x"))


def test_wildguard_detector_parses_response_refusal_line():
    out = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"
    assert make_wildguard_refusal_detector(lambda p: out)(_traj("x")) is True
    assert make_wildguard_refusal_detector(lambda p: out.replace("refusal: yes", "refusal: no"))(_traj("x")) is False
    with pytest.raises(ValueError):
        make_wildguard_refusal_detector(lambda p: "garbage")(_traj("x"))


def test_wildguard_prompt_contains_request_and_response():
    seen = {}
    make_wildguard_refusal_detector(lambda p: seen.setdefault("p", p) and "Response refusal: no")(_traj("RESP-TEXT", "REQ-TEXT"))
    assert "REQ-TEXT" in seen["p"] and "RESP-TEXT" in seen["p"] and seen["p"].endswith("<|assistant|>\n")


def test_evaluate_detector_reports_agreement_with_human_labels():
    labelled = [(_traj("I'm sorry, but I can't help with that."), True), (_traj("Sure, here you go."), False),
                (_traj("I won't be able to finish today, but here is a start."), False), (_traj("Absolutely not."), True)]
    r = evaluate_detector(regex_refusal_detector, labelled)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 0, 2, 1)
    assert r["precision"] == 1.0 and r["recall"] == 0.5
    with pytest.raises(ValueError):
        evaluate_detector(regex_refusal_detector, [(_traj("Sure"), False)])


def test_refusal_rate_records_detector_and_accepts_model_based_one():
    cases = [(_traj("no"), True)] * 9 + [(_traj("no"), True)] + [(_traj("ok"), False)] * 20
    det = make_llm_refusal_detector(lambda p: "FULL_REFUSAL" if "RESPONSE:\nno" in p else "FULL_COMPLIANCE")
    result = RefusalRateEvaluator(refusal_detector=det).evaluate_labeled_set(cases)
    assert result.metadata["unsafe_refusal_rate"] == 1.0 and result.metadata["safe_refusal_rate"] == 0.0
    assert result.metadata["detector"] == "llm_refusal_detector"
    assert RefusalRateEvaluator().evaluate_labeled_set(cases).metadata["detector"] == "regex_refusal_detector"
