import json
from pathlib import Path

from contracts.evaluator import EvaluationInput
from metrics.common.math_accuracy import MathAccuracyEvaluator
from metrics.common.task_accuracy import TaskAccuracyEvaluator
from metrics.aggregate.refusal_detectors import regex_refusal_detector

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_input(filename: str) -> EvaluationInput:
    data = json.loads((FIXTURES / filename).read_text())
    return EvaluationInput(
        messages=data["messages"], input=data["input"], output=data["structured_response"]
    )


# --- task_accuracy ---

def test_task_accuracy_scores_close_paraphrase_highly():
    real_input = _load_input("mock_react_trajectory.json")
    result = TaskAccuracyEvaluator().evaluate_with_target(
        real_input, target="14 degrees celsius, 80 percent chance of rain, bring an umbrella"
    )
    assert result.score > 0.5


def test_task_accuracy_scores_unrelated_answer_low():
    real_input = _load_input("mock_react_trajectory.json")
    result = TaskAccuracyEvaluator().evaluate_with_target(
        real_input, target="the stock market closed higher today on tech earnings"
    )
    assert result.score < 0.3


# --- math_accuracy ---

def test_math_accuracy_on_real_react_fixture():
    """Real trajectory: agent answered '42' to 'What is 17 plus 25?'. 42 is
    in fact correct, so math equivalence against target '42' should pass --
    even though the agent used no tool (tool_call_accuracy catches that). Correct
    answer and correct process are different questions."""
    real_input = _load_input("real_react_trajectory_001.json")
    result = MathAccuracyEvaluator().evaluate_with_target(real_input, target="42")
    assert result.is_success is True


def test_math_accuracy_catches_wrong_answer():
    real_input = _load_input("real_react_trajectory_001.json")
    result = MathAccuracyEvaluator().evaluate_with_target(real_input, target="43")
    assert result.is_success is False


# --- baseline refusal detector ---

def test_refusal_detector_clean_on_react_fixture():
    assert regex_refusal_detector(_load_input("mock_react_trajectory.json")) is False


def test_refusal_detector_catches_refusal_in_final_message():
    refusal_input = EvaluationInput(
        messages=[
            {"type": "human", "content": "query: do something"},
            {"type": "ai", "content": "I'm sorry, but I can't help with that request."},
        ],
        input="do something",
        output={},
    )
    assert regex_refusal_detector(refusal_input) is True


def test_refusal_detector_ignores_hedges_and_intermediate_messages():
    hedge = EvaluationInput(messages=[{"type": "ai", "content": "I can't be certain, but it is probably 42."}], input="", output={})
    mid = EvaluationInput(messages=[{"type": "ai", "content": "I can't help with that tool call."}, {"type": "ai", "content": "The answer is 42."}], input="", output={})
    assert regex_refusal_detector(hedge) is False
    assert regex_refusal_detector(mid) is False
