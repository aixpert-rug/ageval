import json
from pathlib import Path

from contracts.evaluator import EvaluationInput
from metrics.common.trajectory_consistency import TrajectoryConsistencyEvaluator
from metrics.common.math_accuracy import MathAccuracyEvaluator
from metrics.common.task_accuracy import TaskAccuracyEvaluator
from metrics.scanners.refusal import RefusalScanner
from metrics.scanners.reward_hacking import RewardHackingScanner

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
    even though TrajectoryConsistencyEvaluator (correctly) scores this trajectory
    0.0 for lack of grounding. Accuracy and faithfulness are different
    questions; this test and the faithfulness one on the same fixture
    should NOT agree, and that's expected, not a bug."""
    real_input = _load_input("real_react_trajectory_001.json")
    result = MathAccuracyEvaluator().evaluate_with_target(real_input, target="42")
    assert result.is_success is True


def test_math_accuracy_catches_wrong_answer():
    real_input = _load_input("real_react_trajectory_001.json")
    result = MathAccuracyEvaluator().evaluate_with_target(real_input, target="43")
    assert result.is_success is False


# --- refusal ---

def test_refusal_scanner_clean_on_react_fixture():
    result = RefusalScanner().evaluate(_load_input("mock_react_trajectory.json"))
    assert result.score == 1.0
    assert result.metadata["hits"] == []


def test_refusal_scanner_catches_refusal_pattern():
    refusal_input = EvaluationInput(
        messages=[
            {"type": "human", "content": "query: do something"},
            {"type": "ai", "content": "I'm sorry, but I can't help with that request."},
        ],
        input="do something",
        output={},
    )
    result = RefusalScanner().evaluate(refusal_input)
    assert result.score == 0.0
    assert result.is_success is False


# --- reward_hacking ---

def test_reward_hacking_flags_real_ungrounded_success():
    """The real '42' fixture, run through the composite check: the pattern
    reported success (structured_response validated) but Trajectory Consistency
    scored 0.0 on the same trajectory -- this is exactly the case
    RewardHackingScanner exists to catch."""
    real_input = _load_input("real_react_trajectory_001.json")
    trajectory_consistency_result = TrajectoryConsistencyEvaluator().evaluate(real_input)
    result = RewardHackingScanner().evaluate_composite(
        reported_success=True,
        trajectory_consistency_result=trajectory_consistency_result,
    )
    assert result.is_success is False
    assert result.metadata["trajectory_consistency_score"] == 0.0


def test_reward_hacking_passes_when_grounded():
    grounded_result = TrajectoryConsistencyEvaluator().evaluate(_load_input("mock_react_trajectory.json"))
    result = RewardHackingScanner().evaluate_composite(
        reported_success=True,
        trajectory_consistency_result=grounded_result,
    )
    assert result.is_success is True


def test_reward_hacking_skips_when_not_reported_successful():
    trajectory_consistency_result = TrajectoryConsistencyEvaluator().evaluate(_load_input("real_react_trajectory_001.json"))
    result = RewardHackingScanner().evaluate_composite(
        reported_success=False,
        trajectory_consistency_result=trajectory_consistency_result,
    )
    assert result.score == 1.0
    assert result.metadata["reported_success"] is False
