import json
from pathlib import Path

import pytest

from contracts.evaluator import EvaluationInput
from metrics.common.subgoal_progress import SubgoalProgressEvaluator
from metrics.trajectory.step_efficiency import StepEfficiencyEvaluator

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_input(filename: str) -> EvaluationInput:
    data = json.loads((FIXTURES / filename).read_text())
    return EvaluationInput(messages=data["messages"], input=data["input"], output=data["structured_response"])


def test_step_efficiency_optimal_path_scores_perfectly():
    result = StepEfficiencyEvaluator().evaluate_with_optimal(
        _load_input("mock_react_trajectory.json"), optimal_steps=1
    )
    assert result.score == 1.0
    assert result.metadata["actual_steps"] == 1
    assert result.is_success is True


def test_step_efficiency_penalizes_detours():
    result = StepEfficiencyEvaluator(step_counter_fn=lambda messages: 4).evaluate_with_optimal(
        _load_input("mock_react_trajectory.json"), optimal_steps=1
    )
    assert result.score == 0.25
    assert result.is_success is False


def test_step_efficiency_never_exceeds_one_even_with_fewer_actual_steps():
    result = StepEfficiencyEvaluator(step_counter_fn=lambda messages: 1).evaluate_with_optimal(
        _load_input("mock_react_trajectory.json"), optimal_steps=3
    )
    assert result.score == 1.0


def test_step_efficiency_zero_steps_and_zero_optimal_is_vacuously_perfect():
    result = StepEfficiencyEvaluator(step_counter_fn=lambda messages: 0).evaluate_with_optimal(
        _load_input("mock_react_trajectory.json"), optimal_steps=0
    )
    assert result.score == 1.0


def test_step_efficiency_refuses_undefined_ratio_on_real_bypassed_tool_fixture():
    with pytest.raises(ValueError, match="undefined ratio"):
        StepEfficiencyEvaluator().evaluate_with_optimal(
            _load_input("real_react_trajectory_001.json"), optimal_steps=1
        )


def test_step_efficiency_rejects_negative_optimal():
    with pytest.raises(ValueError):
        StepEfficiencyEvaluator().evaluate_with_optimal(_load_input("mock_react_trajectory.json"), optimal_steps=-1)


def test_subgoal_progress_all_achieved():
    result = SubgoalProgressEvaluator().evaluate_with_subgoals(
        _load_input("mock_react_trajectory.json"),
        subgoal_checks=[
            lambda output: "summary" in output,
            lambda output: output.get("bring_umbrella") is True,
        ],
    )
    assert result.score == 1.0
    assert result.metadata["achieved_count"] == 2
    assert result.is_success is True


def test_subgoal_progress_partial_credit():
    result = SubgoalProgressEvaluator().evaluate_with_subgoals(
        _load_input("mock_react_trajectory.json"),
        subgoal_checks=[
            lambda output: "summary" in output,
            lambda output: "nonexistent_field" in output,
            lambda output: output.get("bring_umbrella") is True,
            lambda output: output.get("temperature_kelvin") == 287,
        ],
    )
    assert result.score == 0.5
    assert result.metadata["achieved_count"] == 2
    assert result.metadata["total_subgoals"] == 4


def test_subgoal_progress_handles_checker_exception_as_not_achieved():
    result = SubgoalProgressEvaluator().evaluate_with_subgoals(
        _load_input("mock_react_trajectory.json"),
        subgoal_checks=[
            lambda output: output["summary"] == "14C, 80% rain chance, bring an umbrella",
            lambda output: output["this_key_does_not_exist"] == "anything",
        ],
    )
    assert result.score == 0.5
    assert result.metadata["subgoal_results"][1]["error"] is not None


def test_subgoal_progress_requires_nonempty_checklist():
    with pytest.raises(ValueError):
        SubgoalProgressEvaluator().evaluate_with_subgoals(_load_input("mock_react_trajectory.json"), subgoal_checks=[])
