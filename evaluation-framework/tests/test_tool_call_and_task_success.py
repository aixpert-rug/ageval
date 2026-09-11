import json
from pathlib import Path

import pytest

from contracts.evaluator import EvaluationInput
from metrics.common.task_success import TaskSuccessEvaluator
from metrics.common.trajectory_consistency import TrajectoryConsistencyEvaluator
from metrics.trajectory.log_completeness import LogCompletenessEvaluator
from metrics.trajectory.tool_call_accuracy import ToolCallAccuracyEvaluator

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_input(filename: str) -> EvaluationInput:
    data = json.loads((FIXTURES / filename).read_text())
    return EvaluationInput(messages=data["messages"], input=data["input"], output=data["structured_response"])


def test_tool_call_accuracy_on_real_bypassed_tool_fixture():
    result = ToolCallAccuracyEvaluator().evaluate_with_expected_calls(
        _load_input("real_react_trajectory_001.json"),
        expected_calls=[{"name": "add"}],
    )
    assert result.score == 0.0
    assert result.is_success is False
    assert result.metadata["actual_tool_names"] == ["finish_task"]


def test_tool_call_accuracy_on_mock_fixture_correct_tool_used():
    result = ToolCallAccuracyEvaluator().evaluate_with_expected_calls(
        _load_input("mock_react_trajectory.json"),
        expected_calls=[{"name": "get_weather"}],
    )
    assert result.score == 1.0
    assert result.is_success is True


def test_tool_call_accuracy_with_args_checking():
    evaluator = ToolCallAccuracyEvaluator(check_args=True)
    correct_args = evaluator.evaluate_with_expected_calls(
        _load_input("mock_react_trajectory.json"),
        expected_calls=[{"name": "get_weather", "args": {"location": "Groningen"}}],
    )
    assert correct_args.score == 1.0

    wrong_args = evaluator.evaluate_with_expected_calls(
        _load_input("mock_react_trajectory.json"),
        expected_calls=[{"name": "get_weather", "args": {"location": "Amsterdam"}}],
    )
    assert wrong_args.score == 0.0


def test_tool_call_accuracy_requires_nonempty_expectations():
    with pytest.raises(ValueError):
        ToolCallAccuracyEvaluator().evaluate_with_expected_calls(_load_input("mock_react_trajectory.json"), expected_calls=[])


def test_task_success_on_real_fixture_is_correct_despite_ungrounded():
    result = TaskSuccessEvaluator().evaluate_with_oracle(
        _load_input("real_react_trajectory_001.json"),
        oracle_fn=lambda output: output.get("answer") == "42",
    )
    assert result.score == 1.0
    assert result.is_success is True

    consistency_result = TrajectoryConsistencyEvaluator().evaluate(_load_input("real_react_trajectory_001.json"))
    tool_result = ToolCallAccuracyEvaluator().evaluate_with_expected_calls(
        _load_input("real_react_trajectory_001.json"), expected_calls=[{"name": "add"}]
    )
    assert consistency_result.score == 0.0
    assert tool_result.score == 0.0


def test_task_success_fails_on_wrong_answer():
    result = TaskSuccessEvaluator().evaluate_with_oracle(
        _load_input("real_react_trajectory_001.json"),
        oracle_fn=lambda output: output.get("answer") == "43",
    )
    assert result.score == 0.0
    assert result.is_success is False


def test_task_success_handles_oracle_exception_gracefully():
    result = TaskSuccessEvaluator().evaluate_with_oracle(
        _load_input("real_react_trajectory_001.json"),
        oracle_fn=lambda output: output["nonexistent_key"] == "42",
    )
    assert result.score == 0.0
    assert result.is_success is False
    assert "oracle_error" in result.metadata
