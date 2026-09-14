import json
from pathlib import Path

import pytest

from contracts.evaluator import EvaluationInput
from contracts.registry import REGISTRY, find, metrics_for_applicability, single_trajectory_metrics
from contracts.task_definition import TaskDefinition
from harness.evaluate_trajectory import evaluate_trajectory

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_input(filename: str) -> EvaluationInput:
    data = json.loads((FIXTURES / filename).read_text())
    return EvaluationInput(messages=data["messages"], input=data["input"], output=data["structured_response"])


def test_registry_has_no_duplicate_metric_ids():
    ids = [m.metric_id for m in REGISTRY]
    assert len(ids) == len(set(ids))


def test_registry_find_returns_correct_spec():
    spec = find("trajectory_consistency")
    assert spec.applicability == "common"


def test_registry_find_raises_on_unknown_id():
    with pytest.raises(KeyError):
        find("not_a_real_metric")


def test_metrics_for_applicability_filters_correctly():
    aggregate_metrics = metrics_for_applicability("aggregate")
    assert all(m.applicability == "aggregate" for m in aggregate_metrics)
    assert len(aggregate_metrics) >= 9


def test_single_trajectory_metrics_excludes_batch_only_ones():
    single = single_trajectory_metrics()
    single_ids = {m.metric_id for m in single}
    assert "trajectory_consistency" in single_ids
    assert "cross_run_consistency" not in single_ids
    assert "latency" not in single_ids


def test_evaluate_trajectory_with_no_task_def_runs_only_label_free_metrics():
    trajectory = _load_input("real_react_trajectory_001.json")
    report = evaluate_trajectory(trajectory, task_def=None)
    ids = set(report.keys())
    assert ids == {"trajectory_consistency", "log_completeness", "refusal"}


def test_evaluate_trajectory_no_labels_produces_expected_real_result():
    trajectory = _load_input("real_react_trajectory_001.json")
    report = evaluate_trajectory(trajectory, task_def=None)
    assert report["trajectory_consistency"].score == 0.0
    assert report["trajectory_consistency"].is_success is False


def _full_task_def() -> TaskDefinition:
    return TaskDefinition(
        task_id="real_001_addition",
        input_query="What is 17 plus 25?",
        target="42",
        expected_tool_calls=[{"name": "add"}],
        oracle_fn=lambda output: output.get("answer") == "42",
        optimal_steps=1,
    )


def test_evaluate_trajectory_with_full_task_def_runs_more_metrics():
    trajectory = _load_input("real_react_trajectory_001.json")
    report = evaluate_trajectory(trajectory, _full_task_def(), skip_metric_ids=("bertscore", "meteor"))
    assert "task_accuracy" in report
    assert "tool_call_accuracy" in report
    assert "task_success" in report
    assert len(report) > 3


def test_evaluate_trajectory_reveals_correct_but_ungrounded_pattern():
    trajectory = _load_input("real_react_trajectory_001.json")
    report = evaluate_trajectory(trajectory, _full_task_def(), skip_metric_ids=("bertscore", "meteor"))
    assert report["task_success"].is_success is True
    assert report["task_accuracy"].is_success is True
    assert report["trajectory_consistency"].is_success is False
    assert report["tool_call_accuracy"].is_success is False


def test_evaluate_trajectory_captures_exceptions_rather_than_raising():
    trajectory = _load_input("real_react_trajectory_001.json")
    report = evaluate_trajectory(trajectory, _full_task_def(), skip_metric_ids=("bertscore", "meteor"))
    assert "step_efficiency" in report
    assert isinstance(report["step_efficiency"], Exception)
    assert report["task_success"].score == 1.0


def test_evaluate_trajectory_skip_metric_ids_excludes_requested_metrics():
    trajectory = _load_input("real_react_trajectory_001.json")
    report = evaluate_trajectory(trajectory, task_def=None, skip_metric_ids=("refusal",))
    assert "refusal" not in report
    assert "trajectory_consistency" in report


def test_evaluate_trajectory_on_well_grounded_trajectory_passes_consistency():
    trajectory = _load_input("mock_react_trajectory.json")
    report = evaluate_trajectory(trajectory, task_def=None)
    assert report["trajectory_consistency"].is_success is True
    assert report["log_completeness"].score == 1.0
