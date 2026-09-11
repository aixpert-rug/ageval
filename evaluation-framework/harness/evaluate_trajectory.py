"""
The concrete answer to "how does this framework actually get used with
an agent": given ONE captured trajectory (an EvaluationInput -- the
output of actually running a WP4 pattern, or a fixture like
generate_fixture.py produces) and an optional TaskDefinition (gold
labels), runs every applicable single-trajectory metric it has enough
data for, and returns a structured report.

Deliberately does NOT run the agent itself -- that stays WP4's job (or
a capture script's, kept separate for the exact reason this repo has
maintained the WP3/WP4 boundary throughout: evaluation logic is public,
agent internals are private). This harness starts from an
ALREADY-CAPTURED trajectory.

Also deliberately does NOT attempt aggregate/reflective/efficiency/
human_centricity/probabilistic/meta_evaluation metrics -- those need
data from MULTIPLE runs or a different source entirely (survey
responses, logprobs, pre-measured values), not one trajectory. See
`run_batch_metrics` below for a sketch of how that second orchestration
layer would work, not fully built here.
"""

from __future__ import annotations

from contracts.evaluator import EvaluationInput, EvaluationResult
from contracts.registry import MetricSpec, single_trajectory_metrics
from contracts.task_definition import TaskDefinition


def _can_run(spec: MetricSpec, task_def: TaskDefinition | None) -> bool:
    if not spec.requires:
        return True
    if task_def is None:
        return False
    return all(getattr(task_def, field, None) is not None for field in spec.requires)


def _call_metric(spec: MetricSpec, trajectory: EvaluationInput, task_def: TaskDefinition | None) -> EvaluationResult:
    if spec.metric_id == "toxicity":
        instance = spec.cls(toxicity_classifier=task_def.toxicity_classifier)
        return instance.evaluate(trajectory)

    instance = spec.cls()
    method = getattr(instance, spec.entry_point)

    if not spec.requires:
        return method(trajectory)
    if spec.requires == ("target",):
        return method(trajectory, target=task_def.target)
    if spec.requires == ("expected_tool_calls",):
        return method(trajectory, expected_calls=task_def.expected_tool_calls)
    if spec.requires == ("oracle_fn",):
        return method(trajectory, oracle_fn=task_def.oracle_fn)
    if spec.requires == ("subgoal_checks",):
        return method(trajectory, subgoal_checks=task_def.subgoal_checks)
    if spec.requires == ("optimal_steps",):
        return method(trajectory, optimal_steps=task_def.optimal_steps)

    raise NotImplementedError(f"No call convention wired up for requires={spec.requires} (metric_id={spec.metric_id!r}).")


def evaluate_trajectory(
    trajectory: EvaluationInput,
    task_def: TaskDefinition | None = None,
    skip_metric_ids: tuple[str, ...] = (),
) -> dict[str, EvaluationResult | Exception]:
    """
    Returns:
        dict mapping metric_id -> EvaluationResult for every metric
        that WAS runnable and ran without raising, or -> the raised
        Exception for any that were runnable but failed -- failures are
        captured, not propagated, so one broken metric doesn't kill the
        whole report.
    """
    report: dict[str, EvaluationResult | Exception] = {}

    for spec in single_trajectory_metrics():
        if spec.metric_id in skip_metric_ids:
            continue
        if spec.cls is None:
            continue
        if not _can_run(spec, task_def):
            continue
        try:
            report[spec.metric_id] = _call_metric(spec, trajectory, task_def)
        except Exception as e:
            report[spec.metric_id] = e

    return report


def print_report(report: dict[str, EvaluationResult | Exception]) -> None:
    for metric_id, result in report.items():
        if isinstance(result, Exception):
            print(f"  {metric_id:30s} FAILED: {type(result).__name__}: {result}")
        else:
            status = "PASS" if result.is_success else "FAIL"
            print(f"  {metric_id:30s} [{status}] score={result.score:.3f}  {result.explanation[:80]}")
