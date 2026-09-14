"""
Schema-violation rate -- Level 2 metric under Accuracy.

Applicability: common. Works on any pattern that produces a structured
output validated against an output_schema (all four patterns reviewed so
far do this via a `finish_task`-style tool). Was previously specified in
docs/pattern_metric_matrix.md but not implemented -- closing that gap.

What it measures: across a batch of runs, how often the agent's final
output actually failed schema validation (missing required fields, wrong
types, or -- in ReAct's case -- no `finish_task` call ever made). This is
a coarser, cheaper signal than Trajectory Consistency or Task Accuracy:
it doesn't ask whether the output is CORRECT, only whether it's even
well-formed enough to be scored at all. A model that reliably produces
valid-but-wrong outputs should score well here and poorly on
Task Accuracy -- those are different failure modes and this metric is
designed to isolate the first one.
"""

from __future__ import annotations

from typing import Any

from contracts.evaluator import EvaluationResult, WP3Evaluator


class SchemaViolationRateEvaluator(WP3Evaluator):
    """Fraction of a batch of runs whose structured_response is missing or invalid.

    score: 1.0 = no violations across the batch, 0.0 = every run violated.
        Higher is better, unlike the metric's name might suggest -- the
        score is framed as "conformance", not raw violation count, to
        stay consistent with this repo's higher-is-better convention.
    """

    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "schema_violation_rate"

    def evaluate_batch(self, structured_responses: list[Any]) -> EvaluationResult:
        """
        Args:
            structured_responses: one entry per run, taken directly from
                each run's `result["structured_response"]`. A `None`
                entry means validation failed for that run (matches
                ReactAgent._respond's own behaviour: it catches the
                validation exception and returns None rather than
                raising, so `None` is the correct signal to check for,
                not an exception).
        """
        if not structured_responses:
            raise ValueError("evaluate_batch requires at least one run in the batch.")

        violations = [i for i, r in enumerate(structured_responses) if r is None]
        conformance = 1 - (len(violations) / len(structured_responses))

        return EvaluationResult(
            score=conformance,
            explanation=(
                f"{len(structured_responses) - len(violations)}/{len(structured_responses)} "
                f"runs produced a valid structured output. Violating run indices: {violations}"
            ),
            is_success=conformance == 1.0,
            metadata={"batch_size": len(structured_responses), "violation_count": len(violations)},
        )

    def evaluate(self, input) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "SchemaViolationRateEvaluator operates on a batch of runs, not a "
            "single EvaluationInput. Call evaluate_batch(structured_responses) instead."
        )
