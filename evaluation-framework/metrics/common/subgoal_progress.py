"""
Sub-goal / progress -- Level 2 metric under Accuracy (Vector's "Sg").

Applicability: common. Fractional-credit sibling of task_success.py:
where TaskSuccessEvaluator asks one binary question ("was the goal
achieved"), this asks "what fraction of a labeled sub-goal checklist was
achieved" -- for tasks with a natural decomposition into checkpoints
(e.g. "found the file" / "read its contents" / "extracted the right
field" / "returned it correctly"), giving partial credit for genuine
partial progress rather than an all-or-nothing score.

Deliberately pluggable, same reasoning as task_success.py: sub-goal
decomposition is task-specific and can't be inferred generically from
output text.
"""

from __future__ import annotations

from typing import Any, Callable

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator


class SubgoalProgressEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "subgoal_progress"

    def evaluate_with_subgoals(self, input: EvaluationInput, subgoal_checks: list[Callable[[Any], bool]]) -> EvaluationResult:
        """
        Args:
            subgoal_checks: list of task-specific checker functions, each
                taking the agent's final output (input.output) and
                returning True if that sub-goal was achieved. Order
                matters only for the explanation/metadata, not the score
                -- score is the fraction achieved, unweighted. A checker
                that raises is treated as that sub-goal not achieved
                (same convention as TaskSuccessEvaluator), not propagated.
        """
        if not subgoal_checks:
            raise ValueError("evaluate_with_subgoals requires at least one sub-goal check.")

        results = []
        for i, check in enumerate(subgoal_checks):
            try:
                achieved = bool(check(input.output))
                results.append({"index": i, "achieved": achieved, "error": None})
            except Exception as e:
                results.append({"index": i, "achieved": False, "error": str(e)})

        achieved_count = sum(1 for r in results if r["achieved"])
        score = achieved_count / len(subgoal_checks)
        band = higher_is_better_rate_band(score)

        return EvaluationResult(
            score=score,
            explanation=(
                f"{achieved_count}/{len(subgoal_checks)} sub-goals achieved "
                f"-- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"subgoal_results": results, "achieved_count": achieved_count, "total_subgoals": len(subgoal_checks), "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("SubgoalProgressEvaluator requires a sub-goal checklist. Call evaluate_with_subgoals(input, subgoal_checks) instead.")
