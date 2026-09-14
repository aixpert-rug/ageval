"""
Trial-to-trial consistency — Level 2 metric under Robustness.

Applicability: reflective only. Needs a trial loop (Reflexion, Self-Refine).
Does not apply to ReAct, which runs once and has no `reflections`.

What it measures: how much the evaluator's own score varies across trials
for nominally the same task. High variance suggests either the agent's
behavior is unstable, or the evaluator itself is noisy (e.g. an LLM-judge
with high sampling variance) — this metric alone can't distinguish the two;
see `notes` in the result.

Unlike faithfulness.py, this does not operate on a raw EvaluationInput —
it operates on the accumulated trial history a Reflexion/Self-Refine run
produces. Call it once, after a run completes, with that pattern's
`reflections` list (each entry has at least `trial_num` and an associated
`eval_result`/score — see fixtures/mock_reflexion_trajectory.json for the
exact shape this expects).
"""

from __future__ import annotations

import statistics
from typing import Any

from contracts.evaluator import EvaluationResult, WP3Evaluator


class ConsistencyEvaluator(WP3Evaluator):
    """Score variance across a completed trial history."""

    dimensions = ("Robustness",)
    applicability = "reflective"
    metric_id = "trial_consistency"

    def evaluate_trials(self, trial_scores: list[float]) -> EvaluationResult:
        """
        Args:
            trial_scores: the `eval_result.score` from each trial, in order.
                Extract these from a pattern's `reflections`/trial history
                after a run completes — see fixtures for the shape.
        """
        if len(trial_scores) < 2:
            return EvaluationResult(
                score=1.0,
                explanation="Fewer than 2 trials; consistency undefined, treated as vacuously consistent.",
                is_success=True,
                metadata={"num_trials": len(trial_scores)},
            )

        stdev = statistics.stdev(trial_scores)
        # Normalise: assume scores are on a 0-1 scale (matches WP4's
        # TrajectoryEvaluator/OutputEvaluator convention). A stdev of 0
        # -> consistency score 1.0; stdev of 0.5+ -> consistency score ~0.
        consistency_score = max(0.0, 1.0 - 2 * stdev)

        return EvaluationResult(
            score=consistency_score,
            explanation=(
                f"Score stdev across {len(trial_scores)} trials: {stdev:.3f} "
                f"(scores: {trial_scores}). Note: does not distinguish agent "
                "instability from evaluator noise — pair with a fixed-seed "
                "rerun of the same evaluator to isolate the two."
            ),
            is_success=consistency_score >= 0.7,
            metadata={"num_trials": len(trial_scores), "stdev": stdev},
        )

    def evaluate(self, input: Any) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "ConsistencyEvaluator operates on a trial history, not a single "
            "EvaluationInput. Call evaluate_trials(trial_scores) instead."
        )
