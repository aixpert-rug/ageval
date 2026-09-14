"""
Score monotonicity — Level 2 metric under Robustness.

Applicability: reflective only (Reflexion, Self-Refine) -- same data
requirement as consistency.py and convergence_rate.py: a trial loop with a
per-trial evaluator score. Does not apply to ReAct.

What it measures: whether self-correction is actually correcting. A trial
loop that "improves" 0.9 -> 0.3 -> 0.7 is doing something very different
from one that goes 0.2 -> 0.6 -> 0.95, even if both eventually succeed --
the first is thrashing, the second is genuinely converging. Consistency
(stdev-based) doesn't distinguish these either: a wildly oscillating
sequence and a steadily improving one can have comparable spread. This
metric checks direction, not just spread.
"""

from __future__ import annotations

from contracts.evaluator import EvaluationResult, WP3Evaluator


class ScoreMonotonicityEvaluator(WP3Evaluator):
    """Fraction of trial-to-trial transitions that are non-decreasing."""

    dimensions = ("Robustness",)
    applicability = "reflective"
    metric_id = "score_monotonicity"

    def evaluate_trials(self, trial_scores: list[float]) -> EvaluationResult:
        if len(trial_scores) < 2:
            return EvaluationResult(
                score=1.0,
                explanation="Fewer than 2 trials; monotonicity undefined, treated as vacuously monotonic.",
                is_success=True,
                metadata={"num_trials": len(trial_scores)},
            )

        deltas = [trial_scores[i + 1] - trial_scores[i] for i in range(len(trial_scores) - 1)]
        non_decreasing = [d >= 0 for d in deltas]
        score = sum(non_decreasing) / len(non_decreasing)
        fully_monotonic = all(non_decreasing)

        return EvaluationResult(
            score=score,
            explanation=(
                f"{sum(non_decreasing)}/{len(non_decreasing)} trial-to-trial transitions "
                f"were non-decreasing (deltas: {[round(d, 3) for d in deltas]})."
            ),
            is_success=fully_monotonic,
            metadata={"num_trials": len(trial_scores), "deltas": deltas, "fully_monotonic": fully_monotonic},
        )

    def evaluate(self, input) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "ScoreMonotonicityEvaluator operates on a trial history, not a single "
            "EvaluationInput. Call evaluate_trials(trial_scores) instead."
        )
