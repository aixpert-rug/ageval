"""
Consistency -- Level 2 metric under Robustness.

Applicability: needs variance across MULTIPLE INDEPENDENT COMPLETE RUNS of the same task -- applies to 
ANY pattern (ReAct, Reflexion, Self-Refine, governed or not), not scoped
to a specific pattern architecture the way the other categories are.

Named CrossRunConsistencyEvaluator, distinct from
metrics/reflective/consistency.py's ConsistencyEvaluator
(metric_id="trial_consistency"), which measures variance ACROSS TRIALS WITHIN one
Reflexion/Self-Refine self-correction loop. This one measures variance
across REPEATED, INDEPENDENT full runs of the same task -- e.g. running
the exact same ReAct query 5 separate times and checking how much the
final score varies.

"""

from __future__ import annotations

import statistics

from contracts.banding import lower_is_better_continuous_band
from contracts.evaluator import EvaluationResult, WP3Evaluator

_CONSISTENCY_BUDGET_CEILING = 0.10 


class CrossRunConsistencyEvaluator(WP3Evaluator):
    dimensions = ("Robustness",)
    applicability = "aggregate"
    metric_id = "cross_run_consistency"

    def evaluate_runs(self, run_scores: list[float]) -> EvaluationResult:
        """
        Args:
            run_scores: the final task-level score (e.g. from
                TaskAccuracyEvaluator, TaskSuccessEvaluator, or any other
                per-run metric) from each of several INDEPENDENT, complete
                runs of the same task with the same agent configuration.
                Not trial-loop scores from a single Reflexion run -- see
                module docstring.
        """
        if len(run_scores) < 2:
            return EvaluationResult(
                score=1.0,
                explanation="Fewer than 2 runs; variance undefined, treated as vacuously consistent.",
                is_success=True,
                metadata={"num_runs": len(run_scores), "band_level": 4, "band_label": "Excellent"},
            )

        variance = statistics.variance(run_scores)
        band = lower_is_better_continuous_band(variance, budget_ceiling=_CONSISTENCY_BUDGET_CEILING)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=(
                f"Score variance across {len(run_scores)} independent runs: {variance:.4f} "
                f"(scores: {run_scores}) -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"num_runs": len(run_scores), "variance": variance, "budget_ceiling": _CONSISTENCY_BUDGET_CEILING, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("CrossRunConsistencyEvaluator operates on scores from multiple independent runs, not a single EvaluationInput. Call evaluate_runs(run_scores) instead.")
