"""
Test-retest stability -- Level 2 meta-evaluation metric
under Robustness.

Applicability: meta_evaluation. Needs a metric's score on the SAME
item, rerun multiple times -- variance across reruns. Mathematically
IDENTICAL to metrics/aggregate/cross_run_consistency.py (same C-family
formula, same budget=0.10 ceiling). Kept as a separate class because the two measure conceptually
different things despite the shared math: cross_run_consistency asks
"is the AGENT's behaviour stable across repeated runs", this asks "is
the EVALUATION PROCESS's score stable when rerun on the same item" --
same statistical question, different subject under test. Same
"consistency of pattern over DRY" reasoning as help_seeking.py vs.
abstention_rate.py.
"""

from __future__ import annotations

import statistics

from contracts.banding import lower_is_better_continuous_band
from contracts.evaluator import EvaluationResult, WP3Evaluator

_BUDGET_CEILING = 0.10  


class TestRetestStabilityEvaluator(WP3Evaluator):
    dimensions = ("Robustness",)
    applicability = "meta_evaluation"
    metric_id = "test_retest_stability"

    def evaluate_reruns(self, scores: list[float]) -> EvaluationResult:
        """
        Args:
            scores: the same evaluation (same item, same metric) scored
                multiple times independently -- e.g. rerunning an
                LLM-judge on the identical input to check its own
                consistency, not the agent's.
        """
        if len(scores) < 2:
            return EvaluationResult(
                score=1.0,
                explanation="Fewer than 2 reruns; variance undefined, treated as vacuously stable.",
                is_success=True,
                metadata={"n_reruns": len(scores), "band_level": 4, "band_label": "Excellent"},
            )

        variance = statistics.variance(scores)
        band = lower_is_better_continuous_band(variance, budget_ceiling=_BUDGET_CEILING)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=f"Score variance across {len(scores)} reruns: {variance:.4f} -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"n_reruns": len(scores), "variance": variance, "budget_ceiling": _BUDGET_CEILING, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("TestRetestStabilityEvaluator operates on scores from multiple reruns, not a single EvaluationInput. Call evaluate_reruns(...) instead.")
