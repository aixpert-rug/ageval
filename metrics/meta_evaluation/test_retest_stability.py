"""
Test-retest stability -- Level 2 meta-evaluation metric
under Robustness.

Applicability: meta_evaluation. Needs a metric's score on each item, rerun
multiple times -- variance across reruns. Uses the SAME aggregation as
metrics/aggregate/cross_run_consistency.py (metric updates, Oct 2026):
per item, the sample variance of its rerun scores; the mean of those variances
over all items; one C-family band with ceiling 0.10 (see
metrics/aggregate/_run_variance.py). At least 2 reruns per item (5
recommended); scores must be normalised to [0, 1] first.

Kept as a separate class because the two measure conceptually
different things despite the shared math: cross_run_consistency asks
"is the AGENT's behaviour stable across repeated runs", this asks "is
the EVALUATION PROCESS's score stable when rerun on the same item" --
same statistical question, different subject under test. Same
"consistency of pattern over DRY" reasoning as help_seeking.py vs.
abstention_rate.py.
"""

from __future__ import annotations

from typing import Sequence

from contracts.evaluator import EvaluationResult, WP3Evaluator
from metrics.aggregate._run_variance import describe, summarise_run_variances


class TestRetestStabilityEvaluator(WP3Evaluator):
    dimensions = ("Robustness",)
    applicability = "meta_evaluation"
    metric_id = "test_retest_stability"

    def evaluate_reruns(self, scores: Sequence[Sequence[float]], item_ids: Sequence | None = None) -> EvaluationResult:
        """
        Args:
            scores: one list per evaluated item, holding the same evaluation
                (same item, same metric) scored multiple times independently
                -- e.g. rerunning an LLM-judge on the identical input to
                check its own consistency, not the agent's. At least 2
                reruns per item, scores in [0, 1]. A flat list of numbers is
                treated as the reruns of a single item.
            item_ids: optional labels for listing the most unstable items.
        """
        s = summarise_run_variances(scores, item_ids)
        band = s["band"]
        return EvaluationResult(
            score=band.level / 4.0,
            explanation=describe(s, "items"),
            is_success=band.level >= 2,
            metadata={
                "n_items": s["n_problems"],
                "reruns_per_item": s["runs_per_problem"],
                "mean_variance": s["mean_variance"], "per_item_variances": s["per_problem_variances"],
                "n_unstable": s["n_unstable"], "unstable_share": s["unstable_share"],
                "most_unstable": s["most_unstable"], "variance_ci_95": s["variance_ci_95"],
                "below_recommended_runs": s["below_recommended_runs"],
                "budget_ceiling": s["budget_ceiling"], "band_level": band.level, "band_label": band.label,
            },
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("TestRetestStabilityEvaluator operates on scores from multiple reruns per item, not a single EvaluationInput. Call evaluate_reruns(...) instead.")
