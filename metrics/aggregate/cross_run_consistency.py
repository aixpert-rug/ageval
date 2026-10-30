"""
Cross-run consistency -- Level 2 metric under Robustness.

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

Method (metric updates, Oct 2026; see metrics/aggregate/_run_variance.py):
run each problem (one task instance) n times with the same agent
configuration (5 is enough, minimum 2), take the sample variance of each
problem's run scores, average those variances over all problems, and band the
mean ONCE with the C family, ceiling 0.10. Metadata keeps every problem's
variance, the share of problems that were not fully stable, the most unstable
problems and a bootstrap CI over problems.

Scores must be normalised to [0, 1] first. For binary (pass/fail) outcomes
pass_hat_k_agent is the more informative companion (an agent that always
fails has variance 0 and looks fully consistent here).
"""

from __future__ import annotations

from typing import Sequence

from contracts.evaluator import EvaluationResult, WP3Evaluator
from metrics.aggregate._run_variance import describe, summarise_run_variances


class CrossRunConsistencyEvaluator(WP3Evaluator):
    dimensions = ("Robustness",)
    applicability = "aggregate"
    metric_id = "cross_run_consistency"

    def evaluate_runs(self, run_scores: Sequence[Sequence[float]], problem_ids: Sequence | None = None) -> EvaluationResult:
        """
        Args:
            run_scores: one list per problem (task instance), holding the
                final task-level score (from TaskAccuracyEvaluator,
                TaskSuccessEvaluator, F1, a normalised judge score, ...) of
                each of that problem's INDEPENDENT, complete runs with the
                same agent configuration. At least 2 runs per problem (5
                recommended), scores in [0, 1]. A flat list of numbers is
                treated as the runs of a single problem. Not trial-loop
                scores from a single Reflexion run -- see module docstring.
            problem_ids: optional labels used when listing the most unstable
                problems (defaults to the problem index).
        """
        s = summarise_run_variances(run_scores, problem_ids)
        band = s["band"]
        return EvaluationResult(
            score=band.level / 4.0,
            explanation=describe(s, "problems"),
            is_success=band.level >= 2,
            metadata={
                "n_problems": s["n_problems"], "runs_per_problem": s["runs_per_problem"],
                "mean_variance": s["mean_variance"], "per_problem_variances": s["per_problem_variances"],
                "n_unstable": s["n_unstable"], "unstable_share": s["unstable_share"],
                "most_unstable": s["most_unstable"], "variance_ci_95": s["variance_ci_95"],
                "below_recommended_runs": s["below_recommended_runs"],
                "budget_ceiling": s["budget_ceiling"], "band_level": band.level, "band_label": band.label,
            },
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("CrossRunConsistencyEvaluator operates on scores from multiple independent runs per problem, not a single EvaluationInput. Call evaluate_runs(run_scores) instead.")
