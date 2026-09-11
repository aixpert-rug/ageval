"""
Pass@k (agent) -- Level 2 metric under Accuracy (Vector's "Pa").

Applicability: aggregate. Needs, per problem, n = total sampled attempts
and c = number of those attempts that succeeded, plus a chosen k (the
"tries budget" being evaluated). Aggregates the unbiased pass@k
estimator across multiple problems.

Formula (Chen et al. 2021, "Evaluating Large Language Models Trained on
Code" -- the same estimator Vector's own document cites for its
LLM-level Pass@k, Eq. 10 in their Appendix): the numerically stable
product form, avoiding large-factorial overflow:

    pass@k = 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i),  if n-c >= k
    pass@k = 1.0,                                  if n-c < k
             (fewer than k failures exist, so k random draws can't miss
             a success -- the edge case, not a special-case hack)

Verified against known reference values before use: n=5,c=2,k=1 -> 0.4
(equals c/n exactly, as it must for k=1); n=5,c=5,k=1 -> 1.0 (all
correct); n=5,c=0,k=1 -> 0.0 (none correct).
"""

from __future__ import annotations

import math

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


def _pass_at_k_single_problem(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n}) -- can't sample k attempts from only n total.")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


class PassAtKAgentEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "aggregate"
    metric_id = "pass_at_k_agent"

    def evaluate_problems(self, problem_results: list[tuple[int, int]], k: int) -> EvaluationResult:
        """
        Args:
            problem_results: list of (n, c) pairs, one per problem --
                n = total independent attempts sampled for that problem,
                c = number of those attempts that succeeded.
            k: the "tries" budget to evaluate pass-rate-within-k-attempts for.
        """
        if not problem_results:
            raise ValueError("evaluate_problems requires at least one problem.")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")

        per_problem_scores = [_pass_at_k_single_problem(n, c, k) for n, c in problem_results]
        score = sum(per_problem_scores) / len(per_problem_scores)
        band = higher_is_better_rate_band(score)

        return EvaluationResult(
            score=score,
            explanation=(
                f"pass@{k} averaged across {len(problem_results)} problems: {score:.3f} "
                f"-- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"k": k, "n_problems": len(problem_results), "per_problem_scores": per_problem_scores, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("PassAtKAgentEvaluator operates on (n,c) results across multiple problems. Call evaluate_problems(problem_results, k) instead.")
