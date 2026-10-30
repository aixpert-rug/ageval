"""
Pass^k (agent) -- Level 2 metric under Accuracy (reliability of binary success).

Applicability: aggregate. Needs, per problem, n = total independent runs and
c = number of those runs that succeeded, plus a chosen k. Same (n, c) input as
pass_at_k_agent, opposite question:

  - pass@k: probability that AT LEAST ONE of k runs succeeds (capability with
    retries; Chen et al. 2021).
  - pass^k: probability that ALL k runs succeed (reliability; Yao et al. 2024,
    tau-bench). An agent that solves a task only some of the time scores high on
    pass@k and low on pass^k.

Added with the metric updates (Oct 2026): for pass/fail outcomes, run-to-
run variance is not the best measure (an agent that always fails has variance 0
and looks fully consistent), so reliability is measured with pass^k.

Formula (unbiased estimator from n runs, k <= n), per problem:

    pass^k = C(c, k) / C(n, k)        (0 when c < k)

then averaged over problems and banded with the H family. Reference values:
5 successes in 5 runs gives pass^5 = 1; 4 in 5 gives pass^5 = 0; n=5, c=4, k=1
gives 0.8 (= c/n, as it must for k=1).

Choice of k: tau-bench defines pass^k for any k, headlines pass^1 (which is just
the mean success rate, so it says nothing about reliability) and plots the curve
over k to show how fast reliability decays (it highlights k = 8). There is no
single mandated k. Default here: k = 5, matching the 5 runs per problem
recommended for the multi-trial metrics, capped at the smallest n so the
estimator stays defined. Pass k explicitly to override; the k used and where it
came from are recorded in the metadata.
"""

from __future__ import annotations

import math

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator

DEFAULT_K = 5


def _pass_hat_k_single_problem(n: int, c: int, k: int) -> float:
    if n < 1 or not 0 <= c <= n:
        raise ValueError(f"Need n >= 1 and 0 <= c <= n, got n={n}, c={c}.")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n}) -- can't pick k runs out of only n.")
    return math.comb(c, k) / math.comb(n, k)


class PassHatKAgentEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "aggregate"
    metric_id = "pass_hat_k_agent"

    def evaluate_problems(self, problem_results: list[tuple[int, int]], k: int | None = None) -> EvaluationResult:
        """
        Args:
            problem_results: list of (n, c) pairs, one per problem --
                n = total independent runs of that problem,
                c = number of those runs that succeeded.
            k: number of runs that must ALL succeed. Defaults to
                min(5, smallest n); an explicit k must satisfy k <= every n.
        """
        if not problem_results:
            raise ValueError("evaluate_problems requires at least one problem.")
        min_n = min(n for n, _ in problem_results)
        if k is None:
            k = min(DEFAULT_K, min_n)
            k_source = "default" if k == DEFAULT_K else f"default capped to smallest n ({min_n})"
        else:
            k_source = "explicit"
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")

        per_problem_scores = [_pass_hat_k_single_problem(n, c, k) for n, c in problem_results]
        score = sum(per_problem_scores) / len(per_problem_scores)
        band = higher_is_better_rate_band(score)

        return EvaluationResult(
            score=score,
            explanation=(
                f"pass^{k} averaged across {len(problem_results)} problems: {score:.3f} "
                f"-- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={
                "k": k, "k_source": k_source, "n_problems": len(problem_results),
                "per_problem_scores": per_problem_scores,
                "band_level": band.level, "band_label": band.label,
            },
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("PassHatKAgentEvaluator operates on (n,c) results across multiple problems. Call evaluate_problems(problem_results, k) instead.")
