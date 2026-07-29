"""
Convergence rate — Level 2 metric under Robustness.

Applicability: reflective only (Reflexion, Self-Refine) -- needs a trial
loop with a per-trial success signal. Does not apply to ReAct.

What it measures: how quickly a trial-based pattern reaches success, as a
fraction of its trial budget. Converging on trial 1 scores higher than
converging on the last available trial; never converging scores 0.

Requires `max_iterations` explicitly rather than inferring it from the
length of the trial list -- the trial list you have might stop early
(success reached) or reflect only what was actually run, not the pattern's
configured budget. Passing the wrong max_iterations silently produces a
wrong score, so this is deliberately not defaulted quietly; see the
ValueError raised if it's obviously inconsistent with the data.
"""

from __future__ import annotations

from contracts.evaluator import EvaluationResult, WP3Evaluator


class ConvergenceRateEvaluator(WP3Evaluator):
    """Score reflecting how many trials were needed to reach success."""

    dimensions = ("Robustness",)
    applicability = "reflective"
    metric_id = "convergence_rate"

    def evaluate_trials(self, is_success_per_trial: list[bool], max_iterations: int) -> EvaluationResult:
        """
        Args:
            is_success_per_trial: per-trial success flags, in order, e.g.
                extracted from ReflexionAgent's `reflections[i]['eval_result']
                ['is_success']` or the equivalent for Self-Refine.
            max_iterations: the pattern's configured trial budget (its
                `max_iterations` constructor argument) -- NOT necessarily
                len(is_success_per_trial), since a run that converges early
                stops logging further trials.
        """
        if len(is_success_per_trial) > max_iterations:
            raise ValueError(
                f"Got {len(is_success_per_trial)} trials but max_iterations={max_iterations}; "
                "these are inconsistent -- check you passed the pattern's actual configured budget."
            )

        converged_at = next((i + 1 for i, s in enumerate(is_success_per_trial) if s), None)

        if converged_at is None:
            return EvaluationResult(
                score=0.0,
                explanation=(
                    f"Did not converge within {len(is_success_per_trial)} trial(s) "
                    f"(budget: {max_iterations})."
                ),
                is_success=False,
                metadata={"converged_at": None, "max_iterations": max_iterations},
            )

        # Trial 1 success -> 1.0; success on the last possible trial -> 1/max_iterations.
        score = (max_iterations - converged_at + 1) / max_iterations

        return EvaluationResult(
            score=score,
            explanation=f"Converged at trial {converged_at} of {max_iterations} available.",
            is_success=True,
            metadata={"converged_at": converged_at, "max_iterations": max_iterations},
        )

    def evaluate(self, input) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "ConvergenceRateEvaluator operates on a trial history, not a single "
            "EvaluationInput. Call evaluate_trials(is_success_per_trial, max_iterations) instead."
        )
