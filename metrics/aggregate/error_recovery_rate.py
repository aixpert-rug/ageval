"""
Error recovery rate -- Level 2 metric under Robustness .

Applicability: aggregate. Needs labeled recovery outcomes across a set
of fault-injection trials. Per the Appendix, its data need is "env w/
injected faults" -- WP4 doesn't currently expose a fault-injection
harness (same gap as the live-intervention metrics), but the METRIC
ITSELF doesn't need to wait for that: it's a straightforward rate
computation given whatever recovery outcomes are supplied, however
they were generated. Buildable now; the fault-injection data pipeline
is a separate, later question.

Per the Appendix: Norm="H" (Higher-is-better rate).
"""

from __future__ import annotations

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class ErrorRecoveryRateEvaluator(WP3Evaluator):
    dimensions = ("Robustness",)
    applicability = "aggregate"
    metric_id = "error_recovery_rate"

    def evaluate_labeled_set(self, recovery_outcomes: list[bool]) -> EvaluationResult:
        """
        Args:
            recovery_outcomes: one entry per fault-injection trial --
                True if the agent successfully recovered from the
                injected fault, False if it didn't.
        """
        if not recovery_outcomes:
            raise ValueError("evaluate_labeled_set requires at least one recovery outcome.")

        rate = sum(recovery_outcomes) / len(recovery_outcomes)
        band = higher_is_better_rate_band(rate)
        score = rate

        return EvaluationResult(
            score=score,
            explanation=(
                f"Recovered from {sum(recovery_outcomes)}/{len(recovery_outcomes)} injected faults "
                f"({rate:.3f}) -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"recovery_rate": rate, "n_trials": len(recovery_outcomes), "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("ErrorRecoveryRateEvaluator operates on a labeled set of recovery outcomes, not a single EvaluationInput. Call evaluate_labeled_set(recovery_outcomes) instead.")
