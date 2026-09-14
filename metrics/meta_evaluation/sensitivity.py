"""
Sensitivity -- Level 2 meta-evaluation metric, under
Auditability / Verifiability.

Applicability: meta_evaluation. Needs pairs of systems with a KNOWN,
gold-labeled quality gap (per the Appendix: "systems w/ known gaps") --
checks whether the metric being validated actually detects that known
difference in the right direction. A metric that scores two
genuinely-different systems as statistically indistinguishable fails
this even if its scores individually look reasonable.

Formula: fraction of known-gap pairs where the metric's score ordering
matches the known ordering (the better system per the gold label also
scored higher by the metric). Simple accuracy over known-easy cases,
by design -- this isn't testing subtle discrimination, it's testing
whether the metric gets the CLEAR cases right at all.
"""

from __future__ import annotations

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class SensitivityEvaluator(WP3Evaluator):
    dimensions = ("Auditability / Verifiability",)
    applicability = "meta_evaluation"
    metric_id = "sensitivity"

    def evaluate_known_gaps(self, gap_cases: list[tuple[float, float, bool]]) -> EvaluationResult:
        """
        Args:
            gap_cases: list of (score_a, score_b, a_is_known_better)
                tuples -- score_a/score_b are what the metric under
                test assigned; a_is_known_better is the GOLD label for
                whether system A is actually better than system B.
        """
        if not gap_cases:
            raise ValueError("evaluate_known_gaps requires at least one known-gap case.")

        correct = 0
        for score_a, score_b, a_is_known_better in gap_cases:
            detected_a_better = score_a > score_b
            if detected_a_better == a_is_known_better:
                correct += 1

        rate = correct / len(gap_cases)
        band = higher_is_better_rate_band(rate)
        score = rate

        return EvaluationResult(
            score=score,
            explanation=(
                f"Correctly detected {correct}/{len(gap_cases)} known quality gaps "
                f"({rate:.3f}) -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"sensitivity_rate": rate, "n_cases": len(gap_cases), "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("SensitivityEvaluator operates on known-gap system pairs, not a single EvaluationInput. Call evaluate_known_gaps(...) instead.")
