"""
Judge-human agreement AND Metric validity -- two Level 2 meta-evaluation
metrics. Grouped in one file because
they are, per the Appendix, the IDENTICAL computation (Pearson
correlation) applied to two different data sources -- not a coincidence
worth hiding across two files that would otherwise duplicate the same
math:

  - Judge-human agreement: correlation between an LLM judge's scores and
    human gold ratings on the same items.
  - Metric validity: correlation between any metric's scores and an
    independent criterion/ground-truth quality measure on the same items.

Applicability: meta_evaluation

Formula: Pearson correlation coefficient, via Python's stdlib
`statistics.correlation` (available 3.10+; this repo requires 3.11+, so
no new dependency). Verified against a known perfectly-linear case
(r=1.0 exactly) before use.

Per Table 5, both use the "Higher-is-better rate" family. Correlation
coefficients range [-1, 1], so (metric updates, Oct 2026) the value
banded is max(0, r): no (r+1)/2 rescaling, which moved r = 0 (no agreement) to
0.5 = band 2 = pass. r <= 0.20 is band 0, and any negative r is band 0 too.
The raw r is kept in metadata and negative values are flagged for review (they
often signal an inverted scale rather than poor agreement). If either input is
constant, Pearson r is undefined (statistics.correlation raises); that case
returns band 0 with an explanatory message instead of raising.
"""

from __future__ import annotations

import statistics

from contracts.banding import agreement_coefficient_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


def _correlation_result(x: list[float], y: list[float], label: str) -> EvaluationResult:
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}.")
    if len(x) < 2:
        raise ValueError("Correlation requires at least 2 paired observations.")

    try:
        r = statistics.correlation(x, y)
    except statistics.StatisticsError:
        # At least one input is constant, so Pearson r is undefined.
        return EvaluationResult(
            score=0.0,
            explanation=f"{label}: Pearson r undefined because at least one input is constant -- band 0 (Unacceptable). REVIEW: check the scores.",
            is_success=False,
            metadata={
                "pearson_r": None, "n_pairs": len(x), "review_flag": True, "degenerate": True,
                "degenerate_reason": "constant input",
                "band_level": 0, "band_label": "Unacceptable",
            },
        )

    band = agreement_coefficient_band(r)
    score = band.level / 4.0
    review_flag = r < 0

    return EvaluationResult(
        score=score,
        explanation=(
            f"{label}: Pearson r={r:.3f} -- band {band.level} ({band.label})."
            + (" REVIEW: negative r (worse than chance); check for an inverted scale." if review_flag else "")
        ),
        is_success=band.level >= 2,
        metadata={"pearson_r": r, "n_pairs": len(x), "review_flag": review_flag, "degenerate": False, "band_level": band.level, "band_label": band.label},
    )


class JudgeHumanAgreementEvaluator(WP3Evaluator):
    dimensions = ("Transparency & Explainability",)
    applicability = "meta_evaluation"
    metric_id = "judge_human_agreement"

    def evaluate_agreement(self, judge_scores: list[float], human_scores: list[float]) -> EvaluationResult:
        """
        Args:
            judge_scores, human_scores: paired scores on the SAME items,
                same order -- judge_scores[i] and human_scores[i] must
                refer to the same evaluated item.
        """
        return _correlation_result(judge_scores, human_scores, "Judge-human agreement")

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("JudgeHumanAgreementEvaluator operates on paired judge/human scores, not a single EvaluationInput. Call evaluate_agreement(...) instead.")


class MetricValidityEvaluator(WP3Evaluator):
    dimensions = ("Auditability / Verifiability",)
    applicability = "meta_evaluation"
    metric_id = "metric_validity"

    def evaluate_validity(self, metric_scores: list[float], criterion_scores: list[float]) -> EvaluationResult:
        """
        Args:
            metric_scores: scores produced by the metric being validated.
            criterion_scores: an independent ground-truth quality
                measure on the SAME items, same order.
        """
        return _correlation_result(metric_scores, criterion_scores, "Metric validity")

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("MetricValidityEvaluator operates on paired metric/criterion scores, not a single EvaluationInput. Call evaluate_validity(...) instead.")
