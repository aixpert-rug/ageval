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

Per Table 5, both use the "Higher-is-better rate" family -- but
correlation coefficients range [-1, 1], not [0, 1], so this evaluator
rescales r to [0,1] via (r+1)/2 before banding, since
higher_is_better_rate_band expects a [0,1] input. This rescaling is
noted explicitly in metadata (both the raw r and the rescaled value are
reported) so nothing is silently lost.
"""

from __future__ import annotations

import statistics

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


def _correlation_result(x: list[float], y: list[float], label: str) -> EvaluationResult:
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}.")
    if len(x) < 2:
        raise ValueError("Correlation requires at least 2 paired observations.")

    r = statistics.correlation(x, y)
    rescaled = (r + 1) / 2
    band = higher_is_better_rate_band(rescaled)
    score = band.level / 4.0

    return EvaluationResult(
        score=score,
        explanation=f"{label}: Pearson r={r:.3f} (rescaled {rescaled:.3f}) -- band {band.level} ({band.label}).",
        is_success=band.level >= 2,
        metadata={"pearson_r": r, "rescaled_0_1": rescaled, "n_pairs": len(x), "band_level": band.level, "band_label": band.label},
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
