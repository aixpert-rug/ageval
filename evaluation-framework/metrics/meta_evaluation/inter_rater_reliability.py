"""
Inter-rater reliability -- Level 2 meta-evaluation metric , under Transparency & Explainability.

Applicability: meta_evaluation. Needs paired labels from exactly two
raters on the same items. Implements Cohen's Kappa -- the standard,
well-defined two-rater agreement statistic.  Krippendorff's
alpha (the >2-rater, more general generalisation) is NOT implemented
here -- flagged as a known, real gap, not silently substituted. Use this
for the common two-rater case; a separate evaluator would be needed for
>2 raters or non-nominal data.

Formula: kappa = (p_o - p_e) / (1 - p_e), where p_o is observed
agreement proportion and p_e is the chance-agreement proportion implied
by each rater's own marginal label frequencies. Verified against a
worked example before use: 10 items, 8/10 agreement, marginals giving
p_e=0.5 -> kappa=0.6 exactly (a standard textbook "substantial
agreement" result).
"""

from __future__ import annotations

from collections import Counter

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class InterRaterReliabilityEvaluator(WP3Evaluator):
    dimensions = ("Transparency & Explainability",)
    applicability = "meta_evaluation"
    metric_id = "inter_rater_reliability"

    def evaluate_ratings(self, rater_a_labels: list, rater_b_labels: list) -> EvaluationResult:
        """
        Args:
            rater_a_labels, rater_b_labels: paired categorical labels
                from two raters on the SAME items, same order. Labels
                can be any hashable type (strings, ints, bools).
        """
        if len(rater_a_labels) != len(rater_b_labels):
            raise ValueError(f"Both raters must have the same number of labels, got {len(rater_a_labels)} and {len(rater_b_labels)}.")
        n = len(rater_a_labels)
        if n < 1:
            raise ValueError("evaluate_ratings requires at least one rated item.")

        po = sum(1 for a, b in zip(rater_a_labels, rater_b_labels) if a == b) / n

        count_a = Counter(rater_a_labels)
        count_b = Counter(rater_b_labels)
        all_labels = set(count_a) | set(count_b)
        pe = sum((count_a.get(lbl, 0) / n) * (count_b.get(lbl, 0) / n) for lbl in all_labels)

        if pe == 1.0:
            kappa = 1.0 if po == 1.0 else 0.0
        else:
            kappa = (po - pe) / (1 - pe)

        rescaled = max(0.0, min(1.0, (kappa + 1) / 2))
        band = higher_is_better_rate_band(rescaled)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=f"Cohen's kappa={kappa:.3f} (po={po:.3f}, pe={pe:.3f}) -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"kappa": kappa, "observed_agreement": po, "expected_agreement": pe, "n_items": n, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("InterRaterReliabilityEvaluator operates on two raters' paired labels, not a single EvaluationInput. Call evaluate_ratings(...) instead.")
