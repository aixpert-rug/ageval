"""
Judge bias -- Level 2 meta-evaluation metric, under
Fairness, Ethical & Legal Aspects.

Applicability: meta_evaluation. Needs paired scores from a "controlled
set" (per the Appendix) -- the SAME underlying content scored twice,
once with some artefact present and once without (e.g. same answer
presented in position A vs. B for position bias, same content padded
to different lengths for verbosity bias). Measures how much the score
SHIFTS due to the artefact alone, holding actual quality constant.

Formula: mean absolute score shift across paired (with-artefact,
without-artefact) observations. Banded via the "Lower-is-better
continuous" family, budget ceiling = 0.15."""

from __future__ import annotations

from contracts.banding import lower_is_better_continuous_band
from contracts.evaluator import EvaluationResult, WP3Evaluator

_JUDGE_BIAS_BUDGET_CEILING = 0.15  


class JudgeBiasEvaluator(WP3Evaluator):
    dimensions = ("Fairness, Ethical & Legal Aspects",)
    applicability = "meta_evaluation"
    metric_id = "judge_bias"

    def evaluate_controlled_pairs(self, paired_scores: list[tuple[float, float]]) -> EvaluationResult:
        """
        Args:
            paired_scores: list of (score_with_artefact,
                score_without_artefact) tuples on otherwise-identical
                content -- e.g. the same answer judged in two different
                presentation positions.
        """
        if not paired_scores:
            raise ValueError("evaluate_controlled_pairs requires at least one paired observation.")

        shifts = [abs(with_a - without_a) for with_a, without_a in paired_scores]
        mean_shift = sum(shifts) / len(shifts)

        band = lower_is_better_continuous_band(mean_shift, budget_ceiling=_JUDGE_BIAS_BUDGET_CEILING)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=(
                f"Mean absolute score shift from artefact: {mean_shift:.4f} across "
                f"{len(paired_scores)} controlled pairs -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"mean_shift": mean_shift, "n_pairs": len(paired_scores), "budget_ceiling": _JUDGE_BIAS_BUDGET_CEILING, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("JudgeBiasEvaluator operates on controlled paired scores, not a single EvaluationInput. Call evaluate_controlled_pairs(...) instead.")
