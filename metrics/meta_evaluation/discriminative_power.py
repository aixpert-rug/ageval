"""
Discriminative power -- Level 2 meta-evaluation metric,
under Auditability / Verifiability.

Applicability: meta_evaluation. Needs scores grouped by which system
produced them ("multi-system scores" per the Appendix). Implements
eta-squared (the standard one-way-ANOVA effect-size statistic): the
fraction of total score variance explained by which system produced the
score. High eta-squared means the metric spreads different systems'
scores apart meaningfully (good discrimination); near-zero means the
metric can't tell systems apart even when scoring genuinely different
systems.

Formula: eta_squared = SS_between / SS_total. Verified against two
known extremes before use: clearly-separated groups give eta-squared
close to 1.0; near-identical groups give exactly 0.0.
"""

from __future__ import annotations

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class DiscriminativePowerEvaluator(WP3Evaluator):
    dimensions = ("Auditability / Verifiability",)
    applicability = "meta_evaluation"
    metric_id = "discriminative_power"

    def evaluate_groups(self, scores_by_system: dict[str, list[float]]) -> EvaluationResult:
        """
        Args:
            scores_by_system: dict mapping a system identifier to the
                list of scores THIS metric assigned to that system
                across multiple items/runs. Need at least 2 systems.
        """
        if len(scores_by_system) < 2:
            raise ValueError("evaluate_groups requires at least 2 systems to compare.")
        if any(len(scores) == 0 for scores in scores_by_system.values()):
            raise ValueError("Every system must have at least one score.")

        all_scores = [s for scores in scores_by_system.values() for s in scores]
        n_total = len(all_scores)
        grand_mean = sum(all_scores) / n_total

        ss_total = sum((s - grand_mean) ** 2 for s in all_scores)
        ss_between = sum(
            len(scores) * ((sum(scores) / len(scores)) - grand_mean) ** 2
            for scores in scores_by_system.values()
        )

        eta_squared = (ss_between / ss_total) if ss_total > 0 else 0.0
        band = higher_is_better_rate_band(eta_squared)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=(
                f"eta-squared={eta_squared:.3f} across {len(scores_by_system)} systems "
                f"({n_total} total scores) -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"eta_squared": eta_squared, "n_systems": len(scores_by_system), "n_total_scores": n_total, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("DiscriminativePowerEvaluator operates on scores grouped by system, not a single EvaluationInput. Call evaluate_groups(...) instead.")
