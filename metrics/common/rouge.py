"""
ROUGE-L -- Level 2 metric under Accuracy.

Applicability: common. Reference-comparison metric based on longest
common subsequence, needs a labeled target.

Wraps Google's `rouge-score` package. Banded via
contracts.banding.higher_is_better_rate_band.
"""

from __future__ import annotations

from rouge_score import rouge_scorer

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator
from metrics.common.task_accuracy import _extract_answer_text

_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


class ROUGEEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "rouge_l"

    def evaluate_with_target(self, input: EvaluationInput, target: str) -> EvaluationResult:
        hypothesis = _extract_answer_text(input.output)
        if not hypothesis.strip():
            return EvaluationResult(score=0.0, explanation="Empty hypothesis text.", is_success=False, metadata={"precision": 0.0, "recall": 0.0, "fmeasure": 0.0, "band_level": 0})

        scores = _SCORER.score(target, hypothesis)["rougeL"]
        band = higher_is_better_rate_band(scores.fmeasure)

        return EvaluationResult(
            score=scores.fmeasure,
            explanation=(
                f"ROUGE-L F1: {scores.fmeasure:.3f} (precision={scores.precision:.3f}, "
                f"recall={scores.recall:.3f}) -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"precision": scores.precision, "recall": scores.recall, "fmeasure": scores.fmeasure, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("ROUGEEvaluator requires a labeled target. Call evaluate_with_target(input, target) instead.")
