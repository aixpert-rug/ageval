"""
BLEU -- Level 2 metric under Accuracy.

Applicability: common. Reference-comparison, n-gram precision metric --
needs a labeled target.

Wraps `sacrebleu` rather than reimplementing (see module history for why).
Banded via contracts.banding.higher_is_better_rate_band -- Vector's
Appendix tags BLEU with Norm="H".

Note: BLEU is a corpus-level metric by design and noisy at the
single-sentence level; sentence_bleu is used here to fit this repo's
per-EvaluationInput convention, but treat single-sentence BLEU as
indicative, not precise.
"""

from __future__ import annotations

import sacrebleu

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator
from metrics.common.task_accuracy import _extract_answer_text


class BLEUEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "bleu"

    def evaluate_with_target(self, input: EvaluationInput, target: str) -> EvaluationResult:
        hypothesis = _extract_answer_text(input.output)
        if not hypothesis.strip():
            return EvaluationResult(score=0.0, explanation="Empty hypothesis text.", is_success=False, metadata={"bleu_0_100": 0.0, "band_level": 0})

        result = sacrebleu.sentence_bleu(hypothesis, [target])
        bleu_0_100 = result.score
        normalized = bleu_0_100 / 100.0
        band = higher_is_better_rate_band(normalized)

        return EvaluationResult(
            score=normalized,
            explanation=f"BLEU: {bleu_0_100:.2f}/100 -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"bleu_0_100": bleu_0_100, "band_level": band.level, "band_label": band.label, "hypothesis": hypothesis, "target": target},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("BLEUEvaluator requires a labeled target. Call evaluate_with_target(input, target) instead.")
