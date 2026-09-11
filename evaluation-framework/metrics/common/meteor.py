"""
METEOR -- Level 2 metric under Accuracy.

Applicability: common. Reference-comparison metric augmenting n-gram
matching with stemming and WordNet synonymy. Needs a labeled target.

Wraps `nltk.translate.meteor_score`. See meteor_setup.py for the
WordNet corpus dependency and its known failure modes. Banded via
contracts.banding.higher_is_better_rate_band -- Vector's Appendix tags
METEOR with Norm="H".
"""

from __future__ import annotations

import nltk

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator
from metrics.common.task_accuracy import _extract_answer_text


def _wordnet_available() -> bool:
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        return False


class METEOREvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "meteor"

    def __init__(self):
        if not _wordnet_available():
            raise RuntimeError(
                "METEOREvaluator requires the NLTK WordNet corpus, which is not "
                "currently available. Run:\n\n"
                "    aixpert-setup-meteor\n\n"
                "once, before using this evaluator."
            )

    def evaluate_with_target(self, input: EvaluationInput, target: str) -> EvaluationResult:
        from nltk.translate.meteor_score import meteor_score

        hypothesis = _extract_answer_text(input.output)
        if not hypothesis.strip():
            return EvaluationResult(score=0.0, explanation="Empty hypothesis text.", is_success=False, metadata={"band_level": 0})

        score = meteor_score([target.split()], hypothesis.split())
        band = higher_is_better_rate_band(score)

        return EvaluationResult(
            score=score,
            explanation=f"METEOR: {score:.3f} -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"meteor_score": score, "band_level": band.level, "band_label": band.label, "hypothesis": hypothesis, "target": target},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("METEOREvaluator requires a labeled target. Call evaluate_with_target(input, target) instead.")
