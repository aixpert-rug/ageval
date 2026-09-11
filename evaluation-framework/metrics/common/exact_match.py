"""
Exact Match -- Level 2 metric under Accuracy.

Applicability: common. The strictest reference-comparison metric: does
the predicted answer match the target exactly, after light, standard
normalisation (lowercasing, whitespace collapsing, stripping punctuation
Vector's own document lists this alongside F1 as the standard pairing --
EM is precision-at-the-extreme (only a perfect match counts), F1 (already
implemented in task_accuracy.py) is the more forgiving partial-credit
version of the same comparison. Reporting both side by side is standard
practice (e.g. SQuAD-style QA evaluation) since the gap between them is
itself informative: EM << F1 suggests the model is "close" but not
precise, EM \u2248 F1 suggests it's mostly all-or-nothing.
"""

from __future__ import annotations

import re
import string

from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator
from metrics.common.task_accuracy import _extract_answer_text


def _normalize(text: str) -> str:
    """Standard QA-style normalisation: lowercase, strip punctuation,
    collapse whitespace, drop leading/trailing articles. Intentionally
    the same normalisation family used by SQuAD-style EM scoring, so
    scores are comparable to how EM is usually reported elsewhere."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ExactMatchEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "exact_match"

    def evaluate_with_target(self, input: EvaluationInput, target: str) -> EvaluationResult:
        predicted = _normalize(_extract_answer_text(input.output))
        expected = _normalize(target)
        is_match = predicted == expected

        return EvaluationResult(
            score=1.0 if is_match else 0.0,
            explanation=(
                f"Normalised prediction {predicted!r} "
                f"{'==' if is_match else '!='} normalised target {expected!r}."
            ),
            is_success=is_match,
            metadata={"normalized_prediction": predicted, "normalized_target": expected},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("ExactMatchEvaluator requires a labeled target. Call evaluate_with_target(input, target) instead.")
