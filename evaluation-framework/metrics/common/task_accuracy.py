"""
Task accuracy vs. ground truth — Level 2 metric under Accuracy.

Applicability: common. Works on any pattern's final output, given a labeled
target to compare against.

Implementation follows Inspect AI's `f1()` scorer (inspect_ai.scorer.f1):
F1 = harmonic mean of precision/recall over token overlap between the
predicted answer and the target. Chosen over exact-match because it's
appropriate for short free-text answers where exact wording can vary but
the substance shouldn't (e.g. "14C, 80% rain" vs. "14 degrees, 80% chance
of rain" should score well, not 0).

For structured/numeric outputs where exact or mathematical equivalence is
more appropriate than token overlap, see `math_accuracy.py` instead.
"""

from __future__ import annotations

import re
from typing import Any

from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator

_DEFAULT_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "of", "to", "in", "on", "at",
}


def _tokenize(text: str, stop_words: set[str] = frozenset()) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in stop_words]


def _extract_answer_text(output: Any) -> str:
    """Default answer extraction: join all string-valued output fields.
    Pass a custom `answer_fn` to TaskAccuracyEvaluator for other shapes."""
    if isinstance(output, dict):
        return " ".join(str(v) for v in output.values() if isinstance(v, str))
    return str(output)


class TaskAccuracyEvaluator(WP3Evaluator):
    """F1 token-overlap accuracy against a labeled target.

    score: F1 = 2 * precision * recall / (precision + recall), over token
        sets. 1.0 = identical token content, 0.0 = no overlap at all.
    is_success: score >= threshold (default 0.7).
    """

    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "task_accuracy"

    def __init__(
        self,
        threshold: float = 0.7,
        stop_words: set[str] = _DEFAULT_STOP_WORDS,
        answer_fn=_extract_answer_text,
    ):
        self.threshold = threshold
        self.stop_words = stop_words
        self.answer_fn = answer_fn

    def evaluate_with_target(self, input: EvaluationInput, target: str) -> EvaluationResult:
        predicted_tokens = set(_tokenize(self.answer_fn(input.output), self.stop_words))
        target_tokens = set(_tokenize(target, self.stop_words))

        if not target_tokens:
            raise ValueError("Target produced no tokens after tokenization; cannot score.")

        if not predicted_tokens:
            return EvaluationResult(
                score=0.0,
                explanation="Predicted answer produced no tokens (empty or unparseable output).",
                is_success=False,
                metadata={"precision": 0.0, "recall": 0.0},
            )

        overlap = predicted_tokens & target_tokens
        precision = len(overlap) / len(predicted_tokens)
        recall = len(overlap) / len(target_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return EvaluationResult(
            score=f1,
            explanation=(
                f"F1={f1:.3f} (precision={precision:.3f}, recall={recall:.3f}). "
                f"Predicted tokens: {sorted(predicted_tokens)}. Target tokens: {sorted(target_tokens)}."
            ),
            is_success=f1 >= self.threshold,
            metadata={"precision": precision, "recall": recall},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "TaskAccuracyEvaluator requires a labeled target. "
            "Call evaluate_with_target(input, target) instead."
        )
