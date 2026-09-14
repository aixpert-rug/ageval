"""
BERTScore -- Level 2 metric under Accuracy.

Applicability: common. Embedding-similarity reference-comparison metric.
See module history for the real, confirmed-by-testing dependency weight
(torch + transformers + a HuggingFace model download) and its failure
modes.

Banded via contracts.banding.higher_is_better_rate_band

VERIFIED: end-to-end scoring confirmed working on real infrastructure
(distilbert-base-uncased downloaded and scored correctly, F1=0.940 for
a genuine near-paraphrase pair).
"""

from __future__ import annotations

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator
from metrics.common.task_accuracy import _extract_answer_text


class BERTScoreEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "bertscore"

    def __init__(self, model_type: str = "distilbert-base-uncased", lang: str = "en"):
        self.model_type = model_type
        self.lang = lang
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is None:
            try:
                from bert_score import BERTScorer
            except ImportError as e:
                raise RuntimeError(
                    "BERTScoreEvaluator requires the 'bert-score' package "
                    "(pip install bert-score) plus torch and transformers."
                ) from e
            try:
                self._scorer = BERTScorer(model_type=self.model_type, lang=self.lang)
            except OSError as e:
                raise RuntimeError(
                    f"Could not load BERTScore's underlying model ({self.model_type!r}). "
                    f"Usually means no network route to huggingface.co. Original error: {e}"
                ) from e
        return self._scorer

    def evaluate_with_target(self, input: EvaluationInput, target: str) -> EvaluationResult:
        hypothesis = _extract_answer_text(input.output)
        if not hypothesis.strip():
            return EvaluationResult(score=0.0, explanation="Empty hypothesis text.", is_success=False, metadata={"band_level": 0})

        scorer = self._get_scorer()
        precision, recall, f1 = scorer.score([hypothesis], [target])
        p, r, f = precision.item(), recall.item(), f1.item()
        band = higher_is_better_rate_band(f)

        return EvaluationResult(
            score=f,
            explanation=(
                f"BERTScore F1: {f:.3f} (precision={p:.3f}, recall={r:.3f}, "
                f"model={self.model_type}) -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"precision": p, "recall": r, "f1": f, "model_type": self.model_type, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("BERTScoreEvaluator requires a labeled target. Call evaluate_with_target(input, target) instead.")
