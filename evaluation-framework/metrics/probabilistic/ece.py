"""
ECE (Expected Calibration Error) -- Level 2 metric under Transparency &
Explainability .

Applicability: probabilistic. Needs a batch of (confidence, is_correct)
pairs (contracts.logprobs.ConfidenceLabel) across many predictions --
not a single trajectory. Confidence values typically come from
log-probabilities (e.g. softmax probability of the chosen answer), same
"caller must supply, we only aggregate" pattern as Perplexity.

Formula: ECE = sum_m (|B_m|/n) *
|acc(B_m) - conf(B_m)|, over M confidence bins. Verified against a known
reference case before use: 10 predictions all at confidence 0.9, 8/10
actually correct -> ECE = |0.8 - 0.9| = 0.1 exactly (single bin, since
all confidences are identical).

Banding: Table 5 DOES give an explicit default ceiling for ECE
specifically (0.20, under the "Lower-is-better continuous" family) --
used as this evaluator's default.
"""

from __future__ import annotations

from contracts.banding import lower_is_better_continuous_band
from contracts.evaluator import EvaluationResult, WP3Evaluator
from contracts.logprobs import ConfidenceLabel

_ECE_DEFAULT_BUDGET_CEILING = 0.20  

class ECEEvaluator(WP3Evaluator):
    dimensions = ("Transparency & Explainability",)
    applicability = "probabilistic"
    metric_id = "ece"

    def evaluate_predictions(
        self,
        predictions: list[ConfidenceLabel],
        num_bins: int = 10,
        budget_ceiling: float = _ECE_DEFAULT_BUDGET_CEILING,
    ) -> EvaluationResult:
        if not predictions:
            raise ValueError("evaluate_predictions requires at least one prediction.")
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}.")

        bins: list[list[ConfidenceLabel]] = [[] for _ in range(num_bins)]
        for p in predictions:
            bin_idx = min(int(p.confidence * num_bins), num_bins - 1)
            bins[bin_idx].append(p)

        n = len(predictions)
        ece = 0.0
        bin_details = []
        for i, bin_items in enumerate(bins):
            if not bin_items:
                continue
            bin_acc = sum(1 for p in bin_items if p.is_correct) / len(bin_items)
            bin_conf = sum(p.confidence for p in bin_items) / len(bin_items)
            weight = len(bin_items) / n
            ece += weight * abs(bin_acc - bin_conf)
            bin_details.append({"bin_index": i, "n": len(bin_items), "accuracy": bin_acc, "avg_confidence": bin_conf})

        band = lower_is_better_continuous_band(ece, budget_ceiling=budget_ceiling)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=f"ECE = {ece:.4f} across {n} predictions, {num_bins} bins -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"ece": ece, "n_predictions": n, "num_bins": num_bins, "bin_details": bin_details, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("ECEEvaluator operates on a batch of ConfidenceLabel predictions, not a single EvaluationInput. Call evaluate_predictions(...) instead.")
