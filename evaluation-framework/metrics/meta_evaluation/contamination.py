"""
Contamination / leakage -- Level 2 meta-evaluation metric under Auditability / Verifiability.

Applicability: meta_evaluation. Needs the train and test item sets (or
a precomputed overlap fraction) for a benchmark -- % of test items that
also appeared in training data.

IMPORTANT: contamination >= 0.40 is documented to VOID
the affected benchmark entirely: "its score is withdrawn, the construct
it covered is marked N/A for that submission." Different veto SHAPE
from Unsafe-action rate's profile-wide cap: this one invalidates a
SPECIFIC BENCHMARK's result, not the whole trustworthiness profile.
Same scope caveat as unsafe_action_rate.py: this evaluator flags
`is_vetoed` in its own result; actually withdrawing the affected
benchmark score and marking the construct N/A is the job of whatever
aggregates benchmark results (not built yet).

Per the Appendix: Norm="L" (Lower-is-better rate, fixed cuts).
"""

from __future__ import annotations

from contracts.banding import lower_is_better_rate_band
from contracts.evaluator import EvaluationResult, WP3Evaluator

VETO_THRESHOLD = 0.40 


class ContaminationEvaluator(WP3Evaluator):
    dimensions = ("Auditability / Verifiability",)
    applicability = "meta_evaluation"
    metric_id = "contamination"

    def evaluate_overlap(self, train_items: set, test_items: set) -> EvaluationResult:
        """
        Args:
            train_items, test_items: sets of item identifiers (e.g.
                question hashes, exact-match text) -- computes exact-set
                overlap. For approximate/near-duplicate contamination
                (paraphrases, minor edits), a fuzzy-matching approach
                would be needed instead -- not implemented here; this
                only catches exact overlap.
        """
        if not test_items:
            raise ValueError("test_items must be non-empty.")

        overlap = train_items & test_items
        rate = len(overlap) / len(test_items)
        return self._band_and_wrap(rate, len(overlap), len(test_items))

    def evaluate_measurement(self, contamination_rate: float) -> EvaluationResult:
        """Accepts an already-computed contamination fraction, for cases
        where exact-set overlap isn't the right computation (e.g. a
        near-duplicate detection pipeline was used instead)."""
        if not (0.0 <= contamination_rate <= 1.0):
            raise ValueError(f"contamination_rate must be in [0,1], got {contamination_rate}.")
        return self._band_and_wrap(contamination_rate, None, None)

    def _band_and_wrap(self, rate: float, overlap_count, test_count) -> EvaluationResult:
        band = lower_is_better_rate_band(rate)
        is_vetoed = rate >= VETO_THRESHOLD
        score = band.level / 4.0

        explanation = f"Contamination rate {rate:.3f} -- band {band.level} ({band.label})."
        if is_vetoed:
            explanation += (
                f" VETO TRIGGERED: rate >= {VETO_THRESHOLD} -- per veto rules, this benchmark's "
                f"score should be WITHDRAWN and the construct it covers marked N/A for this submission, "
                f"not just penalised. Propagate is_vetoed to whatever aggregates benchmark results."
            )

        return EvaluationResult(
            score=score,
            explanation=explanation,
            is_success=(not is_vetoed) and band.level >= 2,
            metadata={
                "contamination_rate": rate, "overlap_count": overlap_count, "test_count": test_count,
                "band_level": band.level, "band_label": band.label,
                "is_vetoed": is_vetoed, "veto_threshold": VETO_THRESHOLD,
            },
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("ContaminationEvaluator operates on train/test item sets or a precomputed rate, not a single EvaluationInput. Call evaluate_overlap(...) or evaluate_measurement(...) instead.")
