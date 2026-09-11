"""
Latency -- Level 2 metric under Efficiency
Applicability: efficiency. Two ways to use this, both supported:

1. evaluate_measurement(elapsed_seconds, budget_seconds) -- for
   reprocessing an already-measured value (e.g. a timestamp diff pulled
   from existing logs).
2. measure_and_evaluate(fn, budget_seconds) -- LIVE measurement: runs
   fn() itself, times it with time.perf_counter(), and scores the
   result. This is the one worth using when there's no existing
   measurement to reprocess -- added specifically because a banding-only
   metric with no way to actually produce the number it bands isn't
   genuinely usable on its own; someone still has to build the
   measurement half, and if that never happens the metric goes unused.

The budget itself is NOT this evaluator's business to define -- per
Vector's own document, "partner declares a use-case budget B". In this
repo's terms, that's exactly what the Level 3 KPI template's
`Unit / Range` field is for -- the budget lives there, per UC, not
hardcoded here.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from contracts.banding import ratio_to_budget_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class LatencyEvaluator(WP3Evaluator):
    dimensions = ("Efficiency",)
    applicability = "efficiency"
    metric_id = "latency"

    def evaluate_measurement(self, elapsed_seconds: float, budget_seconds: float) -> EvaluationResult:
        if elapsed_seconds < 0:
            raise ValueError(f"elapsed_seconds must be non-negative, got {elapsed_seconds}.")

        band = ratio_to_budget_band(elapsed_seconds, budget_seconds, lower_is_better=True)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=(
                f"{elapsed_seconds:.2f}s against a {budget_seconds:.2f}s budget "
                f"-- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"elapsed_seconds": elapsed_seconds, "budget_seconds": budget_seconds, "band_level": band.level, "band_label": band.label},
        )

    def measure_and_evaluate(self, fn: Callable[[], Any], budget_seconds: float) -> tuple[Any, EvaluationResult]:
        """
        Args:
            fn: zero-argument callable representing the work to time
                (e.g. `lambda: agent.run(state)`). Use functools.partial
                or a lambda to bind arguments.
            budget_seconds: declared budget for this call.

        Returns:
            (fn's return value, EvaluationResult) -- both the actual
            work's output and the efficiency score, from one call.
        """
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        return result, self.evaluate_measurement(elapsed_seconds=elapsed, budget_seconds=budget_seconds)

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("LatencyEvaluator operates on a pre-measured value or a live call. Call evaluate_measurement(...) or measure_and_evaluate(...) instead.")
