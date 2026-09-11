"""
Throughput -- Level 2 metric under Efficiency (Vector's "Th").

Applicability: efficiency. Same two-mode support as latency.py.
measure_and_evaluate() times a call AND expects it to return a count of
work units processed (e.g. tokens generated, queries handled) -- rate is
computed as units/elapsed automatically.

Direction note: unlike Latency/Cost/Memory/Energy, Throughput is
HIGHER-is-better (Appendix Dir: up-arrow) -- more tokens/queries per
second is better. Uses ratio_to_budget_band's lower_is_better=False
branch.
"""

from __future__ import annotations

import time
from typing import Callable

from contracts.banding import ratio_to_budget_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class ThroughputEvaluator(WP3Evaluator):
    dimensions = ("Efficiency",)
    applicability = "efficiency"
    metric_id = "throughput"

    def evaluate_measurement(self, throughput: float, budget: float, unit: str = "tokens/sec") -> EvaluationResult:
        if throughput < 0:
            raise ValueError(f"throughput must be non-negative, got {throughput}.")

        band = ratio_to_budget_band(throughput, budget, lower_is_better=False)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=f"{throughput:.2f} {unit} against a {budget:.2f} {unit} target -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"throughput": throughput, "budget": budget, "unit": unit, "band_level": band.level, "band_label": band.label},
        )

    def measure_and_evaluate(self, fn: Callable[[], int], budget: float, unit: str = "tokens/sec") -> tuple[int, EvaluationResult]:
        """
        Args:
            fn: zero-argument callable that does the work AND RETURNS the
                count of work units processed (e.g. number of tokens
                generated) -- not the work's own output, the unit count.
            budget: target throughput.

        Raises:
            ValueError if elapsed time is at or below a tiny floor
            (~1 microsecond) -- rather than silently returning an
            effectively-infinite throughput from a degenerate/near-zero
            timing measurement, which would trivially pass banding
            (ratio >= 1.0 always true) without meaning anything.
        """
        start = time.perf_counter()
        work_units = fn()
        elapsed = time.perf_counter() - start

        if elapsed <= 1e-6:
            raise ValueError(
                f"Measured elapsed time ({elapsed}s) is at or below the measurement floor -- "
                "too close to zero to compute a meaningful throughput. This usually means fn() "
                "did negligible work or the timer resolution was exceeded, not that throughput "
                "was actually near-infinite."
            )

        throughput = work_units / elapsed
        return work_units, self.evaluate_measurement(throughput=throughput, budget=budget, unit=unit)

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("ThroughputEvaluator operates on a pre-measured value or a live call. Call evaluate_measurement(...) or measure_and_evaluate(...) instead.")
