"""
Memory footprint -- Level 2 metric under Efficiency .

Applicability: efficiency. Same two-mode support as latency.py:
evaluate_measurement() for a pre-measured value, measure_and_evaluate()
for live measurement.

IMPORTANT caveat on the default live measurer: it uses `tracemalloc`
(Python stdlib), which tracks PYTHON-OBJECT memory allocations only. For
an LLM agent, the overwhelming majority of memory is NOT in Python
objects -- it's in model weights and activations living in GPU VRAM (or
off-heap native memory for CPU inference), which tracemalloc cannot see
at all. Using the default measurer on an actual agent call will report a
number far too low to be meaningful.

The default is still useful for measuring THIS repo's own Python-side
memory use (e.g. building large trajectory logs, batch evaluation
state), which is a legitimate thing to want to measure. For measuring an
agent's real memory footprint, pass a custom `memory_measurer` -- e.g.
one wrapping `torch.cuda.max_memory_allocated()` for GPU workloads, or
shelling out to `nvidia-smi` for a process-level view. This repo doesn't
ship a GPU-specific measurer since it would make torch a hard dependency
for a metric most callers won't run on GPU workloads at all.
"""

from __future__ import annotations

import tracemalloc
from typing import Any, Callable

from contracts.banding import ratio_to_budget_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


def _tracemalloc_measurer(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Default measurer: Python-object memory only. See module docstring
    for why this is NOT appropriate for measuring GPU-resident agent
    memory -- swap in a custom measurer for that case."""
    tracemalloc.start()
    try:
        result = fn()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak_bytes / (1024 * 1024)


class MemoryFootprintEvaluator(WP3Evaluator):
    dimensions = ("Efficiency",)
    applicability = "efficiency"
    metric_id = "memory_footprint"

    def evaluate_measurement(self, memory_mb: float, budget_mb: float) -> EvaluationResult:
        if memory_mb < 0:
            raise ValueError(f"memory_mb must be non-negative, got {memory_mb}.")

        band = ratio_to_budget_band(memory_mb, budget_mb, lower_is_better=True)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=f"{memory_mb:.1f} MB against a {budget_mb:.1f} MB budget -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"memory_mb": memory_mb, "budget_mb": budget_mb, "band_level": band.level, "band_label": band.label},
        )

    def measure_and_evaluate(
        self, fn: Callable[[], Any], budget_mb: float, memory_measurer: Callable[[Callable[[], Any]], tuple[Any, float]] = _tracemalloc_measurer
    ) -> tuple[Any, EvaluationResult]:
        """
        Args:
            fn: zero-argument callable representing the work to measure.
            budget_mb: declared memory budget in MB.
            memory_measurer: (fn) -> (fn's result, peak_mb). Defaults to
                a tracemalloc-based measurer -- see module docstring for
                its Python-object-only limitation. Pass a GPU-aware
                measurer for agent workloads.
        """
        result, peak_mb = memory_measurer(fn)
        return result, self.evaluate_measurement(memory_mb=peak_mb, budget_mb=budget_mb)

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("MemoryFootprintEvaluator operates on a pre-measured value or a live call. Call evaluate_measurement(...) or measure_and_evaluate(...) instead.")
