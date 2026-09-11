"""
Step efficiency -- Level 2 metric under Efficiency (Vector's "Se").

Applicability: trajectory. Needs a tool-call trace to count actual steps
taken, plus a labeled "optimal step count" for the task.

Worth noting explicitly: this metric is dimension-tagged "Efficiency" in
Vector's Appendix, but banded with the "H" (Higher-is-better rate) family,
NOT ratio-to-budget the way Latency/Cost-per-query are -- despite all
being under the same dimension. Formula and banding family are properties
of the metric, not fully determined by dimension; this is the concrete
case that makes that worth stating rather than assuming.

Formula, direct from the Appendix: eta = min(1, L*/L), where L* = the
labeled optimal step count and L = actual steps taken. Capped at 1 so
a trajectory that takes MORE steps than optimal is penalized, and one
that (somehow) takes FEWER than the labeled optimum doesn't score above
1.0 for it -- per Vector's own text: "capped at 1 so detours are
penalized and shortcuts cannot inflate the score."
"""

from __future__ import annotations

from typing import Any, Callable

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _default_step_counter(messages: list[Any], exclude_tool_names: frozenset[str] = frozenset({"finish_task"})) -> int:
    """Default step definition: number of AI-message turns that made at
    least one tool call EXCLUDING the termination call itself (default:
    "finish_task", the actual name used by every pattern reviewed so
    far -- see e.g. ReactAgent's own `self.finish_tool`).

    This exclusion matters: without it, every completed trajectory gets
    +1 step just for finishing, regardless of how much real work it did
    -- confirmed by testing against the real fixtures, where the "42"
    trajectory (zero real tool calls, immediate termination) was
    initially miscounted as 1 step instead of 0 before this exclusion
    was added, which silently prevented the zero-steps guard below from
    ever triggering on exactly the case it exists to catch.

    Override exclude_tool_names (or pass a different step_counter_fn
    entirely) if a pattern uses a different termination-call convention."""
    count = 0
    for message in messages:
        msg_type = _get(message, "type")
        tool_calls = _get(message, "tool_calls")
        if msg_type in ("ai", "assistant") and tool_calls:
            real_calls = [tc for tc in tool_calls if tc.get("name") not in exclude_tool_names]
            if real_calls:
                count += 1
    return count


class StepEfficiencyEvaluator(WP3Evaluator):
    dimensions = ("Efficiency",)
    applicability = "trajectory"
    metric_id = "step_efficiency"

    def __init__(self, step_counter_fn: Callable[[list[Any]], int] = _default_step_counter):
        self.step_counter_fn = step_counter_fn

    def evaluate_with_optimal(self, input: EvaluationInput, optimal_steps: int) -> EvaluationResult:
        """
        Args:
            optimal_steps: labeled reference step count (L*) for this
                task -- the fewest steps a well-behaved agent would need.
        """
        if optimal_steps < 0:
            raise ValueError(f"optimal_steps must be non-negative, got {optimal_steps}.")

        actual_steps = self.step_counter_fn(input.messages)

        if actual_steps == 0:
            if optimal_steps == 0:
                score = 1.0
            else:
                raise ValueError(
                    f"actual_steps is 0 but optimal_steps is {optimal_steps} -- "
                    "undefined ratio. Check Task Success / Reward Hacking for "
                    "this trajectory first; Step Efficiency assumes the task "
                    "was actually completed."
                )
        else:
            score = min(1.0, optimal_steps / actual_steps)

        band = higher_is_better_rate_band(score)

        return EvaluationResult(
            score=score,
            explanation=(
                f"{actual_steps} actual steps vs. {optimal_steps} optimal "
                f"-- efficiency {score:.3f}, band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"actual_steps": actual_steps, "optimal_steps": optimal_steps, "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("StepEfficiencyEvaluator requires a labeled optimal step count. Call evaluate_with_optimal(input, optimal_steps) instead.")
