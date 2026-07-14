"""
Log completeness score — Level 2 metric under Transparency.

Applicability: trajectory. Needs a tool-call trace (ReAct, Reflexion).
Near-vacuous for Self-Refine (no tools) — will always return 1.0 with
zero calls checked; that's a valid result, not a bug, but flag it as such
in reporting rather than treating a 1.0 score there as meaningful signal.

What it measures: for every tool call in the trajectory, is there a
complete, well-formed record of it — a call with a name and args, and a
matching ToolMessage response with the same tool_call_id? Incomplete
records (a call with no matching response, or a response missing its
linking id) undermine both transparency (can a stakeholder actually follow
what happened) and auditability (can it be verified after the fact).
"""

from __future__ import annotations

from typing import Any

from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class LogCompletenessEvaluator(WP3Evaluator):
    dimensions = ("Transparency", "Auditability / Verifiability")
    applicability = "trajectory"
    metric_id = "log_completeness"

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:
        call_ids: dict[str, dict] = {}
        response_ids: set[str] = set()

        for message in input.messages:
            for tool_call in _get(message, "tool_calls", []) or []:
                call_id = tool_call.get("id")
                if call_id:
                    call_ids[call_id] = tool_call

            # A ToolMessage response
            tool_call_id = _get(message, "tool_call_id")
            if tool_call_id:
                response_ids.add(tool_call_id)

        if not call_ids:
            return EvaluationResult(
                score=1.0,
                explanation="No tool calls in trajectory — vacuously complete. "
                "Expected for tool-free patterns (e.g. Self-Refine); treat as "
                "non-signal in that case, not as evidence of good logging.",
                is_success=True,
                metadata={"tool_calls_seen": 0},
            )

        complete = [
            call_id
            for call_id, call in call_ids.items()
            if call_id in response_ids and call.get("name") and call.get("args") is not None
        ]
        score = len(complete) / len(call_ids)
        incomplete_ids = set(call_ids) - set(complete)

        return EvaluationResult(
            score=score,
            explanation=(
                f"{len(complete)}/{len(call_ids)} tool calls have a complete "
                f"call+response record. Incomplete call ids: {sorted(incomplete_ids)}"
            ),
            is_success=score == 1.0,
            metadata={"tool_calls_seen": len(call_ids), "incomplete_ids": sorted(incomplete_ids)},
        )
