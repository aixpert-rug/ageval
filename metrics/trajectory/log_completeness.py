from __future__ import annotations
from typing import Any
from contracts.banding import higher_is_better_rate_band
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
            tool_call_id = _get(message, "tool_call_id")
            if tool_call_id:
                response_ids.add(tool_call_id)
        if not call_ids:
            return EvaluationResult(score=1.0, explanation="No tool calls in trajectory.", is_success=True, metadata={"tool_calls_seen": 0, "band_level": 4, "band_label": "Excellent"})
        complete = [cid for cid, call in call_ids.items() if cid in response_ids and call.get("name") and call.get("args") is not None]
        score = len(complete) / len(call_ids)
        incomplete_ids = set(call_ids) - set(complete)
        # score is a genuine continuous ratio (e.g. 3/4 complete = 0.75),
        # not a degenerate binary value -- this was previously computed
        # but never banded, an oversight worth correcting, unlike
        # exact_match/task_success where the score truly is only ever
        # 0 or 1 and banding a single binary outcome adds no information.
        # is_success deliberately stays a STRICT score==1.0 check, not the
        # generic band.level>=2 -- for an auditability metric specifically,
        # even one dangling/incomplete tool call is a real integrity
        # problem, not "Adequate." Band info is added context, not a
        # loosened pass bar.
        band = higher_is_better_rate_band(score)
        return EvaluationResult(
            score=score,
            explanation=f"{len(complete)}/{len(call_ids)} tool calls complete. Incomplete: {sorted(incomplete_ids)} -- band {band.level} ({band.label}).",
            is_success=score == 1.0,
            metadata={"tool_calls_seen": len(call_ids), "incomplete_ids": sorted(incomplete_ids), "band_level": band.level, "band_label": band.label},
        )
