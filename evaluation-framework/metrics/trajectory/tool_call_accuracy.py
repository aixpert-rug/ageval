"""
Tool-call accuracy -- Level 2 metric under Accuracy.

Applicability: trajectory. Needs a tool-call trace PLUS a labeled
expectation of which tool(s) should have been called for the task.

Banded via contracts.banding.higher_is_better_rate_band (the same
0.20/0.40/0.60/0.80 cuts used by F1, BLEU, etc.).

This is the metric that would have caught the "17 + 25 = 42" bug more
precisely than log_completeness did: that trajectory has ZERO tool
calls, so log_completeness scored it 1.0 (vacuously complete -- nothing
to be incomplete). Tool-call accuracy, given the expectation that `add`
should have been called, correctly scores it 0.0 -- the agent bypassed
an available, relevant tool. See tests, run against that exact real
fixture.
"""

from __future__ import annotations

from typing import Any

from contracts.banding import higher_is_better_rate_band
from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _actual_tool_calls(messages: list[Any]) -> list[dict]:
    calls = []
    for message in messages:
        for tc in _get(message, "tool_calls", []) or []:
            calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
    return calls


class ToolCallAccuracyEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "trajectory"
    metric_id = "tool_call_accuracy"

    def __init__(self, check_args: bool = False):
        self.check_args = check_args

    def evaluate_with_expected_calls(self, input: EvaluationInput, expected_calls: list[dict]) -> EvaluationResult:
        if not expected_calls:
            raise ValueError("evaluate_with_expected_calls requires at least one expected call.")

        actual_calls = _actual_tool_calls(input.messages)
        matched = 0
        match_details = []

        for expected in expected_calls:
            found = False
            for actual in actual_calls:
                if actual["name"] != expected["name"]:
                    continue
                if self.check_args and "args" in expected:
                    actual_args = actual.get("args", {})
                    if not all(actual_args.get(k) == v for k, v in expected["args"].items()):
                        continue
                found = True
                break
            match_details.append({"expected": expected, "matched": found})
            if found:
                matched += 1

        score = matched / len(expected_calls)
        band = higher_is_better_rate_band(score)

        return EvaluationResult(
            score=score,
            explanation=(
                f"{matched}/{len(expected_calls)} expected tool calls found in trajectory -- "
                f"band {band.level} ({band.label}). Actual tool calls made: {[c['name'] for c in actual_calls]}."
            ),
            is_success=band.level >= 2,
            metadata={"match_details": match_details, "actual_tool_names": [c["name"] for c in actual_calls], "band_level": band.level, "band_label": band.label},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("ToolCallAccuracyEvaluator requires an expected-calls spec. Call evaluate_with_expected_calls(input, expected_calls) instead.")
