"""
Tool-call accuracy -- Level 2 metric under Accuracy.

Applicability: trajectory. Compares the tool calls the agent made with a
labelled list of expected calls (name, and optionally arguments).

  recall    = matched / expected      did it make the calls it needed?
  precision = matched / actual        did it avoid unneeded calls?
  score     = F1(precision, recall)   (or recall alone with penalize_extra=False)

Matching is ONE-TO-ONE: an actual call satisfies at most one expected call.
Arguments, when the expected entry has an "args" key, are checked (every
expected key must be present with an equal value) unless check_args=False.
ordered=True additionally requires the matched calls to appear in the
expected order. Calls to tools in `ignore_tools` (the pattern's own
terminal tool) are removed before matching.
"""

from __future__ import annotations

from typing import Any, Sequence

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
            calls.append({"name": tc.get("name"), "args": tc.get("args", {}) or {}})
    return calls


class ToolCallAccuracyEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "trajectory"
    metric_id = "tool_call_accuracy"

    def __init__(self, check_args: bool = True, ordered: bool = False, penalize_extra: bool = True,
                 ignore_tools: Sequence[str] = ("finish_task",)):
        self.check_args = check_args
        self.ordered = ordered
        self.penalize_extra = penalize_extra
        self.ignore_tools = frozenset(ignore_tools)

    def _ok(self, expected: dict, actual: dict) -> bool:
        if actual["name"] != expected["name"]:
            return False
        if self.check_args and "args" in expected:
            return all(k in actual["args"] and actual["args"][k] == v for k, v in expected["args"].items())
        return True

    def _match(self, expected: list[dict], actual: list[dict]) -> dict[int, int]:
        """Returns {expected index: actual index}."""
        if self.ordered:
            pairs, pos = {}, -1
            for e, exp in enumerate(expected):
                for a in range(pos + 1, len(actual)):
                    if self._ok(exp, actual[a]):
                        pairs[e], pos = a, a
                        break
            return pairs
        owner: dict[int, int] = {}          # actual index -> expected index (maximum bipartite matching)

        def assign(e: int, seen: set[int]) -> bool:
            for a in range(len(actual)):
                if a not in seen and self._ok(expected[e], actual[a]):
                    seen.add(a)
                    if a not in owner or assign(owner[a], seen):
                        owner[a] = e
                        return True
            return False

        for e in range(len(expected)):
            assign(e, set())
        return {e: a for a, e in owner.items()}

    def evaluate_with_expected_calls(self, input: EvaluationInput, expected_calls: list[dict]) -> EvaluationResult:
        if not expected_calls:
            raise ValueError("evaluate_with_expected_calls requires at least one expected call.")

        actual = [c for c in _actual_tool_calls(input.messages) if c["name"] not in self.ignore_tools]
        pairs = self._match(expected_calls, actual)
        matched = len(pairs)

        recall = matched / len(expected_calls)
        precision = matched / len(actual) if actual else None
        f1 = 0.0 if not matched else 2 * precision * recall / (precision + recall)
        score = f1 if self.penalize_extra else recall
        band = higher_is_better_rate_band(score)

        return EvaluationResult(
            score=score,
            explanation=(
                f"recall {recall:.2f} ({matched}/{len(expected_calls)} expected calls found), "
                f"precision {'n/a' if precision is None else f'{precision:.2f}'} ({matched}/{len(actual)} actual calls needed) "
                f"-- band {band.level} ({band.label}). Actual calls: {[c['name'] for c in actual]}."
            ),
            is_success=band.level >= 2,
            metadata={
                "recall": recall, "precision": precision, "f1": f1,
                "matched_pairs": {e: a for e, a in sorted(pairs.items())},
                "unmatched_expected": [expected_calls[e] for e in range(len(expected_calls)) if e not in pairs],
                "extra_actual": [actual[a] for a in range(len(actual)) if a not in pairs.values()],
                "actual_tool_names": [c["name"] for c in actual],
                "band_level": band.level, "band_label": band.label,
            },
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("ToolCallAccuracyEvaluator requires an expected-calls spec. Call evaluate_with_expected_calls(input, expected_calls) instead.")
