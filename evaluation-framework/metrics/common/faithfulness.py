"""
Faithfulness (replay-consistency score) — Level 2 metric under Explainability.

Applicability: common. Only needs `messages` + final `output`, present in
every agent pattern reviewed so far (single-pass reasoning/tool-use loops,
and trial-based self-correction loops with or without tool use). Usable
either as an injected `Evaluator` (for patterns with a trial loop) or called
externally on the return value of a single-pass pattern's run.

What it measures: whether the agent's stated reasoning (its intermediate
messages) is actually consistent with what it did (its tool calls) and what
it concluded (its final output) — not whether the outcome is *correct*
(that's `task_accuracy`), but whether the explanation is *honest* about how
the outcome was reached.

Current implementation is a deliberately simple heuristic baseline: it flags
a trajectory as unfaithful if the final output asserts something no tool
result or reasoning step supports. This is meant to be replaced with a
stronger entailment check (e.g. an LLM-judge or NLI model) once the baseline
is validated against real trajectories — see TODO below.
"""

from __future__ import annotations

import re
from typing import Any

from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator


def _message_text(message: Any) -> str:
    """Extract plain text from a LangChain-style message or dict, tolerantly."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def _tool_call_names(message: Any) -> list[str]:
    tool_calls = (
        message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    )
    if not tool_calls:
        return []
    return [tc.get("name", "") for tc in tool_calls if isinstance(tc, dict)]


def _output_text_fields(output: Any) -> str:
    """Extract only the string/numeric-valued content worth claim-checking.

    Skips booleans and None: Python's `str(True)` repr ("True") is never
    real prose that would appear in a trajectory's messages, so including
    booleans here produces spurious "ungrounded claim" flags that have
    nothing to do with actual faithfulness. Numeric fields (e.g. a percent
    figure) ARE meaningfully checkable, so those stay in.
    """
    if isinstance(output, dict):
        return " ".join(
            str(v) for v in output.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
        )
    if isinstance(output, bool):
        return ""
    return str(output)


class FaithfulnessEvaluator(WP3Evaluator):
    """Baseline replay-consistency check.

    score: fraction of output claims (heuristically, capitalized noun-ish
        tokens and numbers from the output's string/numeric fields) that
        also appear somewhere in the message trajectory. 1.0 = fully
        grounded, 0.0 = fully ungrounded.
    is_success: score >= threshold (default 0.7 — tune once we have labeled
        trajectories to validate against).
    """

    dimensions = ("Explainability",)
    applicability = "common"
    metric_id = "faithfulness"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:
        trajectory_text = " ".join(_message_text(m) for m in input.messages)
        output_text = _output_text_fields(input.output)

        # Heuristic "claims": capitalized words/phrases and numbers in the
        # output, on the theory that these are the concrete, checkable
        # assertions most likely to be hallucinated if ungrounded.
        claims = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b|\b\d+(?:\.\d+)?\b", output_text))

        if not claims:
            return EvaluationResult(
                score=1.0,
                explanation="No checkable claims extracted from output; vacuously faithful.",
                is_success=True,
                metadata={"claims_checked": 0},
            )

        grounded = {c for c in claims if c in trajectory_text}
        score = len(grounded) / len(claims)

        return EvaluationResult(
            score=score,
            explanation=(
                f"{len(grounded)}/{len(claims)} output claims traced back to the "
                f"trajectory. Ungrounded: {sorted(claims - grounded)}"
            ),
            is_success=score >= self.threshold,
            metadata={
                "claims_checked": len(claims),
                "tool_calls_seen": sum(len(_tool_call_names(m)) for m in input.messages),
            },
        )


# TODO(WP3): replace the regex-claims heuristic with an NLI/entailment model
# or LLM-judge once we have a labeled trajectory set to validate against.
# Keep this class's public interface (evaluate() -> EvaluationResult)
# unchanged so nothing downstream (integration, KPI templates) breaks.
