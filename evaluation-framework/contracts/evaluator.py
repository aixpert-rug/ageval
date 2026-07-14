"""
The WP3 / WP4 interface boundary.

This module is a STRUCTURAL mirror of the `Evaluator` / `EvaluationInput` /
`EvaluationResult` shapes used internally by WP4's `inobo` package (private,
D4.1/D4.2). It intentionally does NOT import from `inobo` — this repo is
public and must not depend on WP4's private codebase.

Because Python's `typing.Protocol` uses structural (duck) typing, any
`WP3Evaluator` subclass defined here satisfies WP4's `Evaluator` ABC at
runtime without a shared base class or import, as long as the field/method
shapes match. That match is a manual contract, not an enforced one — see
"Keeping this in sync" below.

Usage (once handed to WP4):

    agent = ReflexionAgent(
        ...,
        evaluator=WP3FaithfulnessEvaluator(),
    )

WP4 calls `evaluator.evaluate(EvaluationInput(...))` from inside its own
private code; WP3 never sees WP4's internals, WP4 never sees WP3's metric
logic beyond the returned score/explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvaluationInput:
    """Mirrors WP4's `inobo.utils.schemas.EvaluationInput`.

    messages: full message trajectory (LangChain-style message objects or
        dicts with at least `content`, and `tool_calls`/`tool_call_id` where
        applicable). Empty for patterns with no tool-call trace.
    input: the original task input, as text.
    output: the agent's final structured output (dict-like or model instance).
    """

    messages: list[Any]
    input: str
    output: Any


@dataclass(frozen=True)
class EvaluationResult:
    """Mirrors WP4's `EvaluationResult`.

    score: numeric score, scale defined by the evaluator (document it).
    explanation: human-readable justification, surfaced in WP4's logs and
        (for Reflexion/Self-Refine) fed into the next trial's reflection step.
    is_success: pass/fail judgement. If the evaluator doesn't have a natural
        threshold, derive this from `score` at construction time.
    metadata: evaluator-specific extra fields (e.g. sub-scores). Not read by
        WP4's core loop; safe to extend freely.
    """

    score: float
    explanation: str
    is_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """Mirrors WP4's `Evaluator` ABC. Structural, not inherited."""

    def evaluate(self, input: EvaluationInput) -> EvaluationResult: ...


class WP3Evaluator:
    """Base class for WP3 metric implementations.

    Not required by the contract (Protocol satisfaction is structural), but
    gives every WP3 evaluator a consistent place to declare which Level 1
    dimension(s) it serves and which applicability category it belongs to,
    so `docs/pattern_metric_matrix.md` can be generated/checked rather than
    maintained by hand.
    """

    #: Level 1 dimension name(s) this metric contributes to, e.g. ["Explainability"]
    dimensions: tuple[str, ...] = ()

    #: One of "common", "reflective", "trajectory", "governance"
    applicability: str = "common"

    #: Short machine-friendly id, e.g. "faithfulness". Used as the Level 2
    #: catalogue key and in KPI templates' `related_metric` field.
    metric_id: str = ""

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:
        raise NotImplementedError
