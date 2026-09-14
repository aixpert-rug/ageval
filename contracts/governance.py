"""
Second contract, separate from evaluator.py: governance/security-style
patterns don't produce EvaluationInput/EvaluationResult at all. They
typically expose a list of escalation or audit records instead. This is a
structural mirror of that shape, believed accurate as of the source
reviewed on [DATE / commit ref once shared] — same rationale as
evaluator.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EscalationEvent:
    """Mirrors a governance pattern's escalation/audit record.

    kind: e.g. "prompt_injection", "missing_policy", "boundary_violation".
    surface: where the check fired, e.g. "input", "tool:search:args", "output".
    reason: human-readable explanation.
    severity: free-text severity label.
    tool_name: set when the event is tool-specific, else None.
    timestamp: ISO 8601 string.
    metadata: kind-specific extra fields (e.g. confidence, detection type).
    """

    kind: str
    surface: str
    reason: str
    severity: str = "high"
    tool_name: str | None = None
    timestamp: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


# A governance-pattern run, for metric purposes, is just its full escalation
# log plus the ground-truth labels for what SHOULD have fired (needed to
# compute recall/precision/false-positive-rate — the escalation log alone
# only tells you what WAS caught, not what was missed).
@dataclass(frozen=True)
class GovernanceTestCase:
    """One labeled test input for a governance pattern evaluation run."""

    input_text: str
    is_actually_malicious: bool
    #: Filled in after running the governed agent/tool on `input_text`.
    escalations: list[EscalationEvent] = field(default_factory=list)
