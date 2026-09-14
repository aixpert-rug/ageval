"""
Second lightweight contract, same rationale as contracts/governance.py:
Perplexity and ECE (calibration) need token log-probabilities, a
fundamentally different data shape from EvaluationInput's
messages/input/output -- rather than bolt an Optional[logprobs] field
onto the core contract (which every OTHER metric would then have to
ignore), this is a separate, minimal type just for the metrics that
actually need it.

Generating log-probabilities requires access to the model's actual
token-level output (via an OpenAI-compatible API's logprobs parameter,
or direct access to an open-weight model) -- that's the caller's
responsibility, same "someone else must supply this" pattern used
throughout this repo (task_success's oracle_fn, energy_carbon's power
draw). This contract only defines the SHAPE of what's needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLogProbs:
    """
    tokens: the token strings, in generation order (for readability/
        debugging; not used in the math itself).
    logprobs: NATURAL log probabilities (base e, matching standard ML convention), one per token, same order
        as tokens. NOT log10, NOT raw probabilities -- get this wrong
        and every downstream metric using this contract is silently
        wrong too.
    """

    tokens: list[str]
    logprobs: list[float]

    def __post_init__(self):
        if len(self.tokens) != len(self.logprobs):
            raise ValueError(f"tokens ({len(self.tokens)}) and logprobs ({len(self.logprobs)}) must be the same length.")
        if any(lp > 0 for lp in self.logprobs):
            raise ValueError("logprobs must all be <= 0 (they are LOG probabilities of events with probability <= 1).")


@dataclass(frozen=True)
class ConfidenceLabel:
    """One prediction's confidence + whether it was actually correct --
    the input shape ECE needs, aggregated across many predictions."""

    confidence: float  # 0-1
    is_correct: bool

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}.")
