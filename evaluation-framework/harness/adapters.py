"""
The "match WP4's outputs to our inputs" step -- the piece that was
missing before. Confirmed against real code from three WP4 patterns
(ReactAgent, ReflexionAgent, BasicRAG) that a compiled pattern's graph
state, once a run completes, exposes exactly two things every metric in
this repo needs: `messages` (the full trajectory) and, where the
pattern produces one, `structured_response` (the validated final
output).

Deliberately thin: EvaluationInput's `messages` field, and every
metric's `_get()` helper throughout this repo, were already designed
tolerant of BOTH shapes -- raw LangChain message objects (attribute
access: `.content`, `.tool_calls`) from a live `.invoke()` call, and
plain dicts (from a JSON fixture). Verified directly during development
of this file: log_completeness, trajectory_consistency, and
tool_call_accuracy all produce correct results against fake objects
built to mimic real LangChain's actual shape -- containing MESSAGES are
objects, but individual `tool_calls` entries are plain dicts even on a
real message object (confirmed independently by ReflexionAgent's own
code: `finish_call.get("args")`, called directly on a tool_calls entry).
So this adapter does almost no transformation -- that's not
laziness, it's confirmation the contract was designed correctly the
first time.
"""

from __future__ import annotations

from contracts.evaluator import EvaluationInput


def from_wp4_pattern_result(result: dict, input_query: str) -> EvaluationInput:
    """
    Converts a WP4 pattern's raw `.invoke()` (or `.run()`) return value
    into an EvaluationInput.

    CONFIRMED working for:
        - ReactAgent (real trajectory already captured and tested
          throughout this repo).
        - ReflexionAgent's FINAL state, after `_respond` runs -- its
          `_respond` builds `structured_response` via the identical
          `output_schema(**tool_calls[0]["args"])` pattern ReactAgent
          uses. NOT yet confirmed for ReflexionAgent's per-TRIAL data
          (see harness module notes -- that needs either a checkpointer
          check or a small WP4 interface addition, still open).

    NOT YET confirmed for BasicRAG -- its `run()` return value's exact
    keys (does it include `structured_response` at all, or only
    `messages` + `context`?) haven't been verified against real output.
    Test this specifically before relying on it for RAG evaluation.

    Args:
        result: the dict returned by calling the pattern (e.g.
            `pattern.run(state)` or the compiled graph's `.invoke(state)`).
        input_query: the original task input -- not reliably recoverable
            from `result` alone across all patterns, so passed in
            explicitly rather than guessed from state.
    """
    if "messages" not in result:
        raise ValueError(
            "result has no 'messages' key -- this doesn't look like a WP4 pattern's "
            "graph state. If this IS a WP4 pattern result, its state shape may differ "
            "from what this adapter assumes; check the pattern's actual return value."
        )

    return EvaluationInput(
        messages=result["messages"],
        input=input_query,
        output=result.get("structured_response"),
    )
