"""
The missing piece that connects a captured trajectory to the metrics
that need gold labels to score it. Every metric that isn't fully
automatic (trajectory_consistency, log_completeness, refusal all need
nothing beyond the trajectory itself) needs SOMETHING extra -- a target
answer, an expected tool call, an oracle function. Right now every test
in this repo hand-builds that extra data inline; in real use, it needs
to live alongside the task itself, not be improvised per evaluation run.

A TaskDefinition is a benchmark record: the input to give the agent,
plus every gold label any applicable metric might need to score the
resulting trajectory. Not every field is required for every task --
only fill in what the metrics you actually plan to run need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    input_query: str

    target: str | None = None
    expected_tool_calls: list[dict] | None = None
    oracle_fn: Callable[[Any], bool] | None = None
    subgoal_checks: list[Callable[[Any], bool]] | None = None
    optimal_steps: int | None = None
    toxicity_classifier: Callable[[str], float] | None = None

    metadata: dict = field(default_factory=dict)
