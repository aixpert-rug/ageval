import json
from pathlib import Path

from contracts.evaluator import EvaluationInput
from metrics.common.faithfulness import FaithfulnessEvaluator
from metrics.trajectory.log_completeness import LogCompletenessEvaluator

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_react_input() -> EvaluationInput:
    data = json.loads((FIXTURES / "mock_react_trajectory.json").read_text())
    return EvaluationInput(
        messages=data["messages"],
        input=data["input"],
        output=data["structured_response"],
    )


def test_faithfulness_on_react_fixture():
    result = FaithfulnessEvaluator().evaluate(_load_react_input())
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["claims_checked"] > 0


def test_log_completeness_on_react_fixture():
    result = LogCompletenessEvaluator().evaluate(_load_react_input())
    # Fixture is hand-built with matched call/response ids for both tool
    # calls, so this should be perfectly complete.
    assert result.score == 1.0
    assert result.is_success is True
    assert result.metadata["tool_calls_seen"] == 2


def test_log_completeness_vacuous_when_no_tools():
    empty_input = EvaluationInput(messages=[{"type": "human", "content": "hi"}], input="hi", output={})
    result = LogCompletenessEvaluator().evaluate(empty_input)
    assert result.score == 1.0
    assert result.metadata["tool_calls_seen"] == 0

def test_faithfulness_ignores_boolean_output_fields():
    """Regression test: str(dict) rendering of a boolean output field
    (e.g. bring_umbrella: True) used to get extracted as a spurious
    'ungrounded claim' since Python's str(True) == 'True' never appears
    as prose in a trajectory. Output dicts with only a boolean + a
    grounded numeric field should score 1.0, not be dragged down by the
    boolean."""
    output_with_bool = {"bring_umbrella": True}
    trajectory_input = EvaluationInput(
        messages=[{"type": "human", "content": "no numbers or claims here"}],
        input="irrelevant",
        output=output_with_bool,
    )
    result = FaithfulnessEvaluator().evaluate(trajectory_input)
    # No string/numeric claims at all once the boolean is excluded ->
    # vacuously faithful, not penalized for the boolean.
    assert result.score == 1.0
    assert result.metadata["claims_checked"] == 0

def test_faithfulness_catches_unsupported_claim():
    """Negative control: output claims something the trajectory never
    established. Score should be low, not high."""
    trajectory_input = EvaluationInput(
        messages=[
            {"type": "human", "content": "location: Groningen"},
            {"type": "tool", "tool_call_id": "call_1", "name": "get_weather",
             "content": "Groningen: 14C, sunny"},
        ],
        input="What's the weather?",
        output={"summary": "It's 25C and stormy in Amsterdam"},
    )
    result = FaithfulnessEvaluator().evaluate(trajectory_input)
    assert result.score < 0.5
    assert result.is_success is False


def test_log_completeness_catches_dangling_tool_call():
    """Negative control: a tool call with no matching ToolMessage response.
    Score should reflect the incompleteness, not read as fully complete."""
    trajectory_input = EvaluationInput(
        messages=[
            {"type": "human", "content": "location: Groningen"},
            {
                "type": "ai",
                "content": "Checking weather.",
                "tool_calls": [{"id": "call_1", "name": "get_weather", "args": {"location": "Groningen"}}],
            },
            # No matching ToolMessage for call_1 -- dangling call.
        ],
        input="irrelevant",
        output={},
    )
    result = LogCompletenessEvaluator().evaluate(trajectory_input)
    assert result.score < 1.0
    assert "call_1" in result.metadata["incomplete_ids"]
