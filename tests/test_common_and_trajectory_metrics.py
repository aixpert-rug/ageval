import json
from pathlib import Path

from contracts.evaluator import EvaluationInput
from metrics.trajectory.log_completeness import LogCompletenessEvaluator

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_react_input() -> EvaluationInput:
    data = json.loads((FIXTURES / "mock_react_trajectory.json").read_text())
    return EvaluationInput(
        messages=data["messages"],
        input=data["input"],
        output=data["structured_response"],
    )




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
