from dataclasses import dataclass, field

import pytest

from harness.adapters import from_wp4_pattern_result
from metrics.trajectory.log_completeness import LogCompletenessEvaluator


@dataclass
class FakeAIMessage:
    content: str
    tool_calls: list = field(default_factory=list)
    type: str = "ai"


@dataclass
class FakeToolMessage:
    content: str
    tool_call_id: str
    name: str
    type: str = "tool"


@dataclass
class FakeHumanMessage:
    content: str
    type: str = "human"


def _fake_wp4_result() -> dict:
    return {
        "messages": [
            FakeHumanMessage(content="query: What is 17 plus 25?"),
            FakeAIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 17, "b": 25}, "id": "c1"}]),
            FakeToolMessage(content="42", tool_call_id="c1", name="add"),
            FakeAIMessage(content="", tool_calls=[{"name": "finish_task", "args": {"answer": "42"}, "id": "c2"}]),
            FakeToolMessage(content="Task completed.", tool_call_id="c2", name="finish_task"),
        ],
        "structured_response": {"answer": "42"},
    }


def test_adapter_produces_usable_evaluation_input():
    result = _fake_wp4_result()
    trajectory = from_wp4_pattern_result(result, input_query="What is 17 plus 25?")
    assert trajectory.input == "What is 17 plus 25?"
    assert trajectory.output == {"answer": "42"}
    assert len(trajectory.messages) == 5


def test_adapter_output_is_genuinely_scoreable_by_real_metrics():
    result = _fake_wp4_result()
    trajectory = from_wp4_pattern_result(result, input_query="What is 17 plus 25?")
    lc_result = LogCompletenessEvaluator().evaluate(trajectory)
    assert lc_result.score == 1.0


def test_adapter_handles_pattern_with_no_structured_response():
    result = {"messages": [FakeHumanMessage(content="q")], "context": ["some retrieved text"]}
    trajectory = from_wp4_pattern_result(result, input_query="q")
    assert trajectory.output is None


def test_adapter_rejects_result_with_no_messages_key():
    with pytest.raises(ValueError, match="no 'messages' key"):
        from_wp4_pattern_result({"structured_response": {"x": 1}}, input_query="q")
