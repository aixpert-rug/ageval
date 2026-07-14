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
