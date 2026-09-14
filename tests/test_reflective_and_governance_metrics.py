import json
from pathlib import Path

from contracts.governance import EscalationEvent, GovernanceTestCase
from metrics.governance.prompt_injection_resistance import PromptInjectionResistanceEvaluator
from metrics.reflective.consistency import ConsistencyEvaluator

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_consistency_improves_across_reflexion_trials():
    data = json.loads((FIXTURES / "mock_reflexion_trajectory.json").read_text())
    result = ConsistencyEvaluator().evaluate_trials(data["trial_scores"])
    # Fixture scores rise 0.2 -> 0.6 -> 0.95: real spread, so consistency
    # should be middling, not perfect.
    assert 0.0 <= result.score < 1.0
    assert result.metadata["num_trials"] == 3


def test_consistency_vacuous_with_single_trial():
    result = ConsistencyEvaluator().evaluate_trials([0.8])
    assert result.score == 1.0
    assert result.metadata["num_trials"] == 1


def test_prompt_injection_resistance_on_fixture():
    data = json.loads((FIXTURES / "mock_governance_escalations.json").read_text())
    cases = [
        GovernanceTestCase(
            input_text=c["input_text"],
            is_actually_malicious=c["is_actually_malicious"],
            escalations=[EscalationEvent(**e) for e in c["escalations"]],
        )
        for c in data["cases"]
    ]
    result = PromptInjectionResistanceEvaluator().evaluate_test_set(cases)
    # Fixture is built with 1 TP, 1 FN, 1 TN, 1 FP by design (see fixture
    # comments) -- recall should be exactly 0.5, not a perfect score.
    assert result.metadata["recall"] == 0.5
    assert result.metadata["false_positive_rate"] == 0.5
    assert result.is_success is False  # 0.5 recall is below the 0.9 bar

def test_consistency_flags_erratic_trial_history():
    """Negative control: trial scores bouncing around rather than trending
    anywhere. Should score meaningfully low, not just 'not perfect'."""
    result = ConsistencyEvaluator().evaluate_trials([0.9, 0.1, 0.8, 0.05])
    assert result.score < 0.3
