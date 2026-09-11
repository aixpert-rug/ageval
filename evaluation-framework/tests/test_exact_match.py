from contracts.evaluator import EvaluationInput
from metrics.common.exact_match import ExactMatchEvaluator


def _make_input(text: str) -> EvaluationInput:
    return EvaluationInput(messages=[], input="irrelevant", output={"answer": text})


def test_exact_match_identical_strings():
    result = ExactMatchEvaluator().evaluate_with_target(_make_input("Paris"), target="Paris")
    assert result.score == 1.0
    assert result.is_success is True


def test_exact_match_case_and_article_insensitive():
    result = ExactMatchEvaluator().evaluate_with_target(_make_input("The Answer."), target="answer")
    assert result.score == 1.0


def test_exact_match_rejects_partial_overlap():
    result = ExactMatchEvaluator().evaluate_with_target(
        _make_input("Paris, the capital of France"), target="Paris"
    )
    assert result.score == 0.0
    assert result.is_success is False


def test_exact_match_whitespace_normalized():
    result = ExactMatchEvaluator().evaluate_with_target(_make_input("  Paris   "), target="Paris")
    assert result.score == 1.0
