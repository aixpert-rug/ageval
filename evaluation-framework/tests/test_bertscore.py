"""
Test strategy for BERTScoreEvaluator:

1. Lazy-loading and empty-input behavior: tested for real, no network
   needed -- these never touch the model at all.
2. The OSError -> clear RuntimeError conversion: tested via MOCKING
   BERTScorer's constructor to raise OSError directly, rather than
   relying on the test environment actually lacking network access.
   That real-environment-dependent approach was tried first and broke
   twice: once when a network-restricted sandbox got confused with a
   network-having one, and again when floating-point precision from a
   REAL successful score (once network access was confirmed working)
   interacted with a too-tight epsilon elsewhere. Mocking the failure
   directly avoids depending on real environment network state at all.
3. The actual scoring computation logic: tested with a mocked scorer,
   verified independent of whether a real model can load.
4. A real, opt-in (@pytest.mark.integration) end-to-end test, confirmed
   passing manually against real infrastructure (distilbert-base-uncased,
   F1=0.940 for a genuine near-paraphrase pair) -- run explicitly via
   `pytest -m integration`, not part of the default test run, since it
   downloads a real model and is slow.
"""

from unittest.mock import MagicMock, patch

import pytest

from contracts.evaluator import EvaluationInput
from metrics.common.bertscore import BERTScoreEvaluator


def _make_input(text: str) -> EvaluationInput:
    return EvaluationInput(messages=[], input="irrelevant", output={"answer": text})


def test_construction_does_not_trigger_model_load():
    evaluator = BERTScoreEvaluator()
    assert evaluator._scorer is None


def test_empty_hypothesis_short_circuits_before_model_load():
    evaluator = BERTScoreEvaluator()
    result = evaluator.evaluate_with_target(_make_input(""), target="the cat sat on the mat")
    assert result.score == 0.0
    assert evaluator._scorer is None


def test_oserror_during_model_load_raises_clear_runtime_error():
    """Mocked, not real: simulates the huggingface.co-unreachable case
    deterministically, regardless of whether THIS environment actually
    has network access to test that for real."""
    evaluator = BERTScoreEvaluator()
    with patch("bert_score.BERTScorer", side_effect=OSError("could not connect to huggingface.co")):
        with pytest.raises(RuntimeError, match="huggingface.co|network"):
            evaluator.evaluate_with_target(_make_input("test"), target="test")


def test_scoring_logic_with_mocked_scorer():
    mock_scorer = MagicMock()
    mock_precision = MagicMock()
    mock_precision.item.return_value = 0.9
    mock_recall = MagicMock()
    mock_recall.item.return_value = 0.8
    mock_f1 = MagicMock()
    mock_f1.item.return_value = 0.85
    mock_scorer.score.return_value = (mock_precision, mock_recall, mock_f1)

    evaluator = BERTScoreEvaluator()
    evaluator._scorer = mock_scorer

    result = evaluator.evaluate_with_target(_make_input("the cat sat"), target="the cat sat on the mat")

    assert result.score == 0.85
    assert result.metadata["precision"] == 0.9
    assert result.metadata["recall"] == 0.8
    assert result.metadata["band_level"] == 4
    assert result.is_success is True
    mock_scorer.score.assert_called_once_with(["the cat sat"], ["the cat sat on the mat"])


def test_scoring_logic_below_band_2_fails():
    mock_scorer = MagicMock()
    low_score = MagicMock()
    low_score.item.return_value = 0.1
    mock_scorer.score.return_value = (low_score, low_score, low_score)

    evaluator = BERTScoreEvaluator()
    evaluator._scorer = mock_scorer

    result = evaluator.evaluate_with_target(_make_input("unrelated text"), target="the cat sat on the mat")
    assert result.metadata["band_level"] == 0
    assert result.is_success is False


@pytest.mark.integration
def test_real_scoring_end_to_end():
    """Opt-in only (pytest -m integration): downloads a real model,
    slow and network-dependent. Confirmed working manually on real
    infrastructure (F1=0.940 for this exact near-paraphrase pair);
    this pins that result as a regression test rather than a one-off
    manual check."""
    evaluator = BERTScoreEvaluator()
    result = evaluator.evaluate_with_target(
        _make_input("the cat sat on the mat"), target="a cat was sitting on the mat"
    )
    assert result.score > 0.85
    assert result.is_success is True
