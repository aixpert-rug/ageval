import math

import pytest

from contracts.logprobs import ConfidenceLabel, TokenLogProbs
from metrics.aggregate.judge_elo import JudgeEloEvaluator
from metrics.probabilistic.ece import ECEEvaluator
from metrics.probabilistic.perplexity import PerplexityEvaluator


def test_token_logprobs_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        TokenLogProbs(tokens=["a", "b"], logprobs=[-0.1])


def test_token_logprobs_rejects_positive_logprob():
    with pytest.raises(ValueError):
        TokenLogProbs(tokens=["a"], logprobs=[0.5])


def test_confidence_label_rejects_out_of_range():
    with pytest.raises(ValueError):
        ConfidenceLabel(confidence=1.5, is_correct=True)


def test_perplexity_matches_hand_verified_reference_value():
    tlp = TokenLogProbs(tokens=["a", "b", "c", "d"], logprobs=[math.log(0.5)] * 4)
    result = PerplexityEvaluator().evaluate_logprobs(tlp, baseline_perplexity=2.0, ratio_budget_ceiling=0.5)
    assert result.metadata["perplexity"] == pytest.approx(2.0)
    assert result.metadata["ratio"] == pytest.approx(1.0)


def test_perplexity_worse_than_baseline_penalized():
    tlp = TokenLogProbs(tokens=["a", "b"], logprobs=[math.log(0.1)] * 2)
    result = PerplexityEvaluator().evaluate_logprobs(tlp, baseline_perplexity=2.0, ratio_budget_ceiling=0.5)
    assert result.metadata["perplexity"] == pytest.approx(10.0)
    assert result.metadata["ratio"] == pytest.approx(5.0)
    assert result.is_success is False


def test_perplexity_requires_nonempty_logprobs():
    with pytest.raises(ValueError):
        PerplexityEvaluator().evaluate_logprobs(TokenLogProbs(tokens=[], logprobs=[]), baseline_perplexity=2.0, ratio_budget_ceiling=0.5)


def test_perplexity_rejects_nonpositive_baseline():
    tlp = TokenLogProbs(tokens=["a"], logprobs=[-0.1])
    with pytest.raises(ValueError):
        PerplexityEvaluator().evaluate_logprobs(tlp, baseline_perplexity=0.0, ratio_budget_ceiling=0.5)


def test_ece_matches_hand_verified_reference_value():
    predictions = [ConfidenceLabel(0.9, True)] * 8 + [ConfidenceLabel(0.9, False)] * 2
    result = ECEEvaluator().evaluate_predictions(predictions)
    assert result.metadata["ece"] == pytest.approx(0.1)


def test_ece_perfect_calibration_is_zero():
    predictions = [ConfidenceLabel(1.0, True)] * 10
    result = ECEEvaluator().evaluate_predictions(predictions)
    assert result.metadata["ece"] == pytest.approx(0.0)
    assert result.metadata["band_level"] == 4


def test_ece_uses_vector_documented_default_ceiling():
    predictions = [ConfidenceLabel(0.9, True)] * 8 + [ConfidenceLabel(0.9, False)] * 2
    result = ECEEvaluator().evaluate_predictions(predictions)
    assert result.metadata["band_level"] == 2


def test_ece_requires_nonempty_predictions():
    with pytest.raises(ValueError):
        ECEEvaluator().evaluate_predictions([])


def test_elo_matches_standard_textbook_example():
    ratings = JudgeEloEvaluator().compute_ratings([("agent_a", "agent_b", 1.0)])
    assert ratings["agent_a"] == pytest.approx(1516.0)
    assert ratings["agent_b"] == pytest.approx(1484.0)


def test_elo_tie_leaves_equally_rated_players_unchanged():
    ratings = JudgeEloEvaluator().compute_ratings([("agent_a", "agent_b", 0.5)])
    assert ratings["agent_a"] == pytest.approx(1500.0)
    assert ratings["agent_b"] == pytest.approx(1500.0)


def test_elo_accumulates_across_multiple_comparisons():
    comparisons = [
        ("agent_a", "agent_b", 1.0),
        ("agent_a", "agent_c", 1.0),
        ("agent_b", "agent_c", 1.0),
    ]
    ratings = JudgeEloEvaluator().compute_ratings(comparisons)
    assert ratings["agent_a"] > ratings["agent_b"] > ratings["agent_c"]


def test_elo_evaluate_participant_normalizes_within_set():
    ratings = JudgeEloEvaluator().compute_ratings([("agent_a", "agent_b", 1.0)])
    result_a = JudgeEloEvaluator().evaluate_participant("agent_a", ratings)
    result_b = JudgeEloEvaluator().evaluate_participant("agent_b", ratings)
    assert result_a.score == 1.0
    assert result_b.score == 0.0
    assert result_a.is_success is True
    assert result_b.is_success is False


def test_elo_evaluate_participant_vacuous_with_single_participant():
    ratings = {"agent_a": 1500.0}
    result = JudgeEloEvaluator().evaluate_participant("agent_a", ratings)
    assert result.score == 1.0
    assert result.metadata["n_participants"] == 1


def test_elo_requires_nonempty_comparisons():
    with pytest.raises(ValueError):
        JudgeEloEvaluator().compute_ratings([])


def test_elo_evaluate_participant_rejects_unknown_id():
    ratings = {"agent_a": 1500.0}
    with pytest.raises(ValueError):
        JudgeEloEvaluator().evaluate_participant("agent_z", ratings)
