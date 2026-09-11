import pytest

from contracts.evaluator import EvaluationInput
from metrics.common.bleu import BLEUEvaluator
from metrics.common.rouge import ROUGEEvaluator


def _make_input(text: str) -> EvaluationInput:
    return EvaluationInput(messages=[], input="irrelevant", output={"answer": text})


# --- BLEU ---

def test_bleu_perfect_match():
    result = BLEUEvaluator().evaluate_with_target(
        _make_input("the cat sat on the mat"), target="the cat sat on the mat"
    )
    assert result.metadata["bleu_0_100"] == pytest.approx(100.0, abs=0.01)
    assert result.is_success is True


def test_bleu_no_overlap():
    result = BLEUEvaluator().evaluate_with_target(
        _make_input("completely different text here"), target="the cat sat on the mat"
    )
    assert result.metadata["bleu_0_100"] == 0.0
    assert result.is_success is False


def test_bleu_empty_hypothesis():
    result = BLEUEvaluator().evaluate_with_target(_make_input(""), target="the cat sat on the mat")
    assert result.score == 0.0


# --- ROUGE-L ---

def test_rouge_l_perfect_match():
    result = ROUGEEvaluator().evaluate_with_target(
        _make_input("the cat sat on the mat"), target="the cat sat on the mat"
    )
    assert result.metadata["fmeasure"] == pytest.approx(1.0)
    assert result.is_success is True


def test_rouge_l_no_overlap():
    result = ROUGEEvaluator().evaluate_with_target(
        _make_input("zzz yyy xxx"), target="the cat sat on the mat"
    )
    assert result.metadata["fmeasure"] == 0.0


def test_rouge_l_partial_overlap_scores_between_bounds():
    result = ROUGEEvaluator().evaluate_with_target(
        _make_input("the cat sat"), target="the cat sat on the mat"
    )
    assert 0.0 < result.metadata["fmeasure"] < 1.0


# --- METEOR (environment-conditional: WordNet corpus may not be available) ---

def _wordnet_available() -> bool:
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        return False


@pytest.mark.skipif(_wordnet_available(), reason="WordNet IS available -- see test_meteor_scores_synonym_pair_when_available instead")
def test_meteor_raises_clear_error_when_wordnet_missing():
    from metrics.common.meteor import METEOREvaluator
    with pytest.raises(RuntimeError, match="WordNet"):
        METEOREvaluator()


@pytest.mark.skipif(not _wordnet_available(), reason="WordNet not available in this environment -- run nltk.download('wordnet') first")
def test_meteor_scores_synonym_pair_when_available():
    from metrics.common.meteor import METEOREvaluator
    result = METEOREvaluator().evaluate_with_target(_make_input("the cat is glad"), target="the cat is happy")
    assert result.score > 0.3
