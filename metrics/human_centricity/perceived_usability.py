"""
Perceived Usability (System Usability Scale) -- Level 2 metric under
Human-centricity.

Applicability: human_centricity -- a new category, distinct from the
other five. Unlike every other metric in this repo, this one does NOT
operate on an agent trajectory (EvaluationInput) at all. It operates on
survey responses collected from human participants after they interact
with the system. Requires participant recruitment and, depending on
jurisdiction, ethics review -- see UC1 correspondence for a concrete
example of where this data collection would actually happen.

Implements the System Usability Scale (SUS):
  - Brooke, "SUS: A Quick and Dirty Usability Scale" (1996) -- original
    instrument and scoring algorithm.
  - Bangor, Kortum & Miller, "An Empirical Evaluation of the System
    Usability Scale" (2008) -- adjective-rating interpretation bands
    used below; treat the band boundaries as commonly-cited approximate
    guidance, not exact cutoffs.

SUS is a fixed 10-item questionnaire, alternating positively- and
negatively-worded statements, each rated on a 1-5 Likert scale. Do not
substitute a different item wording or count -- the standard scoring
algorithm depends on the specific 10-item structure.
"""

from __future__ import annotations

from contracts.evaluator import EvaluationResult, WP3Evaluator

#: The 10 standard SUS items, in order, exactly as Brooke (1996) defined
#: them. Odd (1-indexed) items are positively worded; even are negatively
#: worded -- this alternation is what the scoring algorithm depends on.
SUS_ITEMS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]


def _adjective_band(score: float) -> str:
    """Approximate interpretation band per Bangor, Kortum & Miller (2008)."""
    if score >= 80.3:
        return "excellent"
    if score >= 68:
        return "good"
    if score >= 51:
        return "ok"
    return "poor"


class PerceivedUsabilityEvaluator(WP3Evaluator):
    """SUS score from one participant's 10 item responses (1-5 each)."""

    dimensions = ("Human-centricity",)
    applicability = "human_centricity"
    metric_id = "perceived_usability"

    def evaluate_responses(self, responses: list[int]) -> EvaluationResult:
        """
        Args:
            responses: exactly 10 integers, 1-5, one per SUS_ITEMS entry
                in order, from a single participant.
        """
        if len(responses) != 10:
            raise ValueError(f"SUS requires exactly 10 responses, got {len(responses)}.")
        if any(r < 1 or r > 5 for r in responses):
            raise ValueError("SUS responses must each be on a 1-5 Likert scale.")

        # Odd items (0-indexed: 0,2,4,6,8) are positively worded: score = response - 1.
        # Even items (0-indexed: 1,3,5,7,9) are negatively worded: score = 5 - response.
        adjusted = [
            (responses[i] - 1) if i % 2 == 0 else (5 - responses[i])
            for i in range(10)
        ]
        raw_score = sum(adjusted)  # 0-40
        sus_score = raw_score * 2.5  # 0-100

        return EvaluationResult(
            score=sus_score / 100.0,  # normalised to this repo's 0-1 convention
            explanation=(
                f"SUS score: {sus_score:.1f}/100 ({_adjective_band(sus_score)}, "
                f"per Bangor et al. 2008 bands)."
            ),
            is_success=sus_score >= 68,  # "good" or better
            metadata={"sus_score_0_100": sus_score, "band": _adjective_band(sus_score)},
        )

    def evaluate_batch(self, participant_responses: list[list[int]]) -> EvaluationResult:
        """Mean SUS across multiple participants -- the more common reporting unit."""
        if not participant_responses:
            raise ValueError("evaluate_batch requires at least one participant.")
        individual = [self.evaluate_responses(r).metadata["sus_score_0_100"] for r in participant_responses]
        mean_score = sum(individual) / len(individual)
        return EvaluationResult(
            score=mean_score / 100.0,
            explanation=(
                f"Mean SUS across {len(individual)} participants: {mean_score:.1f}/100 "
                f"({_adjective_band(mean_score)})."
            ),
            is_success=mean_score >= 68,
            metadata={"n_participants": len(individual), "individual_scores": individual, "mean_sus_score_0_100": mean_score},
        )

    def evaluate(self, input) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "PerceivedUsabilityEvaluator operates on survey responses, not an "
            "agent trajectory. Call evaluate_responses(responses) or evaluate_batch(...) instead."
        )
