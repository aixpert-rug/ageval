"""
Judge win-rate / Elo -- Level 2 metric under Accuracy 

Applicability: aggregate. Needs a set of PAIRWISE comparison outcomes
(which of two agent responses a judge preferred) -- this evaluator
aggregates them into Elo ratings, it does NOT call a judge itself.
Generating the raw pairwise outcomes needs a live LLM-as-judge (or a
human rater) -- deliberately out of scope here, same "someone else must
supply the raw signal" pattern as task_success's oracle_fn or
energy_carbon's power draw. What's implemented is the well-defined
aggregation math our document specifies as a formula.

Formula (standard Elo rating system):
    E_i = 1 / (1 + 10^((R_j - R_i)/400))    -- expected score for i vs j
    R_i <- R_i + K * (S_i - E_i)             -- rating update
Verified against the standard textbook example before use: two
equally-rated players (1500) with i beating j, K=32 -> winner's new
rating 1516, loser's new rating 1484 exactly.

IMPORTANT limitation, same root cause as Trajectory Optimality: The document tags this metric with the "Comparative" family 
quartile rank within a VERSIONED, SHARED CONSORTIUM POOL. This repo
doesn't maintain that pool, so `evaluate_participant()` below can only
report a rating NORMALISED RELATIVE TO THE OTHER PARTICIPANTS IN THE
SAME COMPARISON SET, not a calibrated cross-consortium score. Treat the
returned score as "how did this participant do relative to the others
actually compared here", not an absolute or consortium-comparable
judgement.
"""

from __future__ import annotations

from contracts.evaluator import EvaluationResult, WP3Evaluator


class JudgeEloEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "aggregate"
    metric_id = "judge_elo"

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def compute_ratings(self, comparisons: list[tuple[str, str, float]]) -> dict[str, float]:
        """
        Pure Elo aggregation across a set of pairwise outcomes.

        Args:
            comparisons: list of (participant_a_id, participant_b_id,
                score_a) tuples, where score_a is 1.0 (a won), 0.0
                (a lost), or 0.5 (tie) -- these outcomes must already
                exist, produced by a judge you ran separately.

        Returns:
            dict mapping participant_id -> final Elo rating.
        """
        if not comparisons:
            raise ValueError("compute_ratings requires at least one comparison.")

        ratings: dict[str, float] = {}

        def get(pid: str) -> float:
            return ratings.setdefault(pid, self.initial_rating)

        for a, b, score_a in comparisons:
            if not (0.0 <= score_a <= 1.0):
                raise ValueError(f"score_a must be in [0,1] (0=a lost, 1=a won, 0.5=tie), got {score_a}.")
            ra, rb = get(a), get(b)
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            eb = 1 - ea
            ratings[a] = ra + self.k_factor * (score_a - ea)
            ratings[b] = rb + self.k_factor * ((1 - score_a) - eb)

        return ratings

    def evaluate_participant(self, participant_id: str, ratings: dict[str, float]) -> EvaluationResult:
        """
        Args:
            participant_id: whose result to report.
            ratings: output of compute_ratings() -- pass the SAME dict
                computed across the full comparison set, not just this
                one participant's comparisons, so the relative
                normalisation below is meaningful.
        """
        if participant_id not in ratings:
            raise ValueError(f"participant_id {participant_id!r} not found in ratings.")

        if len(ratings) < 2:
            return EvaluationResult(
                score=1.0,
                explanation="Only one participant in the comparison set; nothing to compare against.",
                is_success=True,
                metadata={"raw_rating": ratings[participant_id], "all_ratings": dict(ratings), "n_participants": 1},
            )

        min_r, max_r = min(ratings.values()), max(ratings.values())
        normalized = 0.5 if max_r == min_r else (ratings[participant_id] - min_r) / (max_r - min_r)

        return EvaluationResult(
            score=normalized,
            explanation=(
                f"Rating {ratings[participant_id]:.1f}, normalised to {normalized:.3f} relative to "
                f"{len(ratings)} participants in this comparison set (NOT a consortium-calibrated score "
                f"-- see module docstring)."
            ),
            is_success=normalized >= 0.5,
            metadata={"raw_rating": ratings[participant_id], "all_ratings": dict(ratings), "n_participants": len(ratings)},
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("JudgeEloEvaluator operates on pairwise comparison outcomes, not a single EvaluationInput. Call compute_ratings(...) then evaluate_participant(...) instead.")
