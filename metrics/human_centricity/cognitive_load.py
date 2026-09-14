"""
Cognitive Load (NASA Task Load Index) -- Level 2 metric under
Human-centricity.

Applicability: human_centricity. Same data-collection profile as
perceived_usability.py -- survey-based, requires human participants, not
computed from an agent trajectory.

Implements the Raw TLX (unweighted-mean variant):
  - Hart & Staveland, "Development of NASA-TLX (Task Load Index):
    Results of Empirical and Theoretical Research" (1988) -- original
    instrument, 6 subscales, plus the pairwise-comparison weighting
    procedure for the full weighted version.
  - Hart, "NASA-TLX 20 Years Later" (2006) -- shows the unweighted mean
    ("Raw TLX") correlates highly with the full weighted procedure while
    being much simpler to administer; used as the default here for that
    reason. Pass explicit weights if the full weighted procedure's
    pairwise-comparison data is available.

Direction note, easy to get backwards: NASA-TLX's "Performance" subscale
is traditionally rated from 0 ("Perfect") to 100 ("Failure") -- i.e. on
the ORIGINAL scale, a LOW performance rating means the participant felt
they did well. That is the opposite direction from the other five
subscales, where a low rating means low demand/effort/frustration (good).
This implementation assumes the caller has already reverse-coded
Performance onto the same "low = good" direction as the other subscales
before passing it in (i.e. performance_rating = 100 - raw_perfect_to_failure_rating).
This is flagged explicitly here because silently getting this backwards
is the single most common NASA-TLX scoring error.
"""

from __future__ import annotations

from contracts.evaluator import EvaluationResult, WP3Evaluator

SUBSCALES = ["mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration"]


class CognitiveLoadEvaluator(WP3Evaluator):
    """Raw (unweighted) NASA-TLX workload score from one participant's ratings.

    score: workload index, 0-100 native scale (higher = more workload =
        worse). Reported inverted (1 - workload/100) in the returned
        EvaluationResult.score field, to stay consistent with this
        repo's higher-is-better convention -- the raw 0-100 workload
        figure is preserved in metadata for anyone who needs the
        native-direction number.
    """

    dimensions = ("Human-centricity",)
    applicability = "human_centricity"
    metric_id = "cognitive_load"

    def evaluate_ratings(self, ratings: dict[str, float], weights: dict[str, float] | None = None) -> EvaluationResult:
        """
        Args:
            ratings: dict with all six SUBSCALES as keys, each 0-100.
                `performance` must already be reverse-coded onto the
                "low = good" direction -- see module docstring.
            weights: optional, dict with the same six keys, summing to 1.0,
                from the standard NASA-TLX pairwise-comparison procedure.
                If omitted, uses the unweighted Raw TLX mean (Hart, 2006).
        """
        missing = set(SUBSCALES) - set(ratings)
        if missing:
            raise ValueError(f"Missing NASA-TLX subscale ratings: {sorted(missing)}")
        if any(not (0 <= v <= 100) for v in ratings.values()):
            raise ValueError("NASA-TLX subscale ratings must each be 0-100.")

        if weights is None:
            workload = sum(ratings[s] for s in SUBSCALES) / len(SUBSCALES)
            method = "raw_tlx_unweighted"
        else:
            missing_w = set(SUBSCALES) - set(weights)
            if missing_w:
                raise ValueError(f"Missing NASA-TLX subscale weights: {sorted(missing_w)}")
            if abs(sum(weights.values()) - 1.0) > 1e-6:
                raise ValueError(f"NASA-TLX weights must sum to 1.0, got {sum(weights.values())}.")
            workload = sum(ratings[s] * weights[s] for s in SUBSCALES)
            method = "weighted_tlx"

        return EvaluationResult(
            score=1 - (workload / 100.0),
            explanation=f"NASA-TLX workload index: {workload:.1f}/100 (method={method}; lower is better).",
            is_success=workload <= 50.0,  # no strong literature-standard cutoff; flagged as a default, not a validated threshold
            metadata={"workload_0_100": workload, "method": method, "subscale_ratings": ratings},
        )

    def evaluate(self, input) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "CognitiveLoadEvaluator operates on survey ratings, not an agent "
            "trajectory. Call evaluate_ratings(ratings) instead."
        )
