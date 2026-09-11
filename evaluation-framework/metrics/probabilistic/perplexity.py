"""
Perplexity -- Level 2 metric under Accuracy.

Applicability: probabilistic -- a distinct category from all
others. Needs token log-probabilities (contracts.logprobs.TokenLogProbs),
not a trajectory, survey, or pre-measured external value. Generating
log-probabilities is the caller's responsibility (via an OpenAI-
compatible API's logprobs parameter, or direct open-weight model access)
-- this evaluator only does the math once they're supplied.

Formula: PPL = exp(-(1/T) * sum(log
p(x_t | x<t))). Verified against a known reference case before use: if a
model assigns exactly 0.5 probability to every actual token, perplexity
is exactly 2.0 (the "effective branching factor" interpretation).

Banding: per Table 5, Perplexity uses the "Lower-is-better continuous"
family. However, both `baseline_perplexity` and
`ratio_budget_ceiling` are therefore REQUIRED arguments here, not
defaulted - inventing a ceiling would be a specific, undocumented guess
presented with false authority.
"""

from __future__ import annotations

import math

from contracts.banding import lower_is_better_continuous_band
from contracts.evaluator import EvaluationResult, WP3Evaluator
from contracts.logprobs import TokenLogProbs


class PerplexityEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "probabilistic"
    metric_id = "perplexity"

    def evaluate_logprobs(
        self,
        token_logprobs: TokenLogProbs,
        baseline_perplexity: float,
        ratio_budget_ceiling: float,
    ) -> EvaluationResult:
        """
        Args:
            token_logprobs: the generation's per-token log-probabilities.
            baseline_perplexity: a reference/base model's perplexity on
                the SAME corpus with the SAME tokenizer -- per our
                own normalisation requirement. Not this evaluator's job
                to compute; supply it from a separate baseline run.
            ratio_budget_ceiling: budget ceiling for the ratio
                (this_ppl / baseline_ppl) under the lower-is-better
                continuous family. REQUIRED -- see module docstring.
        """
        if not token_logprobs.logprobs:
            raise ValueError("token_logprobs must contain at least one token.")
        if baseline_perplexity <= 0:
            raise ValueError(f"baseline_perplexity must be positive, got {baseline_perplexity}.")

        mean_logprob = sum(token_logprobs.logprobs) / len(token_logprobs.logprobs)
        perplexity = math.exp(-mean_logprob)
        ratio = perplexity / baseline_perplexity

        band = lower_is_better_continuous_band(ratio, budget_ceiling=ratio_budget_ceiling)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=(
                f"Perplexity {perplexity:.3f} vs. baseline {baseline_perplexity:.3f} "
                f"(ratio {ratio:.3f}) -- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={
                "perplexity": perplexity, "baseline_perplexity": baseline_perplexity, "ratio": ratio,
                "n_tokens": len(token_logprobs.logprobs), "band_level": band.level, "band_label": band.label,
            },
        )

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("PerplexityEvaluator operates on TokenLogProbs, not a single EvaluationInput. Call evaluate_logprobs(...) instead.")
