"""
Cost per query -- Level 2 metric under Efficiency .

Applicability: efficiency. Cost isn't something you MEASURE live the way
latency or memory are -- it's CALCULATED from token counts and a price
table. So instead of a measure_and_evaluate() timing wrapper, this adds
compute_and_evaluate(): given input/output token counts and a price
table, it does the calculation itself rather than requiring the caller
to pre-compute cost externally.

evaluate_measurement() still accepts an already-computed cost value too
(e.g. pulled directly from a provider's billing API, which may include
fees/discounts this repo has no way to replicate).
"""

from __future__ import annotations

from contracts.banding import ratio_to_budget_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class CostPerQueryEvaluator(WP3Evaluator):
    dimensions = ("Efficiency",)
    applicability = "efficiency"
    metric_id = "cost_per_query"

    def evaluate_measurement(self, cost: float, budget: float, currency: str = "USD") -> EvaluationResult:
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}.")

        band = ratio_to_budget_band(cost, budget, lower_is_better=True)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=(
                f"{cost:.4f} {currency} against a {budget:.4f} {currency} budget "
                f"-- band {band.level} ({band.label})."
            ),
            is_success=band.level >= 2,
            metadata={"cost": cost, "budget": budget, "currency": currency, "band_level": band.level, "band_label": band.label},
        )

    def compute_and_evaluate(
        self,
        input_tokens: int,
        output_tokens: int,
        price_per_1k_input: float,
        price_per_1k_output: float,
        budget: float,
        currency: str = "USD",
    ) -> EvaluationResult:
        """
        Args:
            input_tokens, output_tokens: token counts for this call.
            price_per_1k_input, price_per_1k_output: provider's pricing,
                per 1,000 tokens -- the standard unit most providers
                quote pricing in. Look these up from the provider's own
                published pricing, not hardcoded here since prices
                change and vary by model/provider.
            budget, currency: same as evaluate_measurement.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative.")

        cost = (input_tokens / 1000) * price_per_1k_input + (output_tokens / 1000) * price_per_1k_output
        result = self.evaluate_measurement(cost=cost, budget=budget, currency=currency)
        result.metadata["input_tokens"] = input_tokens
        result.metadata["output_tokens"] = output_tokens
        result.metadata["price_per_1k_input"] = price_per_1k_input
        result.metadata["price_per_1k_output"] = price_per_1k_output
        return result

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError("CostPerQueryEvaluator operates on a pre-measured value or token counts + pricing. Call evaluate_measurement(...) or compute_and_evaluate(...) instead.")
