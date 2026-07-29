"""
Math accuracy — Level 2 metric under Accuracy.

Applicability: common. Compares a numeric/mathematical answer against a
target for mathematical equivalence rather than exact text match (e.g.
"0.5", "1/2", and "50%" should all be treated as equal).

Follows Inspect AI's `math()` scorer: uses SymPy when available for real
equivalence checking (handles fractions, algebra, basic LaTeX); falls back
to plain numeric comparison when SymPy isn't installed, since we don't want
a missing optional dependency to silently disable this metric. The fallback
is weaker (won't recognise "1/2" == "0.5" as equivalent) — this is
documented in the result's metadata so it's visible which path was used,
not silently degraded.

Install `sympy` (`pip install sympy` or the `math` extra on this package)
for full equivalence checking.
"""

from __future__ import annotations

import re

from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


def _extract_number(text: str) -> str | None:
    """Pull the last number-like substring out of free text, stripping
    currency/percent/thousands-separator formatting -- same normalisation
    Inspect's match(numeric=True) applies."""
    cleaned = text.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("%", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", cleaned)
    return matches[-1] if matches else None


def _sympy_equivalent(predicted: str, target: str) -> bool:
    try:
        return bool(sympy.simplify(parse_expr(predicted) - parse_expr(target)) == 0)
    except Exception:
        return False


def _numeric_equivalent(predicted: str, target: str, tolerance: float = 1e-6) -> bool:
    def to_float(s: str) -> float | None:
        if "/" in s:
            num, _, den = s.partition("/")
            try:
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return None
        try:
            return float(s)
        except ValueError:
            return None

    p, t = to_float(predicted), to_float(target)
    if p is None or t is None:
        return predicted.strip() == target.strip()
    return abs(p - t) < tolerance


class MathAccuracyEvaluator(WP3Evaluator):
    """Mathematical equivalence check against a labeled target."""

    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "math_accuracy"

    def evaluate_with_target(self, input: EvaluationInput, target: str) -> EvaluationResult:
        output_text = str(input.output)
        predicted = _extract_number(output_text)
        target_number = _extract_number(target) or target

        if predicted is None:
            return EvaluationResult(
                score=0.0,
                explanation=f"No number found in output: {output_text!r}",
                is_success=False,
                metadata={"sympy_used": False},
            )

        if _SYMPY_AVAILABLE:
            is_equal = _sympy_equivalent(predicted, target_number)
            method = "sympy"
        else:
            is_equal = _numeric_equivalent(predicted, target_number)
            method = "numeric_fallback"

        return EvaluationResult(
            score=1.0 if is_equal else 0.0,
            explanation=(
                f"Predicted {predicted!r} vs. target {target_number!r}: "
                f"{'equivalent' if is_equal else 'not equivalent'} (method={method})."
            ),
            is_success=is_equal,
            metadata={"sympy_used": _SYMPY_AVAILABLE, "method": method},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError(
            "MathAccuracyEvaluator requires a labeled target. "
            "Call evaluate_with_target(input, target) instead."
        )

