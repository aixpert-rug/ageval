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


def _answer_text(output) -> str:
    """Text of the answer: for a dict, every str/int/float field (a numeric field such as
    {"answer": 42} must not be dropped); otherwise str(output)."""
    if isinstance(output, dict):
        return " ".join(str(v) for v in output.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
    return str(output)


def _extract_number(text: str) -> str | None:
    """Last number-like token in free text, e.g. "50%", "1/2", "-3.5". Currency symbols and
    thousands separators are removed; a trailing "%" is KEPT so the caller can interpret it.
    A hyphen counts as a minus sign only when it is not glued to a preceding word or digit."""
    cleaned = re.sub(r"\s+%", "%", text.replace(",", "").replace("$", "").replace("€", "").replace("£", ""))
    matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?%?|\d+(?:\.\d+)?(?:/\d+)?%?", cleaned)
    return matches[-1] if matches else None


def is_numeric_target(task_def) -> bool:
    """True when the task's target is a single number ("42", "1/2", "50%", "$1,000"). Used by the
    registry so the harness runs math_accuracy only on numeric tasks."""
    target = getattr(task_def, "target", None)
    if target is None:
        return False
    cleaned = re.sub(r"\s+%", "%", str(target).replace(",", "").replace("$", "").replace("€", "").replace("£", "")).strip()
    return re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?%?", cleaned) is not None


def _candidates(number: str) -> list[str]:
    """"50%" may mean 50 or 0.5, so both readings are accepted."""
    if number.endswith("%"):
        base = number[:-1]
        return [base, f"({base})/100"]
    return [number]


def _sympy_equivalent(predicted: str, target: str) -> bool:
    try:
        return abs(float(sympy.N(parse_expr(predicted) - parse_expr(target)))) < 1e-9
    except Exception:
        return False


def _numeric_equivalent(predicted: str, target: str, tolerance: float = 1e-6) -> bool:
    def to_float(s: str) -> float | None:
        s = s.strip()
        if s.startswith("(") and ")/" in s:
            inner, _, den = s[1:].partition(")/")
            s = f"{inner}/{den}"
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
        output_text = _answer_text(input.output)
        predicted = _extract_number(output_text)
        target_number = _extract_number(target) or target

        if predicted is None:
            return EvaluationResult(
                score=0.0,
                explanation=f"No number found in output: {output_text!r}",
                is_success=False,
                metadata={"sympy_used": False},
            )

        equivalent = _sympy_equivalent if _SYMPY_AVAILABLE else _numeric_equivalent
        method = "sympy" if _SYMPY_AVAILABLE else "numeric_fallback"
        is_equal = any(equivalent(p, t) for p in _candidates(predicted) for t in _candidates(target_number))

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

