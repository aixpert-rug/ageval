"""
Small shared statistics helpers (stdlib only).

bootstrap_ci: percentile bootstrap confidence interval over a list of
per-unit values (e.g. per-problem variances). Resampling is over the
UNITS (problems), not over individual runs, because problems are the
independent sampling units of a test set. Seeded by default so a reported
interval is reproducible.
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Sequence


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `statistic` over `values`.

    With a single value (or all-equal values) the interval is degenerate
    (low == high); that reflects "no between-problem spread observed", not
    certainty, so callers should report n alongside it.
    """
    if not values:
        raise ValueError("bootstrap_ci requires at least one value.")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0,1), got {level}.")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}.")

    rng = random.Random(seed)
    data = list(values)
    n = len(data)
    stats = sorted(statistic([data[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    alpha = (1 - level) / 2
    lo = stats[min(n_resamples - 1, max(0, int(alpha * n_resamples)))]
    hi = stats[min(n_resamples - 1, max(0, int((1 - alpha) * n_resamples) - 1))]
    return lo, hi
