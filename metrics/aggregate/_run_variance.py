"""
Shared multi-run variance aggregation, used by cross_run_consistency and
test_retest_stability (metric updates, Oct 2026).

Method, following the usual multi-trial setup (tau-bench: Yao et al. 2024):
run each problem (one task instance, one item of the test set) several times,
compute ONE value per problem, then average over problems.

  1. per problem: sample variance of its run scores (statistics.variance)
  2. v = mean of the per-problem variances
  3. band v ONCE with the C family, ceiling 0.10:
         band = 4 - floor(4 * v / 0.10), clipped to [0, 4]

Why not band each problem's variance on its own: with 0/1 scores a single
different result among n runs gives a variance of exactly 1/n, so band 4
(variance below 0.025) would only be reachable with 41+ runs per problem.
Averaging gives one band for the whole test set (what the profile needs) with
about 5 runs per problem.

Because an average can hide a few very unstable problems, the summary also
reports the share of problems whose runs were not all identical, plus the most
unstable problems, so reviewers only need to check those. A seeded percentile
bootstrap CI over problems is reported for the mean variance.

Scores must already be normalised to [0, 1]: the 0.10 ceiling is defined on
that scale (F1 is already there; judge scores on other scales must be rescaled
by the caller first).
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from contracts.banding import lower_is_better_continuous_band
from contracts.stats import bootstrap_ci

BUDGET_CEILING = 0.10
MIN_RUNS = 2
RECOMMENDED_RUNS = 5
MOST_UNSTABLE_N = 5
_TOL = 1e-4  # same tolerance as the banding module's unit-interval clamp


def normalise_runs(runs, what: str = "runs") -> list[list[float]]:
    """Accept list[list[float]] (one inner list per problem). A flat list of
    numbers is treated as the runs of a single problem."""
    if not runs:
        raise ValueError(f"At least one problem with {what} is required.")
    if all(isinstance(r, (int, float)) and not isinstance(r, bool) for r in runs):
        runs = [list(runs)]
    out = [list(r) for r in runs]
    for i, scores in enumerate(out):
        if len(scores) < MIN_RUNS:
            raise ValueError(f"Problem {i} has {len(scores)} {what}; at least {MIN_RUNS} are required to compute a variance (5 recommended).")
        for s in scores:
            if not isinstance(s, (int, float)) or not math.isfinite(s):
                raise ValueError(f"Problem {i} has a non-numeric or non-finite score: {s!r}.")
            if s < -_TOL or s > 1 + _TOL:
                raise ValueError(f"Problem {i} has score {s}, outside [0, 1]. Normalise scores to [0, 1] first (the 0.10 variance ceiling assumes that scale).")
    return out


def summarise_run_variances(runs: Sequence[Sequence[float]], problem_ids: Sequence | None = None) -> dict:
    runs = normalise_runs(runs)
    if problem_ids is not None and len(problem_ids) != len(runs):
        raise ValueError(f"problem_ids has {len(problem_ids)} entries but there are {len(runs)} problems.")
    ids = list(problem_ids) if problem_ids is not None else list(range(len(runs)))

    variances = [statistics.variance(r) for r in runs]
    mean_var = statistics.fmean(variances)
    band = lower_is_better_continuous_band(mean_var, budget_ceiling=BUDGET_CEILING)
    unstable = [i for i, r in enumerate(runs) if max(r) != min(r)]
    ranked = sorted(unstable, key=lambda i: variances[i], reverse=True)[:MOST_UNSTABLE_N]
    ci_lo, ci_hi = bootstrap_ci(variances)
    n_runs = [len(r) for r in runs]

    return {
        "band": band,
        "mean_variance": mean_var,
        "per_problem_variances": variances,
        "n_problems": len(runs),
        "runs_per_problem": n_runs,
        "min_runs": min(n_runs),
        "below_recommended_runs": min(n_runs) < RECOMMENDED_RUNS,
        "n_unstable": len(unstable),
        "unstable_share": len(unstable) / len(runs),
        "most_unstable": [{"problem": ids[i], "variance": variances[i]} for i in ranked],
        "variance_ci_95": (ci_lo, ci_hi),
        "budget_ceiling": BUDGET_CEILING,
    }


def describe(s: dict, noun: str = "problems") -> str:
    text = (
        f"Mean per-problem score variance {s['mean_variance']:.4f} over {s['n_problems']} {noun} "
        f"(min {s['min_runs']} runs each) -- band {s['band'].level} ({s['band'].label}). "
        f"{100 * s['unstable_share']:.0f}% of {noun} were not fully stable. "
        f"95% bootstrap CI over {noun}: [{s['variance_ci_95'][0]:.4f}, {s['variance_ci_95'][1]:.4f}]."
    )
    if s["below_recommended_runs"]:
        text += f" Note: fewer than {RECOMMENDED_RUNS} runs for some {noun}; the estimate is noisier."
    return text
