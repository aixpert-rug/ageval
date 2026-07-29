# AIXPERT WP3 Evaluation Framework

Public repository under the AIXPERT project, representing WP3's auditing and evaluation framework for agentic AI. 
This repo holds **what is measured and how** — dimension
definitions, metric/method specifications and implementations, and KPI
templates. It does not hold WP4's agent implementations. 
See `contracts/README.md` for more information.

## Structure

```
dimensions/       Level 1 — the 10-dimension taxonomy, as data (not code)
metrics/          Level 2 — metrics & methods, organised by APPLICABILITY,
                  not by dimension (see docs/pattern_metric_matrix.md for why)
contracts/        The interface boundary — Evaluator protocol,
                  EvaluationInput/EvaluationResult shapes
kpi_templates/    Level 3 — KPI spec template for WP7, UC-agnostic
fixtures/         Mock agent trajectories, so metrics can be built and
                  tested without a live agent
tests/            Metric unit tests against the fixtures
docs/             Reference material, incl. the pattern↔metric applicability
                  matrix
```

## Why `metrics/` is organised by applicability, not by dimension


Dimensions (Level 1) are how we *talk about* trustworthiness. But a metric's
applicability is determined by what data a given agent pattern actually
produces, not which dimension it happens to serve. `Faithfulness` (an
Explainability metric) and `Task accuracy` (an Accuracy metric) are both
computable from any pattern's final trajectory — but `Convergence rate`
(also nominally about Explainability/Robustness) only exists for patterns
with a trial loop. Organising by dimension would mean duplicating the same
implementation across multiple dimension folders. Organising by applicability
means one implementation, referenced from as many dimensions as apply.

Five applicability categories, derived from reviewing several agent pattern
architectures (a single-pass reasoning/tool-use loop, two trial-based
self-correction loops — one with tool use, one without — and a governance/
security wrapper), plus one category adapted from external evaluation
tooling (see below):

| Folder | Requires | Applies to |
|---|---|---|
| `metrics/common/` | `messages` + final `output` | Any pattern (single-pass or trial-based) |
| `metrics/reflective/` | A trial loop (per-trial score/success signal) | Trial-based self-correction patterns only |
| `metrics/trajectory/` | A tool-call trace | Tool-using patterns only |
| `metrics/governance/` | Governance-pattern escalation/policy data | Governance wrappers — mechanism verification, not agent-behavior scoring |
| `metrics/scanners/` | A full trajectory, checked for a pattern's presence rather than scored for task success | Any pattern; a scanner may report nothing at all on a clean trajectory |

Each metric module declares which dimension(s) it serves via its `dimensions`
attribute (see `metrics/common/faithfulness.py` for the pattern) — that's
how Level 1 and Level 2 stay linked without forcing a folder-per-dimension
structure.

## Implemented metrics

| metric_id | File | Category | Dimension(s) |
|---|---|---|---|
| `faithfulness` | `metrics/common/faithfulness.py` | common | Explainability |
| `task_accuracy` | `metrics/common/task_accuracy.py` | common | Accuracy |
| `math_accuracy` | `metrics/common/math_accuracy.py` | common | Accuracy |
| `trial_consistency` | `metrics/reflective/consistency.py` | reflective | Robustness |
| `convergence_rate` | `metrics/reflective/convergence_rate.py` | reflective | Robustness |
| `score_monotonicity` | `metrics/reflective/score_monotonicity.py` | reflective | Robustness |
| `log_completeness` | `metrics/trajectory/log_completeness.py` | trajectory | Transparency, Auditability |
| `prompt_injection_resistance` | `metrics/governance/prompt_injection_resistance.py` | governance | Safety, Security & Privacy |
| `refusal` | `metrics/scanners/refusal.py` | scanners | Robustness |
| `reward_hacking` | `metrics/scanners/reward_hacking.py` | scanners | Accuracy, Robustness |

See `docs/pattern_metric_matrix.md` for the full applicability breakdown,
including metrics that are specified but not yet implemented.

## Where the `scanners` category and some metric designs came from

Several scorers and the whole `scanners` category are adapted from a
well-known open-source LLM evaluation framework's public documentation
(scorer/scanner design patterns: F1-based accuracy, mathematical-equivalence
checking, deterministic pattern scanning, and transcript-wide behavioral
scanning distinct from per-sample scoring). Adapted, not imported — each
implementation here is written from scratch against our own contracts, not
a dependency on that framework's package.

## Status

Structural scaffold + worked examples across all five categories, each with
both positive and negative-control tests. Reflection-groundedness and the
remaining governance-side metrics (governance coverage, escalation
completeness, policy enforcement correctness) are specified but not yet
implemented — see `docs/pattern_metric_matrix.md`. One real (non-mock) agent
trajectory has been validated against; a second, forcing genuine tool use,
is still needed to confirm the metrics also recognise *good* grounding on
real output, not just bad.
