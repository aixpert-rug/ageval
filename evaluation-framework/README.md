# AIXPERT WP3 Evaluation Framework

Public repository under `aixpert-eu`, sibling to WP4's `agentic-design-patterns`
(D2.3) catalogue. This repo holds **what is measured and how** — dimension
definitions, metric/method specifications and implementations, and KPI
templates. It does **not** hold WP4's agent implementations; those stay
private in D4.1/D4.2. See `contracts/README.md` for exactly where that line
sits and why.

## Structure

```
dimensions/       Level 1 — the 10-dimension taxonomy, as data (not code)
metrics/          Level 2 — metrics & methods, organised by APPLICABILITY,
                  not by dimension (see docs/pattern_metric_matrix.md for why)
contracts/        The WP3/WP4 interface boundary — Evaluator protocol,
                  EvaluationInput/EvaluationResult shapes
kpi_templates/    Level 3 — KPI spec template for WP7, UC-agnostic
fixtures/         Mock agent trajectories, so metrics can be built and
                  tested without a live WP4 agent
tests/            Metric unit tests against the fixtures
docs/             Reference material, incl. the pattern↔metric applicability
                  matrix
```

## Why `metrics/` is organised by applicability, not by dimension

Dimensions (Level 1) are how we *talk about* trustworthiness. But a metric's
applicability is determined by what data a given WP4 pattern actually
produces, not which dimension it happens to serve. `Faithfulness` (an
Explainability metric) and `Task accuracy` (an Accuracy metric) are both
computable from any pattern's final trajectory — but `Convergence rate`
(also nominally about Explainability/Robustness) only exists for patterns
with a trial loop. Organising by dimension would mean duplicating the same
implementation across multiple dimension folders. Organising by applicability
means one implementation, referenced from as many dimensions as apply.

Four applicability categories, derived directly from the WP4 patterns
reviewed so far:

| Folder | Requires | Applies to |
|---|---|---|
| `metrics/common/` | `messages` + final `output` | Any pattern (ReAct, Reflexion, Self-Refine) |
| `metrics/reflective/` | A trial loop (`reflections`, per-trial `eval_result`) | Reflexion, Self-Refine — not ReAct |
| `metrics/trajectory/` | A tool-call trace | ReAct, Reflexion — not Self-Refine |
| `metrics/governance/` | Governance-pattern escalation/policy data | Governance wrappers (e.g. Isolated Agent Self-Defence) — mechanism verification, not agent-behavior scoring |

Each metric module declares which dimension(s) it serves via its `dimensions`
attribute (see `metrics/common/faithfulness.py` for the pattern) — that's
how Level 1 and Level 2 stay linked without forcing a folder-per-dimension
structure.

## Status

Structural scaffold + one worked example per category. Not yet integrated
with a live WP4 agent — see `docs/pattern_metric_matrix.md` and
`contracts/README.md` for the integration plan and current open questions.
