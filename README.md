# AIXPERT WP3 Evaluation Framework

Public repository under `aixpert-eu`, sibling to the AIXPERT agentic patterns catalogue. This repo holds **what is measured and how** —
dimension definitions, metric/method specifications and
implementations, KPI templates, and the harness that connects them to a
real agent run. It does **not** hold agent implementations.

## Status: confirmed working end-to-end against a real, live agent

This isn't just a metrics library tested in isolation. As of this
release, the full pipeline has been run live: a real WP4 pattern
(single-pass, ReAct-style, tool-using) executed for real against a live
LLM provider, its output adapted directly into this framework's
contract, and scored by ten metrics automatically selected from the
registry below: all in one process, no manual data wrangling, no
JSON round-trip required (though that path also works, and is useful on
its own -- see `harness/`)

## Structure
dimensions/ Level 1 -- the 10-dimension taxonomy, as data (not code)
metrics/ Level 2 -- 47 metrics across 10 applicability categories
contracts/ The interface boundary: shared data contracts,
banding logic, and the metric registry (see below)
harness/ The integration layer: adapts an agentic pattern's raw
output into this framework's contract, and runs every
applicable metric against it automatically
kpi_templates/ Level 3 -- KPI spec template for WP7, UC-agnostic
fixtures/ Mock and real agent trajectories
tests/ Metric and harness unit tests -- 201 passing
docs/ Reference material, incl. the pattern <-> metric
applicability matrix and the full periodic-table cross-reference

## The actual integration story: how this gets used with an agent

1. **A WP4 pattern runs**, producing a raw graph-state result (`messages`
   + a validated `structured_response`, where the pattern produces one).
2. **`harness/adapters.py`** converts that raw result into this
   framework's `EvaluationInput` contract. This conversion is
   deliberately thin -- the contract was designed from the start to be
   tolerant of both live LangChain message objects and serialized JSON,
   so almost no transformation is needed.
3. **A `TaskDefinition`** (`contracts/task_definition.py`), if you have
   one, supplies whatever gold labels specific metrics need (a target
   answer, an expected tool call, an oracle function). Not required --
   metrics needing labels you don't have are skipped, not errored.
4. **`harness/evaluate_trajectory.py`** looks up every metric in the
   registry that can run from a single trajectory, checks what each one
   needs against your `TaskDefinition`, and runs everything it can. One
   metric raising doesn't kill the rest of the report.

Confirmed working for the single-pass ReAct-style pattern reviewed so
far. A trial-based self-correction pattern's *final* trajectory is
expected to work identically (same response-extraction mechanism,
confirmed by reading its source), though not yet run live end-to-end;
its trial-by-trial dynamics need a separate resolution (see Known Gaps).
A retrieve-then-generate pattern's compatibility is unconfirmed -- see
`harness/adapters.py`'s docstring for the specific open question.

## Applicability categories (10)

A metric's applicability is determined by what data it needs, not which
dimension it serves -- see `docs/pattern_metric_matrix.md` for the full
reasoning and the complete periodic-table cross-reference.

| Category | Requires | Runnable by the harness automatically? |
|---|---|---|
| `common` | `messages` + final `output`, optionally a gold label | Yes, where labels are supplied |
| `trajectory` | A tool-call trace, optionally a gold label | Yes, where labels are supplied |
| `scanners` | A full trajectory, no labels needed | Yes |
| `reflective` | A trial loop within one run | No -- needs a second orchestration layer, not yet built |
| `governance` | Governance-pattern escalation data | No |
| `efficiency` | A pre-measured or live-measured external value | No -- different data source entirely |
| `human_centricity` | Survey/participant responses | No |
| `probabilistic` | Token log-probabilities | No |
| `aggregate` | Multiple independent runs or a labeled case set | No |
| `meta_evaluation` | Evaluates the evaluation process itself, not the agent | No |

## Metric inventory: 47 metrics, 35 of 61 periodic table tools implemented

Full per-metric breakdown, including every blocked cell and the
specific reason it's blocked, lives in `docs/pattern_metric_matrix.md`
-- not duplicated here since it changes as gaps close. Headline numbers:

- **35 of 61** direct matches to the periodic table.

## Known gaps -- specific reasons, not "not started yet"

- **Multimodal** -- 0/10, no multimodal pattern reviewed in WP4's
  pattern set yet.
- **MAS-specific metrics** (7 of 61) -- no multi-agent
  trajectory fixture or contract exists yet.
- **Live-intervention metrics** (Interruptibility, Override success,
  Deferral-on-uncertainty, Confirmation-gating) -- blocked on a WP4
  harness hook that doesn't currently exist in any pattern reviewed.
- **Trajectory optimality, Judge win-rate/Elo's proper banding** --
  both need a live, versioned consortium score pool; Judge Elo's
  aggregation math is implemented and tested, just can't be banded the
  way the Comparative family intends without that shared pool.
- **Pass@k (code)** -- needs sandboxed code execution, a deliberate
  security-sensitive design decision not yet made.
- **Bias/fairness gap** -- needs a real decision on what "protected
  groups" means for AIXPERT's specific use cases before implementation.
- **Batch/aggregate orchestration** -- the harness currently closes the
  loop for single-trajectory metrics only (~15 of the 47). Running many
  trials and collecting results for `aggregate`/`reflective` metrics is
  a real, well-scoped next piece, not yet built.

## Where some of this came from

Several scorers and the `scanners` category are adapted from a
well-known open-source LLM evaluation framework's public documentation
(design patterns, not code). `metrics/efficiency/energy_carbon.py`
wraps `codecarbon`, the instrumentation tool recommended by Vector
Institute's "Data and Impact Accounting" position paper.
