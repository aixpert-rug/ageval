# Pattern ↔ Metric Applicability Matrix

Derived from reading four WP4 (`inobo`) pattern implementations:
`ReactAgent`, `ReflexionAgent`, `SelfRefineAgent`, `IsolatedAgentSelfDefence`.
Not exhaustive — WP4's pattern library has more patterns than these four;
extend this table as more are reviewed.

| Metric/Method | metric_id | Dimension(s) | Computed from | ReAct | Reflexion | Self-Refine | Governance |
|---|---|---|---|:---:|:---:|:---:|:---:|
| Faithfulness | `faithfulness` | Explainability | `messages` + `output` | ✅ | ✅ | ✅ | — |
| Schema-violation rate | `schema_violation_rate` | Accuracy | `structured_response is None` count | ✅ | ✅ | ✅ | — |
| Task accuracy vs. ground truth | `task_accuracy` | Accuracy | `structured_response` vs. labels | ✅ | ✅ | ✅ | — |
| Trial-to-trial consistency | `trial_consistency` | Robustness | variance of `eval_result.score` across `reflections` | — | ✅ | ✅ | — |
| Convergence rate | `convergence_rate` | Robustness | `num_iterations` to `is_success=True` | — | ✅ | ✅ | — |
| Score monotonicity | `score_monotonicity` | Robustness | trial-over-trial `eval_result.score` trend | — | ✅ | ✅ | — |
| Reflection-groundedness | `reflection_groundedness` | Explainability | entailment: `reflection` text vs. `eval_result.explanation` | — | ✅ | ✅ | — |
| Log completeness / FLR | `log_completeness` | Transparency, Auditability | tool-call trace completeness | ✅ | ✅ | ⚠️ vacuous (no tools) | — |
| Governance coverage | `governance_coverage` | Auditability | fraction of tool calls passing through governed wrapper | ✅ | ✅ | — | — |
| Prompt Injection Resistance | `prompt_injection_resistance` | Safety, Security & Privacy | `EscalationEvent`s vs. labeled test set | — | — | — | ✅ |
| False-positive rate | *(reported alongside prompt_injection_resistance)* | Safety, Security & Privacy | same, benign-input side | — | — | — | ✅ |
| Escalation/audit completeness | `escalation_completeness` | Auditability | `self.escalations` well-formedness | — | — | — | ✅ |
| Policy Enforcement Correctness | `policy_enforcement_correctness` | Safety, Security & Privacy | default-deny conformance on unclassified tools | — | — | — | ✅ |

**Legend:** ✅ computable now from what's visible in the pattern's own code.
`—` structurally inapplicable (the pattern doesn't produce the needed data).
`⚠️` technically computable but the result carries no signal (documented at
the point of use, not silently treated as a passing score).

## Why this table drives `metrics/`'s folder structure, not `dimensions/`

Organising `metrics/` by dimension would mean e.g. `faithfulness.py` living
under `metrics/explainability/`, `task_accuracy.py` under
`metrics/accuracy/`, and so on — but both are computed identically
regardless of dimension, and both are computable from every pattern. Trial
consistency, by contrast, is meaningless for ReAct no matter which dimension
you file it under. The applicability boundary (common / reflective /
trajectory / governance) is a property of *what data the pattern produces*,
which is orthogonal to *which dimension the metric nominally serves*. Hence:
one metric implementation, tagged with the dimension(s) it serves via
`WP3Evaluator.dimensions`, filed under the folder that reflects what it
actually needs to run.

## Known gap: Human-centricity

None of the four patterns reviewed expose a human-in-the-loop event. If
AIXPERT wants Human-centricity metrics computed against these patterns, that
requires WP4 to instrument a new event (e.g. a human-approval step around
`finish_task`), not just a new WP3 evaluator against existing data. Flagged
here rather than silently worked around — there is currently no
`metrics/human_centricity/` category because there is nothing to build it
against yet.

## Not yet implemented

Metrics listed in the table above without a corresponding file under
`metrics/` are specified (dimension, applicability, data source) but not yet
coded. `faithfulness.py`, `consistency.py` (as `trial_consistency`),
`log_completeness.py`, and `prompt_injection_resistance.py` are the four
worked examples, one per applicability category — see each file's docstring.
