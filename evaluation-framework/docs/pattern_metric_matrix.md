# Pattern ↔ Metric Applicability Matrix

Derived from reviewing four agent pattern implementations covering four
distinct architectures: a single-pass reasoning/tool-use loop (ReAct-style),
two trial-based self-correction loops — one with tool use, one without
(Reflexion-style and Self-Refine-style respectively), and a governance/
security wrapper applied around another pattern's execution. Not
exhaustive — extend this table as more patterns are reviewed.

| Metric/Method | metric_id | Dimension(s) | Computed from | ReAct | Reflexion | Self-Refine | Governance |
|---|---|---|---|:---:|:---:|:---:|:---:|
| Faithfulness | `faithfulness` | Explainability | messages + final output | ✅ | ✅ | ✅ | — |
| Schema-violation rate | `schema_violation_rate` | Accuracy | malformed/missing structured output count | ✅ | ✅ | ✅ | — |
| Task accuracy vs. ground truth | `task_accuracy` | Accuracy | final output vs. labels | ✅ | ✅ | ✅ | — |
| Trial-to-trial consistency | `trial_consistency` | Robustness | variance of per-trial evaluation score | — | ✅ | ✅ | — |
| Convergence rate | `convergence_rate` | Robustness | trial count until success | — | ✅ | ✅ | — |
| Score monotonicity | `score_monotonicity` | Robustness | trial-over-trial score trend | — | ✅ | ✅ | — |
| Reflection-groundedness | `reflection_groundedness` | Explainability | entailment: stated reflection vs. evaluation feedback | — | ✅ | ✅ | — |
| Log completeness / FLR | `log_completeness` | Transparency, Auditability | tool-call trace completeness | ✅ | ✅ | ⚠️ vacuous (no tools) | — |
| Governance coverage | `governance_coverage` | Auditability | fraction of tool calls passing through a governed wrapper | ✅ | ✅ | — | — |
| Prompt Injection Resistance | `prompt_injection_resistance` | Safety, Security & Privacy | escalation/audit events vs. labeled test set | — | — | — | ✅ |
| False-positive rate | *(reported alongside prompt_injection_resistance)* | Safety, Security & Privacy | same, benign-input side | — | — | — | ✅ |
| Escalation/audit completeness | `escalation_completeness` | Auditability | escalation/audit log well-formedness | — | — | — | ✅ |
| Policy Enforcement Correctness | `policy_enforcement_correctness` | Safety, Security & Privacy | default-deny conformance on unclassified tools | — | — | — | ✅ |

**Legend:** ✅ computable now from what's visible in the pattern's own
behavior/output. `—` structurally inapplicable (the pattern doesn't produce
the needed data). `⚠️` technically computable but the result carries no
signal (documented at the point of use, not silently treated as a passing
score).

## Why this table drives `metrics/`'s folder structure, not `dimensions/`

Organising `metrics/` by dimension would mean e.g. `faithfulness.py` living
under `metrics/explainability/`, `task_accuracy.py` under
`metrics/accuracy/`, and so on — but both are computed identically
regardless of dimension, and both are computable from every pattern. Trial
consistency, by contrast, is meaningless for a single-pass pattern no matter
which dimension you file it under. The applicability boundary (common /
reflective / trajectory / governance) is a property of *what data the
pattern produces*, which is orthogonal to *which dimension the metric
nominally serves*. Hence: one metric implementation, tagged with the
dimension(s) it serves via `WP3Evaluator.dimensions`, filed under the folder
that reflects what it actually needs to run.

## Known gap: Human-centricity

None of the four patterns reviewed expose a human-in-the-loop event. If
human-centricity metrics are wanted against these patterns, that requires
the underlying agent framework to instrument a new event (e.g. a
human-approval step before completion), not just a new evaluator against
existing data. Flagged here rather than silently worked around — there is
currently no `metrics/human_centricity/` category because there is nothing
to build it against yet.

## Not yet implemented

Metrics listed in the table above without a corresponding file under
`metrics/` are specified (dimension, applicability, data source) but not yet
coded. `faithfulness.py`, `consistency.py` (as `trial_consistency`),
`log_completeness.py`, and `prompt_injection_resistance.py` are the four
worked examples, one per applicability category — see each file's docstring.
