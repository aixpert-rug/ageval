# Pattern <-> Metric Applicability Matrix

As of this release, the ReAct-style pattern has been run live end to
end -- real agent execution, real adapter, real metrics, in one process
-- including a genuine tool-*selection* test (two tools available, the
framework correctly evaluated whether the right one was chosen, not
merely whether any tool was used). See the repo README's "Status"
section and `harness/` for the integration layer that makes this work.

## The ten applicability categories

| Category | Requires | Applies to |
|---|---|---|
| `common` | `messages` + final `output` | Any pattern (single-pass or trial-based) |
| `reflective` | A trial loop within ONE run | Trial-based self-correction patterns only |
| `trajectory` | A tool-call trace | Tool-using patterns only |
| `governance` | Governance-pattern escalation data (`EscalationEvent`) | Governance wrappers specifically |
| `scanners` | A full trajectory, checked for a pattern's presence | Any pattern; may report nothing on a clean trajectory |
| `efficiency` | A pre-measured or live-measured external value + a declared budget | Any pattern; measurement is the caller's job (or the metric's own live measurer, e.g. `codecarbon`) |
| `human_centricity` | Survey/participant responses | Any pattern's output, evaluated by humans |
| `probabilistic` | Token log-probabilities | Any pattern whose provider exposes logprobs |
| `aggregate` | Multiple independent runs, or a labeled case set | Any pattern; needs more than one trajectory |
| `meta_evaluation` | Evaluates the evaluation process itself, not the agent | N/A -- cross-cutting, not pattern-specific |

## Full metric inventory, mapped against Vector's periodic table

Legend: [OK] = direct match to a periodic-table cell;
[DIFF] = same name, deliberately different construct (both versions
built where it made sense); -- = net-new, not on table;
[BLOCKED reason] = specified but not buildable yet, with the specific
reason noted.

### LLM-level

| Metric | Status | Our metric_id |
|---|---|---|
| Exact Match | [OK] | `exact_match` |
| F1 | [OK] | `task_accuracy` |
| Pass@k (code) | [BLOCKED -- needs sandboxed code execution, a security-sensitive design decision not yet made] | -- |
| BLEU | [OK] | `bleu` |
| ROUGE | [OK] | `rouge_l` |
| METEOR | [OK] | `meteor` |
| BERTScore | [OK] | `bertscore` |
| Judge win-rate / Elo | [OK] (aggregation math implemented + verified against the standard textbook example; proper consortium-quartile banding blocked -- see Trajectory optimality) | `judge_elo` |
| Perplexity | [OK] | `perplexity` |
| ECE (calibration) | [OK] | `ece` |
| Faithfulness | [DIFF] | `trajectory_consistency` (self-consistency against the agent's own trajectory, not source-document entailment -- deliberately deferred pending a real source-grounding use case; a RAG-style pattern now supplies one, not yet built) |
| Toxicity | [OK] (requires an explicit, externally-validated classifier -- deliberately ships no hand-rolled lexicon) | `toxicity` |
| Refusal rate | [OK] | `refusal_rate` (aggregate, calibrated target-band rate) -- see also `refusal` (scanners, per-trajectory, a different question) |
| Bias/fairness gap | [BLOCKED -- needs a real decision on what "protected groups" means for AIXPERT's specific use cases] | -- |
| Abstention rate | [OK] | `abstention_rate` |
| Escalation appropriateness | [OK] | `escalation_appropriateness` |
| Latency | [OK] (live-measurable, not just pre-measured-value-only) | `latency` |
| Throughput | [OK] | `throughput` |
| Cost per query | [OK] (live-computable from token counts + pricing) | `cost_per_query` |
| Memory | [OK] (live-measurable; default measurer is Python-object-only, pluggable for GPU-aware measurement) | `memory_footprint` |
| Energy/carbon | [OK] (wraps `codecarbon`, the tool DIA position paper recommends; includes water tracking) | `energy_carbon` |

### Agentic, single- and multi-agent (Vector's 22)

| Metric | Status | Our metric_id |
|---|---|---|
| Task success rate | [OK] | `task_success` |
| Sub-goal / progress | [OK] | `subgoal_progress` |
| Pass@k (agent) | [OK] (unbiased estimator, verified against known reference values) | `pass_at_k_agent` |
| Tool-call accuracy | [OK] | `tool_call_accuracy` |
| Step efficiency | [OK] | `step_efficiency` |
| Trajectory optimality | [BLOCKED -- Comparative family needs a live, versioned consortium score pool we don't maintain; the within-set relative-normalisation approach used for Judge Elo could plausibly adapt here, not yet attempted] | -- |
| Error recovery rate | [OK] (data need is fault-injection trials; WP4 doesn't currently expose a fault-injection harness, but the metric itself doesn't need to wait for that) | `error_recovery_rate` |
| Consistency | [OK] | `cross_run_consistency` -- see also `trial_consistency` (reflective, within one trial loop, a different question) |
| Unsafe-action rate | [OK] --  implemented with veto logic | `unsafe_action_rate` |
| Interruptibility | [BLOCKED -- live-intervention hook, none of the reviewed patterns expose one] | -- |
| Override success | [BLOCKED -- same live-intervention gap] | -- |
| Appropriate help-seeking | [OK] | `help_seeking` |
| Deferral-on-uncertainty | [BLOCKED -- same live-intervention gap, needs per-step confidence elicitation specifically] | -- |
| Confirmation-gating | [BLOCKED -- same live-intervention gap;] | -- |
| Cost (tokens/$/latency) | [OK] -- covered by `latency`/`cost_per_query`, same ratio-to-budget shape whether applied to one call or a full trajectory | -- |
| MAS-specific (7 metrics: comm. quality, delegation, consensus resolution, error-propagation containment, deadlock rate, cumulative unsafe-action rate, coordination efficiency) | [BLOCKED -- no multi-agent trajectory fixture or contract exists yet] | -- |


All 10 Multimodal patters [BLOCKED -- no multimodal pattern reviewed in WP4's pattern set yet].

### Meta-evaluation-- complete, 8/8

| Metric | Status | Our metric_id |
|---|---|---|
| Judge-human agreement | [OK] (Pearson correlation) | `judge_human_agreement` |
| Inter-rater reliability | [OK] (Cohen's Kappa; Krippendorff's alpha, the >2-rater generalisation, not implemented -- a known, disclosed gap) | `inter_rater_reliability` |
| Metric validity | [OK] (same correlation math as Judge-human agreement, different data source) | `metric_validity` |
| Discriminative power | [OK] (eta-squared, the standard ANOVA effect-size statistic) | `discriminative_power` |
| Test-retest stability | [OK] (mathematically identical to `cross_run_consistency`, kept separate: this asks whether the EVALUATION PROCESS is stable on rerun, that asks whether the AGENT's behaviour is) | `test_retest_stability` |
| Sensitivity | [OK] (accuracy on known-gap system pairs) | `sensitivity` |
| Contamination / leakage | [OK] | `contamination` |
| Judge bias | [OK] (mean absolute score shift from a controlled artefact) | `judge_bias` |

`math_accuracy`, `schema_violation_rate`, `convergence_rate`,
`score_monotonicity`, `log_completeness`, `prompt_injection_resistance`,
`reward_hacking`, `perceived_usability` (SUS), `cognitive_load`
(NASA-TLX) -- extend the framework into territory the table doesn't
cover: agentic self-correction dynamics, governance-mechanism
verification, and Human-centricity, grounded in T3.1's own DoA text requiring
human-centric explainability evaluation.

## The integration layer -- `contracts/registry.py` and `harness/`

`contracts/registry.py` is a machine-readable catalogue of all 47
metrics: applicability, dimensions, which method to call, and what
extra data (if any) it needs beyond the trajectory itself. Exists
because 47 metrics with genuinely different call signatures
(`evaluate`, `evaluate_with_target`, `evaluate_with_oracle`,
`evaluate_labeled_set`, `evaluate_trials`, ...) isn't discoverable by
reading source files one at a time -- this is docs/pattern_metric_matrix.md,
made queryable.

`harness/evaluate_trajectory.py` uses that registry to automatically
run every metric it has enough data for, given one captured trajectory
and an optional `TaskDefinition` of gold labels. Currently covers the
~15 metrics answerable from a single trajectory (`common`, `trajectory`,
`scanners`); the `aggregate`/`reflective`/`efficiency`/
`human_centricity`/`probabilistic`/`meta_evaluation` categories need a
second orchestration layer -- running many trials and collecting
results -- not yet built.
