# The WP3 / WP4 Boundary

**Public (here):** the shape of `Evaluator`, `EvaluationInput`, `EvaluationResult`,
and every `WP3Evaluator` implementation that consumes them. Also: dimension
definitions, metric specs, KPI templates.

**Private (WP4, D4.1/D4.2):** the `inobo` package — pattern implementations
(`ReactAgent`, `ReflexionAgent`, `SelfRefineAgent`, governance wrappers), their
LangGraph wiring, prompts, and the default LLM-as-judge evaluators
(`TrajectoryEvaluator`, `OutputEvaluator`).

## Why structural typing, not a shared import

`evaluator.py` in this repo does not import from `inobo`. If it did, this
public repo would depend on WP4's private package, which defeats the
point of the split. Instead, `EvaluationInput`/`EvaluationResult`/`Evaluator`
here are a **manual structural mirror** of WP4's private schemas. Because
`Evaluator` is a `typing.Protocol`, any `WP3Evaluator` subclass satisfies
WP4's `Evaluator` ABC at runtime purely by shape — no inheritance, no shared
package, no version coupling beyond "the fields still match."

## Keeping this in sync

This mirror can drift if WP4 changes their private schema. Mitigation:

1. **Pin a version note.** The dataclasses above are believed accurate as of
   the `ReflexionAgent`/`SelfRefineAgent` source reviewed [DATE / commit ref
   once WP4 shares one]. Update this file whenever that changes.
2. **Runtime contract test, not just a static one.** Before any WP3 evaluator
   goes to WP4 for a real integration test, run it against a **mock**
   `EvaluationInput` built the way WP4's code actually builds it (see
   `fixtures/`) — this catches shape drift before it reaches WP4.
3. **Escalate schema changes explicitly.** If WP4 changes `EvaluationInput`'s
   fields, that's a breaking change for every `WP3Evaluator` — treat it like
   a versioned API change, not a silent internal refactor.

## What's NOT covered by this contract

- **ReAct has no injected-evaluator hook.** It runs once and returns
  `{messages, structured_response}` — WP3 evaluators for ReAct are external,
  called on the return value of `.run()`, not injected via this contract.
  See `metrics/common/` — those are written to work either way.
- **Governance patterns** (e.g. `IsolatedAgentSelfDefence`) don't produce an
  `EvaluationInput` at all. They expose `self.escalations` (a list of
  `EscalationEvent`s) instead. `metrics/governance/` reads that directly —
  a second, separate contract, documented there rather than here.
