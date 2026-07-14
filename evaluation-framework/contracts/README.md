# The Framework / Agent-Implementation Boundary

**Public (here):** the shape of `Evaluator`, `EvaluationInput`, `EvaluationResult`,
and every metric implementation that consumes them. Also: dimension
definitions, metric specs, KPI templates.

**Private (external, not in this repo):** the underlying agent framework —
pattern implementations, their internal orchestration, prompts, and any
default built-in evaluators they ship with.

## Why structural typing, not a shared import

`evaluator.py` in this repo does not import from the external agent
framework's package. If it did, this public repo would depend on a private
codebase it doesn't own, which defeats the point of the split. Instead,
`EvaluationInput`/`EvaluationResult`/`Evaluator` here are a **manual
structural mirror** of the external framework's private schemas. Because
`Evaluator` is a `typing.Protocol`, any evaluator subclass defined here
satisfies the external framework's own evaluator interface at runtime
purely by shape — no inheritance, no shared package, no version coupling
beyond "the fields still match."

## Keeping this in sync

This mirror can drift if the external framework changes its private schema.
Mitigation:

1. **Pin a version note.** The dataclasses in this repo are believed
   accurate as of the source reviewed on [DATE / commit ref once shared].
   Update this file whenever that changes.
2. **Runtime contract test, not just a static one.** Before any evaluator
   here is used against a real external agent, run it against a **mock**
   `EvaluationInput` built the way the external framework's code actually
   builds it (see `fixtures/`) — this catches shape drift before it reaches
   a live integration.
3. **Escalate schema changes explicitly.** If the external framework changes
   `EvaluationInput`'s fields, that's a breaking change for every evaluator
   here — treat it like a versioned API change, not a silent internal
   refactor.

## What's NOT covered by this contract

- **Not every pattern has an injected-evaluator hook.** Some patterns run
  once and return a final result directly, with no evaluation step built
  into their execution loop. Evaluators for those patterns are external —
  called on the return value of a run, not injected via this contract. See
  `metrics/common/` — those are written to work either way.
- **Governance/security-style patterns** don't produce an `EvaluationInput`
  at all. They typically expose a list of escalation/audit events instead.
  `metrics/governance/` reads that directly — a second, separate contract,
  documented there rather than here.
