"""
Task success -- Level 2 metric under Accuracy 

Applicability: common. Binary goal-oracle check: did the agent actually
achieve the stated goal, per a task-specific checker function.

Deliberately pluggable rather than a fixed algorithm, unlike most other
metrics in metrics/common/ -- "goal achievement" is inherently
task-specific (did the file get created, does the returned value equal
the correct answer, was the booking actually made) in a way that can't
be inferred generically from output text the way EM/F1/BLEU can. The
oracle_fn is where that task-specific logic lives; this evaluator just
provides the consistent EvaluationResult wrapper around it.

Deliberately distinct from Trajectory Consistency / Reward Hacking: task
success asks "was the OUTCOME correct", those ask "was the outcome
GROUNDED in the trajectory". The real "17+25=42" fixture makes the
distinction concrete: task_success (oracle: answer == 42) scores this
1.0 -- the outcome IS correct -- while trajectory_consistency scores it
0.0 and reward_hacking flags it, because the correct answer wasn't
actually grounded in any tool use or reasoning. All three results are
correct simultaneously; they're answering different questions.
"""

from __future__ import annotations

from typing import Any, Callable

from contracts.evaluator import EvaluationInput, EvaluationResult, WP3Evaluator


class TaskSuccessEvaluator(WP3Evaluator):
    dimensions = ("Accuracy",)
    applicability = "common"
    metric_id = "task_success"

    def evaluate_with_oracle(self, input: EvaluationInput, oracle_fn: Callable[[Any], bool]) -> EvaluationResult:
        """
        Args:
            oracle_fn: task-specific function taking the agent's final
                output (input.output) and returning True if the goal
                was achieved, False otherwise. Exceptions raised by the
                oracle (e.g. missing expected field on a malformed
                output) are caught and treated as a failed check, not
                propagated -- a broken output should score as
                "goal not achieved", not crash the evaluation run.
        """
        try:
            achieved = bool(oracle_fn(input.output))
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                explanation=f"Oracle function raised an exception on this output: {e}",
                is_success=False,
                metadata={"oracle_error": str(e)},
            )

        return EvaluationResult(
            score=1.0 if achieved else 0.0,
            explanation=f"Goal {'achieved' if achieved else 'not achieved'} per oracle check.",
            is_success=achieved,
            metadata={"goal_achieved": achieved},
        )

    def evaluate(self, input: EvaluationInput) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError("TaskSuccessEvaluator requires an oracle function. Call evaluate_with_oracle(input, oracle_fn) instead.")
