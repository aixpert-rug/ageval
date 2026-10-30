"""
Machine-readable Level 2 catalogue -- every metric in this repo,
what it needs, and how to call it. Exists because dozens of metrics across 9
applicability categories with inconsistent method signatures
(evaluate, evaluate_with_target, evaluate_with_oracle,
evaluate_labeled_set, evaluate_trials, ...) is not discoverable by
reading source files one at a time. This is the human-readable
docs/pattern_metric_matrix.md, made queryable.

Only imports classes, never instantiates them here -- some constructors
have real side effects or requirements (METEOREvaluator raises if
WordNet isn't installed; ToxicityEvaluator requires a classifier
argument), which is exactly why this file must stay import-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from metrics.common.bertscore import BERTScoreEvaluator
from metrics.common.bleu import BLEUEvaluator
from metrics.common.exact_match import ExactMatchEvaluator
from metrics.common.meteor import METEOREvaluator
from metrics.common.rouge import ROUGEEvaluator
from metrics.common.schema_violation_rate import SchemaViolationRateEvaluator
from metrics.common.subgoal_progress import SubgoalProgressEvaluator
from metrics.common.math_accuracy import MathAccuracyEvaluator, is_numeric_target
from metrics.common.task_accuracy import TaskAccuracyEvaluator
from metrics.common.task_success import TaskSuccessEvaluator
from metrics.common.toxicity import ToxicityEvaluator

from metrics.reflective.consistency import ConsistencyEvaluator
from metrics.reflective.convergence_rate import ConvergenceRateEvaluator
from metrics.reflective.score_monotonicity import ScoreMonotonicityEvaluator

from metrics.trajectory.log_completeness import LogCompletenessEvaluator
from metrics.trajectory.step_efficiency import StepEfficiencyEvaluator
from metrics.trajectory.tool_call_accuracy import ToolCallAccuracyEvaluator

from metrics.governance.prompt_injection_resistance import PromptInjectionResistanceEvaluator


from metrics.efficiency.cost_per_query import CostPerQueryEvaluator
from metrics.efficiency.energy_carbon import EnergyCarbonEvaluator
from metrics.efficiency.latency import LatencyEvaluator
from metrics.efficiency.memory_footprint import MemoryFootprintEvaluator
from metrics.efficiency.throughput import ThroughputEvaluator

from metrics.human_centricity.cognitive_load import CognitiveLoadEvaluator
from metrics.human_centricity.perceived_usability import PerceivedUsabilityEvaluator

from metrics.probabilistic.ece import ECEEvaluator
from metrics.probabilistic.perplexity import PerplexityEvaluator

from metrics.aggregate.abstention_rate import AbstentionRateEvaluator
from metrics.aggregate.cross_run_consistency import CrossRunConsistencyEvaluator
from metrics.aggregate.error_recovery_rate import ErrorRecoveryRateEvaluator
from metrics.aggregate.escalation_appropriateness import EscalationAppropriatenessEvaluator
from metrics.aggregate.help_seeking import HelpSeekingEvaluator
from metrics.aggregate.judge_elo import JudgeEloEvaluator
from metrics.aggregate.pass_at_k import PassAtKAgentEvaluator
from metrics.aggregate.pass_hat_k import PassHatKAgentEvaluator
from metrics.aggregate.refusal_rate import SafeRefusalRateEvaluator, UnsafeRefusalRateEvaluator
from metrics.aggregate.unsafe_action_rate import UnsafeActionRateEvaluator

from metrics.meta_evaluation.contamination import ContaminationEvaluator
from metrics.meta_evaluation.correlation_metrics import JudgeHumanAgreementEvaluator, MetricValidityEvaluator
from metrics.meta_evaluation.discriminative_power import DiscriminativePowerEvaluator
from metrics.meta_evaluation.inter_rater_reliability import InterRaterReliabilityEvaluator
from metrics.meta_evaluation.judge_bias import JudgeBiasEvaluator
from metrics.meta_evaluation.sensitivity import SensitivityEvaluator
from metrics.meta_evaluation.test_retest_stability import TestRetestStabilityEvaluator


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    cls: type
    applicability: str
    dimensions: tuple[str, ...]
    entry_point: str
    requires: tuple[str, ...] = field(default_factory=tuple)
    single_trajectory: bool = True
    notes: str = ""
    #: optional extra applicability test on the TaskDefinition (e.g. numeric target only)
    applies_if: Callable[[Any], bool] | None = None


REGISTRY: list[MetricSpec] = [
    MetricSpec("schema_violation_rate", SchemaViolationRateEvaluator, "common", ("Accuracy",), "evaluate_batch", single_trajectory=False, notes="Needs a BATCH of structured_response values, not one trajectory."),
    MetricSpec("task_accuracy", TaskAccuracyEvaluator, "common", ("Accuracy",), "evaluate_with_target", requires=("target",)),
    MetricSpec("exact_match", ExactMatchEvaluator, "common", ("Accuracy",), "evaluate_with_target", requires=("target",)),
    MetricSpec("bleu", BLEUEvaluator, "common", ("Accuracy",), "evaluate_with_target", requires=("target",)),
    MetricSpec("rouge_l", ROUGEEvaluator, "common", ("Accuracy",), "evaluate_with_target", requires=("target",)),
    MetricSpec("meteor", METEOREvaluator, "common", ("Accuracy",), "evaluate_with_target", requires=("target",), notes="Constructor raises if WordNet isn't installed -- see meteor_setup.py."),
    MetricSpec("bertscore", BERTScoreEvaluator, "common", ("Accuracy",), "evaluate_with_target", requires=("target",), notes="First call downloads a model -- see module docstring."),
    MetricSpec("math_accuracy", MathAccuracyEvaluator, "common", ("Accuracy",), "evaluate_with_target", requires=("target",), applies_if=is_numeric_target, notes="Runs only when the target is a single number."),
    MetricSpec("task_success", TaskSuccessEvaluator, "common", ("Accuracy",), "evaluate_with_oracle", requires=("oracle_fn",)),
    MetricSpec("subgoal_progress", SubgoalProgressEvaluator, "common", ("Accuracy",), "evaluate_with_subgoals", requires=("subgoal_checks",)),
    MetricSpec("toxicity", ToxicityEvaluator, "common", ("Safety, Security & Privacy",), "evaluate", requires=("toxicity_classifier",), notes="Classifier is a CONSTRUCTOR arg, not a call arg."),

    MetricSpec("log_completeness", LogCompletenessEvaluator, "trajectory", ("Transparency", "Auditability / Verifiability"), "evaluate"),
    MetricSpec("tool_call_accuracy", ToolCallAccuracyEvaluator, "trajectory", ("Accuracy",), "evaluate_with_expected_calls", requires=("expected_tool_calls",)),
    MetricSpec("step_efficiency", StepEfficiencyEvaluator, "trajectory", ("Efficiency",), "evaluate_with_optimal", requires=("optimal_steps",)),


    MetricSpec("trial_consistency", ConsistencyEvaluator, "reflective", ("Robustness",), "evaluate_trials", single_trajectory=False),
    MetricSpec("convergence_rate", ConvergenceRateEvaluator, "reflective", ("Robustness",), "evaluate_trials", single_trajectory=False),
    MetricSpec("score_monotonicity", ScoreMonotonicityEvaluator, "reflective", ("Robustness",), "evaluate_trials", single_trajectory=False),

    MetricSpec("prompt_injection_resistance", PromptInjectionResistanceEvaluator, "governance", ("Safety, Security & Privacy",), "evaluate_test_set", single_trajectory=False),

    MetricSpec("latency", LatencyEvaluator, "efficiency", ("Efficiency",), "evaluate_measurement", single_trajectory=False),
    MetricSpec("cost_per_query", CostPerQueryEvaluator, "efficiency", ("Efficiency",), "evaluate_measurement", single_trajectory=False),
    MetricSpec("throughput", ThroughputEvaluator, "efficiency", ("Efficiency",), "evaluate_measurement", single_trajectory=False),
    MetricSpec("memory_footprint", MemoryFootprintEvaluator, "efficiency", ("Efficiency",), "evaluate_measurement", single_trajectory=False),
    MetricSpec("energy_carbon", EnergyCarbonEvaluator, "efficiency", ("Efficiency",), "evaluate_measurement", single_trajectory=False),

    MetricSpec("perceived_usability", PerceivedUsabilityEvaluator, "human_centricity", ("Human-centricity",), "evaluate_responses", single_trajectory=False),
    MetricSpec("cognitive_load", CognitiveLoadEvaluator, "human_centricity", ("Human-centricity",), "evaluate_ratings", single_trajectory=False),

    MetricSpec("perplexity", PerplexityEvaluator, "probabilistic", ("Accuracy",), "evaluate_logprobs", single_trajectory=False),
    MetricSpec("ece", ECEEvaluator, "probabilistic", ("Transparency & Explainability",), "evaluate_predictions", single_trajectory=False),

    MetricSpec("cross_run_consistency", CrossRunConsistencyEvaluator, "aggregate", ("Robustness",), "evaluate_runs", single_trajectory=False),
    MetricSpec("unsafe_refusal_rate", UnsafeRefusalRateEvaluator, "aggregate", ("Safety, Security & Privacy",), "evaluate_labeled_set", single_trajectory=False, notes="Higher is better (H). Scores the unsafe-labeled cases of a (trajectory, is_unsafe) set."),
    MetricSpec("safe_refusal_rate", SafeRefusalRateEvaluator, "aggregate", ("Safety, Security & Privacy",), "evaluate_labeled_set", single_trajectory=False, notes="Over-refusal; lower is better (L). Scores the safe-labeled cases of the same set."),
    MetricSpec("abstention_rate", AbstentionRateEvaluator, "aggregate", ("Autonomy",), "evaluate_labeled_set", single_trajectory=False),
    MetricSpec("help_seeking", HelpSeekingEvaluator, "aggregate", ("Autonomy",), "evaluate_labeled_set", single_trajectory=False),
    MetricSpec("pass_at_k_agent", PassAtKAgentEvaluator, "aggregate", ("Accuracy",), "evaluate_problems", single_trajectory=False),
    MetricSpec("pass_hat_k_agent", PassHatKAgentEvaluator, "aggregate", ("Accuracy",), "evaluate_problems", single_trajectory=False, notes="pass^k reliability: all k runs succeed. Same (n, c) input as pass_at_k_agent."),
    MetricSpec("judge_elo", JudgeEloEvaluator, "aggregate", ("Accuracy",), "compute_ratings", single_trajectory=False),
    MetricSpec("escalation_appropriateness", EscalationAppropriatenessEvaluator, "aggregate", ("Autonomy",), "evaluate_labeled_set", single_trajectory=False),
    MetricSpec("error_recovery_rate", ErrorRecoveryRateEvaluator, "aggregate", ("Robustness",), "evaluate_labeled_set", single_trajectory=False),
    MetricSpec("unsafe_action_rate", UnsafeActionRateEvaluator, "aggregate", ("Safety, Security & Privacy",), "evaluate_steps", single_trajectory=False, notes="VETO metric -- see module docstring."),

    MetricSpec("judge_human_agreement", JudgeHumanAgreementEvaluator, "meta_evaluation", ("Transparency & Explainability",), "evaluate_agreement", single_trajectory=False),
    MetricSpec("metric_validity", MetricValidityEvaluator, "meta_evaluation", ("Auditability / Verifiability",), "evaluate_validity", single_trajectory=False),
    MetricSpec("inter_rater_reliability", InterRaterReliabilityEvaluator, "meta_evaluation", ("Transparency & Explainability",), "evaluate_ratings", single_trajectory=False),
    MetricSpec("discriminative_power", DiscriminativePowerEvaluator, "meta_evaluation", ("Auditability / Verifiability",), "evaluate_groups", single_trajectory=False),
    MetricSpec("test_retest_stability", TestRetestStabilityEvaluator, "meta_evaluation", ("Robustness",), "evaluate_reruns", single_trajectory=False),
    MetricSpec("sensitivity", SensitivityEvaluator, "meta_evaluation", ("Auditability / Verifiability",), "evaluate_known_gaps", single_trajectory=False),
    MetricSpec("contamination", ContaminationEvaluator, "meta_evaluation", ("Auditability / Verifiability",), "evaluate_overlap", single_trajectory=False, notes="VETO metric -- see module docstring."),
    MetricSpec("judge_bias", JudgeBiasEvaluator, "meta_evaluation", ("Fairness, Ethical & Legal Aspects",), "evaluate_controlled_pairs", single_trajectory=False),
]


def metrics_for_applicability(applicability: str) -> list[MetricSpec]:
    return [m for m in REGISTRY if m.applicability == applicability]


def single_trajectory_metrics() -> list[MetricSpec]:
    return [m for m in REGISTRY if m.single_trajectory]


def find(metric_id: str) -> MetricSpec:
    for m in REGISTRY:
        if m.metric_id == metric_id:
            return m
    raise KeyError(f"No metric registered with id {metric_id!r}.")
