"""
Load a captured trajectory and run it through the harness, against one
of a small catalogue of known test tasks (add a new entry to KNOWN_TASKS
each time you test a new query, rather than hand-editing one hardcoded
TaskDefinition every run).

Usage:
    python run_live_evaluation.py fixtures/real_react_trajectory_003.json --task multiply_17_25
    python run_live_evaluation.py fixtures/real_react_trajectory_003.json          # label-free metrics only
"""

import argparse
import json
from pathlib import Path

from contracts.evaluator import EvaluationInput
from contracts.task_definition import TaskDefinition
from harness.evaluate_trajectory import evaluate_trajectory, print_report

KNOWN_TASKS: dict[str, TaskDefinition] = {
    "add_17_25": TaskDefinition(
        task_id="add_17_25",
        input_query="What is 17 plus 25?",
        target="42",
        expected_tool_calls=[{"name": "add"}],
        oracle_fn=lambda output: "42" in str(output.get("answer", "")),
        optimal_steps=1,
    ),
    "multiply_17_25": TaskDefinition(
        task_id="multiply_17_25",
        input_query="What is 17 times 25?",
        target="425",
        # The actual point of this task: does it pick multiply, not add --
        # single-tool tasks can't test tool SELECTION, only tool USE.
        expected_tool_calls=[{"name": "multiply"}],
        oracle_fn=lambda output: "425" in str(output.get("answer", "")),
        optimal_steps=1,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_path", type=Path)
    parser.add_argument(
        "--task", choices=list(KNOWN_TASKS.keys()), default=None,
        help="Which gold-label set to score against. Omit to run label-free metrics only.",
    )
    args = parser.parse_args()

    data = json.loads(args.trajectory_path.read_text())
    trajectory = EvaluationInput(messages=data["messages"], input=data["input"], output=data["structured_response"])

    task_def = KNOWN_TASKS[args.task] if args.task else None

    print(f"=== Evaluating {args.trajectory_path.name} (task={args.task or 'none -- label-free only'}) ===\n")
    report = evaluate_trajectory(trajectory, task_def, skip_metric_ids=("bertscore", "meteor"))
    print_report(report)
    print(f"\n{len(report)} metrics ran.")


if __name__ == "__main__":
    main()
