"""
Generate a real agent trajectory fixture by running a live agent pattern.

NOT part of the public evaluation-framework package -- this imports the
private agent framework directly. Run it locally; commit only the resulting
JSON fixture, never this script, to the public repo.
"""

import argparse
import json
from pathlib import Path

from inobo.utils.schemas import LLMConfig
# Adjust this import to wherever ReactAgent actually lives in your checkout.
from inobo.patterns.agent_orchestration.react_agent.pattern import ReactAgent


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def serialize_message(m) -> dict:
    data = {
        "type": m.__class__.__name__.lower().replace("message", ""),
        "content": m.content,
    }
    if getattr(m, "tool_calls", None):
        data["tool_calls"] = m.tool_calls
    if getattr(m, "tool_call_id", None):
        data["tool_call_id"] = m.tool_call_id
        data["name"] = getattr(m, "name", None)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is 17 plus 25?")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--out", default="real_react_trajectory_001.json")
    args = parser.parse_args()

    agent = ReactAgent(
        agent_role="a helpful assistant",
        task_description="Answer the user's question, using tools if needed.",
        input_schema={"query": str},
        output_schema={"answer": str},
        reasoning_llm_config=LLMConfig(model=args.model, model_provider=args.provider),
        tools=[add],
    )

    result = agent.run({"query": args.query})

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "input": args.query,
        "messages": [serialize_message(m) for m in result["messages"]],
        "structured_response": result["structured_response"].model_dump(),
    }, indent=2))

    print(f"Wrote {out_path}")
    print("structured_response:", result["structured_response"])


if __name__ == "__main__":
    main()
