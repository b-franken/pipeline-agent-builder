"""
Run agents with real-time streaming output.

Usage:
    python run.py "Write a story about a rabbit"
    python run.py "Create a Python script that sorts files"
"""

import sys
import asyncio

from src.config import settings
from src.trace import reset_tracer, TraceLevel

# Check API key (not needed for Ollama)
if settings.provider != "ollama" and not settings.get_api_key():
    print(f"ERROR: No API key found for provider '{settings.provider}'")
    print(f"Set {settings.provider.upper()}_API_KEY in config.py or .env file")
    sys.exit(1)

from src.graph.workflow import create_workflow
from src.registry import get_registry
from langchain_core.messages import HumanMessage


async def run_streaming(prompt: str, trace_level: TraceLevel = TraceLevel.NORMAL):
    """Run agents with real-time token streaming."""
    tracer = reset_tracer(trace_level)
    registry = get_registry()
    agent_names = set(registry.get_agent_names()) | {"supervisor"}

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    workflow = create_workflow(use_persistence=False)

    state = {
        "messages": [HumanMessage(content=prompt)],
        "current_agent": "user",
        "context": {},
        "human_feedback": None,
        "iteration_count": 0,
        "handoff_history": [],
    }

    current_node = None

    async for event in workflow.astream_events(state, version="v2"):
        event_type = event.get("event")

        if event_type == "on_chain_start":
            node_name = event.get("name")
            if node_name in agent_names:
                if current_node != node_name:
                    current_node = node_name
                    print(f"\n[{node_name.upper()}]", flush=True)

        elif event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "tool")
            print(f"\n  → Using {tool_name}...", flush=True)

        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", "")
            if tool_output and "Handing off" in str(tool_output):
                print(f"  → {tool_output}", flush=True)

    tracer.summary()
    print("Done.")


async def run_simple(prompt: str):
    """Run agents without streaming (simpler, for debugging)."""
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    workflow = create_workflow(use_persistence=False)

    state = {
        "messages": [HumanMessage(content=prompt)],
        "current_agent": "user",
        "context": {},
        "human_feedback": None,
        "iteration_count": 0,
        "handoff_history": [],
    }

    async for event in workflow.astream(state):
        for node, output in event.items():
            if node == "__end__":
                continue
            print(f"[{node.upper()}]")
            if "messages" in output:
                for msg in output["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        print(f"{msg.content}\n")

    print(f"{'='*60}\nDone.\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Provide a prompt:")
        print('  python run.py "your question here"')
        print()
        print("Options:")
        print("  --no-stream  Disable streaming (for debugging)")
        print("  --verbose    More details in trace output")
        print("  --debug      Log everything (including state)")
        sys.exit(0)

    use_streaming = "--no-stream" not in sys.argv
    trace_level = TraceLevel.NORMAL
    if "--debug" in sys.argv:
        trace_level = TraceLevel.DEBUG
    elif "--verbose" in sys.argv:
        trace_level = TraceLevel.VERBOSE

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prompt = " ".join(args)

    if use_streaming:
        asyncio.run(run_streaming(prompt, trace_level))
    else:
        asyncio.run(run_simple(prompt))


if __name__ == "__main__":
    main()
