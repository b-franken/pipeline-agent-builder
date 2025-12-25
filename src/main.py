import asyncio
import os
import sys
import time

from langchain_core.messages import HumanMessage

from src.graph.workflow import create_workflow
from src.types import AgentState


async def run_task(task: str) -> None:
    workflow = create_workflow(use_persistence=False)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=task)],
        "current_agent": "supervisor",
        "context": {},
        "human_feedback": None,
        "iteration_count": 0,
        "execution_trace": [],
        "loop_counts": {},
    }

    result = await workflow.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": "main"}},
    )

    print()
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if not msg.content:
            continue
        if msg.content.startswith("[Supervisor]"):
            continue
        from langchain_core.messages import AIMessage as AI

        if isinstance(msg, AI):
            print(msg.content)
            break


async def run_interactive() -> None:
    workflow = create_workflow()
    print("Multi-Agent System Ready. Type 'exit' to quit.")
    thread_id = "interactive"

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "current_agent": "supervisor",
                "context": {},
                "human_feedback": None,
                "iteration_count": 0,
                "execution_trace": [],
                "loop_counts": {},
            }

            result = await workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )

            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                agent_name = getattr(last_msg, "name", "Agent")
                print(f"\n{agent_name}: {last_msg.content}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_server() -> None:
    print("Initializing multi-agent workflow...")
    workflow = create_workflow()
    print(f"Workflow ready: {workflow}")
    print("Agent container running. Waiting for tasks...")

    while True:
        time.sleep(60)


def main() -> None:
    args = sys.argv[1:]

    if not args:
        if os.environ.get("USE_POSTGRES") or os.environ.get("CHROMA_HOST"):
            run_server()
        else:
            asyncio.run(run_interactive())
    elif args[0] == "--interactive" or args[0] == "-i":
        asyncio.run(run_interactive())
    elif args[0] == "--server" or args[0] == "-s":
        run_server()
    else:
        task = " ".join(args)
        asyncio.run(run_task(task))


if __name__ == "__main__":
    main()
