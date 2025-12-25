"""Supervisor agent - initial routing only. Agents can handoff directly to each other."""

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.llm import Provider, create_llm_with_structured_output
from src.prompts import SUPERVISOR_PROMPT
from src.registry import get_registry
from src.trace import get_tracer
from src.types import AgentState


def _build_route_decision_schema(agent_options: list[str]) -> type[BaseModel]:
    """Build RouteDecision schema with current agent options."""
    agent_list = ", ".join(agent_options)

    class RouteDecision(BaseModel):
        next_agent: str = Field(
            description=f"Which agent should start this task. Options: {agent_list}, or FINISH if no work needed."
        )
        reasoning: str = Field(description="Brief explanation.")

    return RouteDecision


class SupervisorAgent:
    """Routes initial task to the right agent. After that, agents handoff directly.

    Also handles escalation when agents are in a ping-pong loop.
    """

    def __init__(self, model: str | None = None, provider: Provider | None = None) -> None:
        self.name = "supervisor"
        self._model = model
        self._provider = provider

    def _create_routing_llm(self, agent_options: list[str]) -> tuple[Any, str]:
        RouteDecision = _build_route_decision_schema(agent_options)
        return create_llm_with_structured_output(
            RouteDecision,
            model=self._model,
            temperature=0.3,
            provider=self._provider,
        )

    async def route(self, state: AgentState) -> Command[Any]:
        tracer = get_tracer()
        tracer.agent_start(self.name)

        registry = get_registry()
        routable_agents = registry.get_routable_agents()

        execution_trace = state.get("execution_trace", [])
        is_escalation = len(execution_trace) > 0

        base_prompt = SUPERVISOR_PROMPT
        if is_escalation:
            recent_agents = execution_trace[-6:] if len(execution_trace) > 6 else execution_trace
            base_prompt += (
                f"\n\nIMPORTANT: This is an escalation. Agents were in a loop: {' -> '.join(recent_agents)}. "
                f"Either route to a DIFFERENT agent, provide guidance, or FINISH the task."
            )

        llm, _ = self._create_routing_llm(routable_agents)
        messages = [SystemMessage(content=base_prompt), *list(state["messages"])]
        decision = await llm.ainvoke(messages)

        valid_agents = [*routable_agents, "FINISH"]
        next_agent = decision.next_agent

        if next_agent not in valid_agents:
            tracer.error(self.name, f"Invalid agent: {next_agent}. Defaulting to FINISH.")
            next_agent = "FINISH"

        target = next_agent if next_agent != "FINISH" else END
        tracer.handoff(self.name, str(target), decision.reasoning)

        updated_trace = list(execution_trace)
        updated_trace.append(self.name)

        return Command(
            goto=target,
            update={
                "current_agent": self.name,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "execution_trace": updated_trace,
                "messages": [
                    AIMessage(content=f"[Supervisor] Routing to {next_agent}: {decision.reasoning}", name=self.name)
                ],
            },
        )
