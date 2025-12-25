"""Factory for creating custom agents quickly."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.types import AgentState


class AgentFactory:
    """Create agents with minimal boilerplate."""

    @staticmethod
    def create(
        name: str,
        system_prompt: str,
        tools: Sequence[BaseTool] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        process_fn: Callable[[AgentState], dict[str, Any]] | None = None,
    ) -> BaseAgent:
        class CustomAgent(BaseAgent):
            async def process(self, state: AgentState) -> dict[str, Any] | Command[Any]:
                if process_fn:
                    return process_fn(state)

                messages = self.get_messages_with_system(state)
                response = await self.llm.ainvoke(messages)
                return {
                    "messages": [self.create_response_message(str(response.content))],
                }

        return CustomAgent(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
            temperature=temperature,
        )


def create_custom_agent(
    name: str,
    role: str,
    capabilities: list[str],
    guidelines: list[str] | None = None,
    tools: list[BaseTool] | None = None,
) -> BaseAgent:
    """Create agent from role and capabilities."""
    prompt_parts: list[str] = [f"You are a {role}.", "", "Capabilities:"]
    prompt_parts.extend(f"- {c}" for c in capabilities)

    if guidelines:
        prompt_parts.extend(["", "Guidelines:"])
        prompt_parts.extend(f"- {g}" for g in guidelines)

    return AgentFactory.create(
        name=name,
        system_prompt="\n".join(prompt_parts),
        tools=tools,
    )


AGENT_TEMPLATES: dict[str, dict[str, str | list[str]]] = {
    "translator": {
        "role": "Expert translator",
        "capabilities": ["Translate between languages", "Preserve formatting", "Adapt tone"],
    },
    "reviewer": {
        "role": "Code reviewer",
        "capabilities": ["Review for bugs", "Check clarity", "Give feedback"],
    },
    "summarizer": {
        "role": "Summarizer",
        "capabilities": ["Summarize documents", "Extract key points"],
    },
}


def create_from_template(
    template_name: str,
    role: str | None = None,
    capabilities: list[str] | None = None,
    guidelines: list[str] | None = None,
) -> BaseAgent:
    """Create agent from pre-built template."""
    if template_name not in AGENT_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")

    template = AGENT_TEMPLATES[template_name]
    final_role = role or str(template.get("role", template_name))
    template_caps = template.get("capabilities", [])
    final_capabilities = capabilities or (list(template_caps) if isinstance(template_caps, list) else [])

    return create_custom_agent(
        name=template_name,
        role=final_role,
        capabilities=final_capabilities,
        guidelines=guidelines,
    )
