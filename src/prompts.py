from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.registry import AgentRegistry

_BASE_PROMPTS: dict[str, str] = {
    "supervisor": """You are a supervisor. Analyze the task and route to the best agent.
Choose FINISH if no work is needed.""",
    "researcher": """You are a research agent. Find accurate information.
Complete your research before handing off.""",
    "coder": """You are a coding agent. Write clean, working code.
IMPORTANT: First write the code, then use finish when done. Only hand off if you need help from another agent.""",
    "writer": """You are a writing agent. Write clear, well-structured content.
Complete your writing before using finish or handing off.""",
}


def _generate_handoff_instructions(agent_name: str, registry: AgentRegistry) -> str:
    other_agents = [name for name in registry.get_agent_names() if name != agent_name]

    if not other_agents:
        return "\n\nWhen done, use finish to complete the task."

    lines = ["\n\nAfter completing your work, use one of these tools:"]
    lines.append("- finish: Use this when the task is complete")
    for other in other_agents:
        agent_def = registry.get(other)
        if agent_def:
            lines.append(f"- handoff_to_{other}: {agent_def.handoff_description}")

    return "\n".join(lines)


def _generate_supervisor_prompt(registry: AgentRegistry) -> str:
    agents = registry.get_routable_agents()

    lines = ["You are a supervisor. Route tasks to the best agent:"]
    for name in agents:
        agent_def = registry.get(name)
        if agent_def:
            lines.append(f"- {name}: {agent_def.description}")
    lines.append("")
    lines.append("Route to an agent to complete the task. Choose FINISH only if no work is needed.")

    return "\n".join(lines)


def _load_base_prompts() -> dict[str, str]:
    base = _BASE_PROMPTS.copy()

    try:
        import config

        for agent_name in list(base.keys()):
            config_key = f"{agent_name.upper()}_PROMPT"
            if hasattr(config, config_key):
                custom_prompt: str = getattr(config, config_key)
                if custom_prompt.strip():
                    base[agent_name] = custom_prompt.strip()
    except ImportError:
        pass

    return base


_base_prompts: dict[str, str] = _load_base_prompts()


def get_prompt(agent_name: str) -> str:
    """Get prompt with dynamic handoff instructions based on registered agents."""
    from src.registry import get_registry

    registry = get_registry()

    if agent_name == "supervisor":
        return _generate_supervisor_prompt(registry)

    base = _base_prompts.get(agent_name, f"You are a {agent_name} agent.")
    handoff = _generate_handoff_instructions(agent_name, registry)
    return base + handoff


def __getattr__(name: str) -> str:
    """Dynamic attribute access for prompt constants."""
    if name in ("SUPERVISOR_PROMPT", "RESEARCHER_PROMPT", "CODER_PROMPT", "WRITER_PROMPT"):
        agent = name.replace("_PROMPT", "").lower()
        return get_prompt(agent)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
