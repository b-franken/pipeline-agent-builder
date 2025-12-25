"""Agent registry for dynamic agent management and handoff tool generation."""

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.tools import StructuredTool
from langgraph.graph import END
from langgraph.types import Command

logger: Final = logging.getLogger(__name__)

_sync_lock = threading.Lock()
_async_lock: asyncio.Lock | None = None


def _get_async_lock() -> asyncio.Lock:
    global _async_lock
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    return _async_lock


@dataclass
class AgentDefinition:
    """Agent metadata for registry."""

    name: str
    description: str
    handoff_description: str
    factory: Callable[..., Any] | None = None
    routable: bool = True
    can_handoff: bool = True


_workflow_invalidation_callback: Callable[[], None] | None = None


def set_workflow_invalidation_callback(callback: Callable[[], None] | None) -> None:
    """Set callback to invalidate workflow when registry changes."""
    global _workflow_invalidation_callback
    _workflow_invalidation_callback = callback


def _notify_workflow_change() -> None:
    """Notify that registry changed and workflow needs rebuild."""
    if _workflow_invalidation_callback is not None:
        _workflow_invalidation_callback()


class AgentRegistry:
    """Central registry for agents. Generates handoff tools automatically."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}
        self._handoff_tools: dict[str, StructuredTool] = {}
        self._db_synced: bool = False

    def register(self, agent: AgentDefinition, notify: bool = True) -> None:
        self._agents[agent.name] = agent
        if agent.can_handoff:
            self._handoff_tools[agent.name] = self._create_handoff_tool(agent)
        if notify:
            _notify_workflow_change()

    def unregister(self, name: str, notify: bool = True) -> None:
        self._agents.pop(name, None)
        self._handoff_tools.pop(name, None)
        if notify:
            _notify_workflow_change()

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def get_agent_names(self, routable_only: bool = False) -> list[str]:
        if routable_only:
            return [name for name, agent in self._agents.items() if agent.routable]
        return list(self._agents.keys())

    def get_routable_agents(self) -> list[str]:
        return [name for name, agent in self._agents.items() if agent.routable]

    def get_handoff_tools(self, for_agent: str) -> list[StructuredTool]:
        """Returns handoff tools for all agents except the requesting one, plus finish."""
        tools = [tool for name, tool in self._handoff_tools.items() if name != for_agent]
        tools.append(self._create_finish_tool())
        return tools

    def create_agent(self, name: str, can_handoff: bool = True, **kwargs: Any) -> Any:
        agent_def = self._agents.get(name)
        if not agent_def:
            raise ValueError(f"Agent '{name}' not registered")
        if not agent_def.factory:
            raise ValueError(f"Agent '{name}' has no factory function")
        return agent_def.factory(can_handoff=can_handoff, **kwargs)

    def _create_handoff_tool(self, agent: AgentDefinition) -> StructuredTool:
        def handoff() -> Command[Any]:
            return Command(goto=agent.name)

        return StructuredTool.from_function(
            func=handoff,
            name=f"handoff_to_{agent.name}",
            description=f"Hand off to {agent.name}. {agent.handoff_description}",
        )

    def _create_finish_tool(self) -> StructuredTool:
        def finish() -> Command[Any]:
            return Command(goto=END)

        return StructuredTool.from_function(
            func=finish,
            name="finish",
            description="Complete the task.",
        )


_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    """Get global registry, registering defaults on first access."""
    _register_default_agents()
    return _registry


def register_agent(agent: AgentDefinition) -> None:
    _registry.register(agent)


_defaults_registered = False


def _register_default_agents() -> None:
    """Register default agents. Uses lazy imports to avoid circular deps."""
    global _defaults_registered
    with _sync_lock:
        if _defaults_registered:
            return
        _defaults_registered = True

    def _create_researcher(**kwargs: Any) -> Any:
        from src.agents.workers.researcher import ResearcherAgent

        return ResearcherAgent(**kwargs)

    def _create_coder(**kwargs: Any) -> Any:
        from src.agents.workers.coder import CoderAgent

        return CoderAgent(**kwargs)

    def _create_writer(**kwargs: Any) -> Any:
        from src.agents.workers.writer import WriterAgent

        return WriterAgent(**kwargs)

    def _create_assistant(**kwargs: Any) -> Any:
        from src.agents.base import BaseAgent

        return BaseAgent(
            name="assistant",
            system_prompt="You are a helpful assistant. Answer questions directly and concisely.",
            **kwargs,
        )

    register_agent(
        AgentDefinition(
            name="assistant",
            description="General assistant for simple questions and tasks",
            handoff_description="Use for simple questions, greetings, or general help.",
            factory=_create_assistant,
        )
    )

    register_agent(
        AgentDefinition(
            name="researcher",
            description="Research agent that finds and synthesizes information",
            handoff_description="Use when you need information gathered or research done.",
            factory=_create_researcher,
        )
    )

    register_agent(
        AgentDefinition(
            name="coder",
            description="Coding agent that writes and reviews code",
            handoff_description="Use when you need code written, reviewed, or debugged.",
            factory=_create_coder,
        )
    )

    register_agent(
        AgentDefinition(
            name="writer",
            description="Writing agent that creates and edits content",
            handoff_description="Use when you need content written, edited, or summarized.",
            factory=_create_writer,
        )
    )

    def _create_reflective_writer(**kwargs: Any) -> Any:
        from src.agents.reflective import ReflectiveAgent

        return ReflectiveAgent(
            name="reflective_writer",
            system_prompt="You are a careful writer who produces high-quality, well-structured content.",
            max_refinements=2,
            **kwargs,
        )

    register_agent(
        AgentDefinition(
            name="reflective_writer",
            description="Self-critiquing writer that refines output for higher quality (slower)",
            handoff_description="Use for important content needing careful review and refinement.",
            factory=_create_reflective_writer,
            routable=True,
        )
    )


def ensure_defaults_registered() -> None:
    _register_default_agents()


async def sync_agents_from_database() -> int:
    """Load custom agents from database into registry.

    Returns the number of agents synced.
    """
    from src.storage import get_repository
    from src.tools.factory import create_tools_for_agent_async

    async with _get_async_lock():
        if _registry._db_synced:
            logger.debug("Database sync already completed, skipping")
            return 0

        try:
            repo = await get_repository()
            configs = await repo.list_agent_configs(active_only=True)
            logger.info("Found %d agent configs in database", len(configs))
        except Exception:
            logger.exception("Failed to load agent configs from database")
            return 0

        synced = 0
        for cfg in configs:
            if cfg.id in _registry._agents:
                continue

            agent_id = cfg.id
            agent_name = cfg.name
            agent_desc = cfg.description
            agent_prompt = cfg.system_prompt
            agent_model = cfg.model_override
            agent_active = cfg.is_active
            agent_tools = cfg.enabled_tools or []
            agent_mcp_servers = cfg.mcp_server_ids or []

            preloaded_tools = await create_tools_for_agent_async(
                agent_id=agent_id,
                enabled_tool_ids=agent_tools,
                mcp_server_ids=agent_mcp_servers if agent_mcp_servers else None,
            )

            def _make_factory(
                config_id: str,
                prompt: str,
                model: str | None,
                base_tools: list[Any],
            ) -> Callable[..., Any]:
                def _factory(**kwargs: Any) -> Any:
                    from src.agents.base import BaseAgent

                    extra_tools = kwargs.pop("tools", None) or []
                    all_tools = list(base_tools) + list(extra_tools) if base_tools or extra_tools else None

                    return BaseAgent(
                        name=config_id,
                        system_prompt=prompt,
                        model=model,
                        tools=all_tools,
                        **kwargs,
                    )

                return _factory

            _registry.register(
                AgentDefinition(
                    name=agent_id,
                    description=agent_desc,
                    handoff_description=f"Hand off to {agent_name} for {agent_desc.lower()}",
                    factory=_make_factory(agent_id, agent_prompt, agent_model, preloaded_tools),
                    routable=agent_active,
                    can_handoff=True,
                ),
                notify=False,
            )
            synced += 1

        _registry._db_synced = True
        if synced > 0:
            builtin_agents = {"assistant", "researcher", "coder", "writer", "reflective_writer"}
            custom_agents = [c.id for c in configs if c.id not in builtin_agents]
            logger.info("Synced %d agents from database: %s", synced, custom_agents)
            _notify_workflow_change()
        else:
            logger.info("No new agents to sync from database")

        return synced


async def register_dynamic_agent_async(
    agent_id: str,
    name: str,
    description: str,
    system_prompt: str,
    model_override: str | None = None,
    enabled_tools: list[str] | None = None,
    mcp_server_ids: list[str] | None = None,
) -> None:
    """Register a dynamic agent from API/database with MCP tool support."""
    from src.agents.base import BaseAgent
    from src.tools.factory import create_tools_for_agent_async

    tool_ids = enabled_tools or []

    preloaded_tools = await create_tools_for_agent_async(
        agent_id=agent_id,
        enabled_tool_ids=tool_ids,
        mcp_server_ids=mcp_server_ids,
    )

    def _factory(**kwargs: Any) -> BaseAgent:
        extra_tools = kwargs.pop("tools", None) or []
        all_tools = list(preloaded_tools) + list(extra_tools) if preloaded_tools or extra_tools else None

        return BaseAgent(
            name=agent_id,
            system_prompt=system_prompt,
            model=model_override,
            tools=all_tools,
            **kwargs,
        )

    _registry.register(
        AgentDefinition(
            name=agent_id,
            description=description,
            handoff_description=f"Hand off to {name} for {description.lower()}",
            factory=_factory,
            routable=True,
            can_handoff=True,
        ),
        notify=True,
    )


def register_dynamic_agent(
    agent_id: str,
    name: str,
    description: str,
    system_prompt: str,
    model_override: str | None = None,
    enabled_tools: list[str] | None = None,
    mcp_server_ids: list[str] | None = None,
) -> None:
    """Register a dynamic agent (sync version, MCP tools not loaded).

    For MCP tool support, use register_dynamic_agent_async instead.
    """
    from src.agents.base import BaseAgent
    from src.tools.factory import create_tools_for_agent

    tool_ids = enabled_tools or []

    def _factory(**kwargs: Any) -> BaseAgent:
        base_tools = create_tools_for_agent(
            enabled_tool_ids=tool_ids,
            source_agent=agent_id,
        )
        extra_tools = kwargs.pop("tools", None) or []
        all_tools = list(base_tools) + list(extra_tools) if base_tools or extra_tools else None

        return BaseAgent(
            name=agent_id,
            system_prompt=system_prompt,
            model=model_override,
            tools=all_tools,
            **kwargs,
        )

    _registry.register(
        AgentDefinition(
            name=agent_id,
            description=description,
            handoff_description=f"Hand off to {name} for {description.lower()}",
            factory=_factory,
            routable=True,
            can_handoff=True,
        ),
        notify=True,
    )


def unregister_dynamic_agent(agent_id: str) -> bool:
    """Unregister a dynamic agent. Returns True if agent was found and removed."""
    if agent_id not in _registry._agents:
        return False

    agent_def = _registry._agents.get(agent_id)
    if agent_def is None:
        return False

    _registry.unregister(agent_id, notify=True)
    return True


def mark_registry_dirty() -> None:
    """Mark registry as needing re-sync from database."""
    _registry._db_synced = False
