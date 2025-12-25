"""LangGraph workflow with direct agent-to-agent handoffs."""

import asyncio
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.supervisor import SupervisorAgent
from src.llm import Provider
from src.memory.checkpointer import create_async_checkpointer, create_checkpointer
from src.registry import get_registry
from src.types import AgentState

_workflow_cache: dict[str, CompiledStateGraph[Any]] = {}
_async_workflow_cache: dict[str, CompiledStateGraph[Any]] = {}
_default_workflow: CompiledStateGraph[Any] | None = None
_async_workflow_lock: asyncio.Lock | None = None


def _get_async_lock() -> asyncio.Lock:
    global _async_workflow_lock
    if _async_workflow_lock is None:
        _async_workflow_lock = asyncio.Lock()
    return _async_workflow_lock


def _make_cache_key(
    use_persistence: bool,
    supervisor_model: str | None,
    supervisor_provider: Provider | None,
    agent_configs: dict[str, dict[str, Any]] | None,
) -> str:
    import hashlib
    import json

    config = {
        "persistence": use_persistence,
        "supervisor_model": supervisor_model,
        "supervisor_provider": str(supervisor_provider) if supervisor_provider else None,
        "agent_configs": agent_configs or {},
    }
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def create_workflow(
    use_persistence: bool = True,
    supervisor_model: str | None = None,
    supervisor_provider: Provider | None = None,
    agent_configs: dict[str, dict[str, Any]] | None = None,
    force_new: bool = False,
) -> CompiledStateGraph[Any]:
    global _workflow_cache

    cache_key = _make_cache_key(use_persistence, supervisor_model, supervisor_provider, agent_configs)

    if not force_new and cache_key in _workflow_cache:
        return _workflow_cache[cache_key]

    registry = get_registry()
    agent_configs = agent_configs or {}

    supervisor = SupervisorAgent(model=supervisor_model, provider=supervisor_provider)

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor.route)
    graph.add_edge(START, "supervisor")

    for agent_name in registry.get_agent_names():
        config = agent_configs.get(agent_name, {})
        agent = registry.create_agent(agent_name, **config)
        graph.add_node(agent_name, agent.invoke)
        graph.add_edge(agent_name, END)

    checkpointer = create_checkpointer() if use_persistence else create_checkpointer(backend="memory")

    compiled = graph.compile(checkpointer=checkpointer)
    _workflow_cache[cache_key] = compiled

    return compiled


def create_simple_workflow() -> CompiledStateGraph[Any]:
    return create_workflow(use_persistence=False)


def clear_workflow_cache() -> None:
    global _workflow_cache, _async_workflow_cache, _default_workflow
    _workflow_cache.clear()
    _async_workflow_cache.clear()
    _default_workflow = None


def get_cached_workflow_count() -> int:
    return len(_workflow_cache)


def invalidate_workflow() -> None:
    """Invalidate the default workflow, forcing rebuild on next access."""
    global _default_workflow
    _default_workflow = None
    clear_workflow_cache()


def get_workflow() -> CompiledStateGraph[Any]:
    global _default_workflow
    if _default_workflow is None:
        _default_workflow = create_workflow(use_persistence=True, force_new=True)
    return _default_workflow


def set_workflow(workflow: CompiledStateGraph[Any]) -> None:
    """Set the default workflow explicitly."""
    global _default_workflow
    _default_workflow = workflow


async def create_async_workflow(
    use_persistence: bool = True,
    supervisor_model: str | None = None,
    supervisor_provider: Provider | None = None,
    agent_configs: dict[str, dict[str, Any]] | None = None,
    force_new: bool = False,
) -> CompiledStateGraph[Any]:
    """Create workflow with async checkpointer for use with ainvoke."""
    global _async_workflow_cache

    cache_key = _make_cache_key(use_persistence, supervisor_model, supervisor_provider, agent_configs)

    if not force_new and cache_key in _async_workflow_cache:
        return _async_workflow_cache[cache_key]

    async with _get_async_lock():
        if not force_new and cache_key in _async_workflow_cache:
            return _async_workflow_cache[cache_key]

        registry = get_registry()
        agent_configs = agent_configs or {}

        supervisor = SupervisorAgent(model=supervisor_model, provider=supervisor_provider)

        graph: StateGraph[AgentState] = StateGraph(AgentState)
        graph.add_node("supervisor", supervisor.route)
        graph.add_edge(START, "supervisor")

        for agent_name in registry.get_agent_names():
            config = agent_configs.get(agent_name, {})
            agent = registry.create_agent(agent_name, **config)
            graph.add_node(agent_name, agent.invoke)
            graph.add_edge(agent_name, END)

        if use_persistence:
            checkpointer = await create_async_checkpointer()
        else:
            checkpointer = await create_async_checkpointer(backend="memory")

        compiled = graph.compile(checkpointer=checkpointer)
        _async_workflow_cache[cache_key] = compiled

        return compiled


async def get_async_workflow() -> CompiledStateGraph[Any]:
    """Get the default async workflow, creating it if needed."""
    return await create_async_workflow(use_persistence=True, force_new=False)
