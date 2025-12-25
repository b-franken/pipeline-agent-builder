"""Type definitions for the multi-agent system."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _last_value(existing: Any, new: Any) -> Any:
    """Reducer that always takes the newest value."""
    return new


def _max_iteration(existing: int, new: int) -> int:
    """Reducer that takes the max iteration count when merging state updates."""
    return max(existing, new)


def _merge_context(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges context dictionaries."""
    result = dict(existing) if existing else {}
    result.update(new or {})
    return result


def _extend_trace(existing: list[str], new: list[str]) -> list[str]:
    """Reducer that extends the execution trace with new entries."""
    return existing + [e for e in new if e not in existing[-len(new) :]] if existing else new


def _merge_loop_counts(existing: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """Reducer that merges loop counts, taking max for each edge."""
    result = dict(existing) if existing else {}
    for edge_id, count in (new or {}).items():
        result[edge_id] = max(result.get(edge_id, 0), count)
    return result


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: Annotated[str, _last_value]
    context: Annotated[dict[str, Any], _merge_context]
    human_feedback: Annotated[str | None, _last_value]
    iteration_count: Annotated[int, _max_iteration]
    execution_trace: Annotated[list[str], _extend_trace]
    loop_counts: Annotated[dict[str, int], _merge_loop_counts]
