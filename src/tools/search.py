"""Search tools for agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from src.config import settings

if TYPE_CHECKING:
    SearchFunc = Callable[[str], str]


def _get_tavily_key() -> str:
    return settings.tavily_api_key


def _create_search_function() -> tuple[Callable[[str], str], str, str]:
    tavily_key = _get_tavily_key()

    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            tavily_tool = TavilySearchResults(max_results=5)

            def _tavily_search(query: str) -> str:
                results = tavily_tool.invoke(query)
                return str(results)

            return _tavily_search, "tavily_search", "Search the web for current information using Tavily."

        except ImportError:
            pass

    def _no_search_configured(query: str) -> str:
        return (
            f"[Search unavailable] No search API configured. "
            f"Set TAVILY_API_KEY environment variable to enable web search. "
            f"Query was: {query}"
        )

    return _no_search_configured, "web_search", "Search the web (requires TAVILY_API_KEY)."


_search_func, _search_name, _search_desc = _create_search_function()

web_search = StructuredTool.from_function(
    func=_search_func,
    name=_search_name,
    description=_search_desc,
)


def create_search_tool() -> StructuredTool:
    return web_search


def create_tavily_search_tool(max_results: int = 5) -> StructuredTool:
    tavily_key = _get_tavily_key()

    if not tavily_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except ImportError as e:
        raise ImportError(
            "langchain-community is required. Install with: pip install langchain-community tavily-python"
        ) from e

    tavily_tool = TavilySearchResults(max_results=max_results)

    def _tavily_invoke(query: str) -> str:
        return str(tavily_tool.invoke(query))

    return StructuredTool.from_function(
        func=_tavily_invoke,
        name="tavily_search",
        description="Search the web for current information using Tavily.",
    )


def is_search_configured() -> bool:
    return bool(_get_tavily_key())
