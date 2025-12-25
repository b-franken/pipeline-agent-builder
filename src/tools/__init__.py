"""
Tools module containing reusable tools for agents.

Tools are LangChain-compatible and can be bound to any agent.
Supports a plugin system for auto-discovery of tools.
"""

from src.tools.code import create_code_tools
from src.tools.factory import create_tools_for_agent
from src.tools.knowledge import (
    StoreFactTool,
    UnifiedSearchTool,
    create_knowledge_tools,
)
from src.tools.plugins import (
    PluginRegistry,
    ToolPlugin,
    create_plugin,
    discover_plugins,
    get_plugin_registry,
    get_plugin_tools,
    register_plugin,
)
from src.tools.search import create_search_tool

__all__ = [
    "PluginRegistry",
    "StoreFactTool",
    "ToolPlugin",
    "UnifiedSearchTool",
    "create_code_tools",
    "create_knowledge_tools",
    "create_plugin",
    "create_search_tool",
    "create_tools_for_agent",
    "discover_plugins",
    "get_plugin_registry",
    "get_plugin_tools",
    "register_plugin",
]
