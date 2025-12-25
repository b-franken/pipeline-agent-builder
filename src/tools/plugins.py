import importlib
import importlib.util
import logging
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

from langchain_core.tools import BaseTool

logger: Final = logging.getLogger(__name__)


class ToolPlugin(ABC):
    name: str = "unnamed_plugin"
    description: str = ""
    version: str = "1.0.0"
    enabled: bool = True

    @abstractmethod
    def create_tools(self) -> list[BaseTool]: ...

    def initialize(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, ToolPlugin] = {}
        self._tools_cache: dict[str, list[BaseTool]] = {}

    def register(self, plugin: ToolPlugin) -> None:
        if not plugin.enabled:
            return

        self._plugins[plugin.name] = plugin
        plugin.initialize()
        self._tools_cache.pop(plugin.name, None)

    def unregister(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.cleanup()
            self._tools_cache.pop(name, None)

    def get_plugin(self, name: str) -> ToolPlugin | None:
        return self._plugins.get(name)

    def get_all_plugins(self) -> list[ToolPlugin]:
        return list(self._plugins.values())

    def get_tools(self, plugin_name: str) -> list[BaseTool]:
        if plugin_name not in self._tools_cache:
            plugin = self._plugins.get(plugin_name)
            if plugin:
                self._tools_cache[plugin_name] = plugin.create_tools()
            else:
                return []
        return self._tools_cache[plugin_name]

    def get_all_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for plugin_name in self._plugins:
            tools.extend(self.get_tools(plugin_name))
        return tools

    def discover(self, directory: str | Path) -> list[str]:
        directory = Path(directory)
        if not directory.exists():
            logger.debug(f"Plugin directory does not exist: {directory}")
            return []

        loaded: list[str] = []

        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                plugin = self._load_plugin_from_file(file_path)
                if plugin:
                    self.register(plugin)
                    loaded.append(plugin.name)
                    logger.info(f"Loaded plugin '{plugin.name}' from {file_path.name}")
            except Exception as e:
                logger.error(
                    f"Failed to load plugin from {file_path}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                )
                continue

        return loaded

    def _load_plugin_from_file(self, file_path: Path) -> ToolPlugin | None:
        module_name = f"plugin_{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            logger.warning(f"Could not create module spec for {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(
                f"Failed to execute plugin module {file_path}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            return None

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, ToolPlugin) and attr is not ToolPlugin:
                return attr()

        if hasattr(module, "get_plugin"):
            result = module.get_plugin()
            if isinstance(result, ToolPlugin):
                return result

        logger.debug(f"No ToolPlugin subclass found in {file_path}")
        return None


_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    return _registry


def register_plugin(plugin: ToolPlugin) -> None:
    _registry.register(plugin)


def discover_plugins(directory: str | Path) -> list[BaseTool]:
    _registry.discover(directory)
    return _registry.get_all_tools()


def get_plugin_tools(plugin_name: str) -> list[BaseTool]:
    return _registry.get_tools(plugin_name)


class SearchPlugin(ToolPlugin):
    name = "search"
    description = "Web search tools"

    def create_tools(self) -> list[BaseTool]:
        from src.tools.search import create_search_tool

        return [create_search_tool()]


class CodePlugin(ToolPlugin):
    name = "code"
    description = "Code analysis and formatting tools"

    def create_tools(self) -> list[BaseTool]:
        from src.tools.code import create_code_tools

        return list(create_code_tools())


class MemoryPlugin(ToolPlugin):
    name = "memory"
    description = "Long-term memory storage and search"

    def create_tools(self) -> list[BaseTool]:
        try:
            from src.memory import create_memory_tools

            return create_memory_tools()
        except ImportError:
            return []


class KnowledgePlugin(ToolPlugin):
    name = "knowledge"
    description = "Knowledge base search, unified search, and fact storage"

    def create_tools(self) -> list[BaseTool]:
        from src.knowledge.retriever import KnowledgeSearchTool
        from src.tools.knowledge import StoreFactTool, UnifiedSearchTool

        return [
            UnifiedSearchTool(),
            KnowledgeSearchTool(),
            StoreFactTool(),
        ]


class DocumentPlugin(ToolPlugin):
    name = "documents"
    description = "Document generation tools for PDF, Excel, and CSV"

    def create_tools(self) -> list[BaseTool]:
        try:
            from src.tools.documents import create_document_tools

            return create_document_tools()
        except ImportError:
            logger.debug("Document tools dependencies not installed (weasyprint, xlsxwriter)")
            return []


class MCPDynamicPlugin(ToolPlugin):
    name = "mcp"
    description = "Model Context Protocol tools from configured servers"
    _server_ids: list[str]
    _cached_tools: list[BaseTool] | None

    def __init__(self, server_ids: list[str] | None = None) -> None:
        self._server_ids = server_ids or []
        self._cached_tools = None

    def set_server_ids(self, server_ids: list[str]) -> None:
        self._server_ids = server_ids
        self._cached_tools = None

    def create_tools(self) -> list[BaseTool]:
        if self._cached_tools is not None:
            return self._cached_tools
        return []

    async def load_tools_async(self) -> list[BaseTool]:
        if not self._server_ids:
            return []

        try:
            from src.mcp.tools import get_mcp_tools_for_servers

            self._cached_tools = await get_mcp_tools_for_servers(self._server_ids)
            return self._cached_tools
        except ImportError:
            logger.debug("MCP adapters not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {type(e).__name__}: {e}")
            return []


def _register_builtin_plugins() -> None:
    register_plugin(SearchPlugin())
    register_plugin(CodePlugin())
    register_plugin(MemoryPlugin())
    register_plugin(KnowledgePlugin())
    register_plugin(DocumentPlugin())


_register_builtin_plugins()


def create_plugin(
    name: str,
    tools: list[BaseTool],
    description: str = "",
) -> ToolPlugin:
    class SimplePlugin(ToolPlugin):
        def __init__(self, plugin_name: str, plugin_tools: list[BaseTool], plugin_desc: str) -> None:
            self.name = plugin_name
            self.description = plugin_desc
            self._tools = plugin_tools

        def create_tools(self) -> list[BaseTool]:
            return self._tools

    return SimplePlugin(name, tools, description)
