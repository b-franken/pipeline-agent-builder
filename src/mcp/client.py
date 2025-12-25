import logging
from typing import Any, Final

from langchain_core.tools import BaseTool

from src.storage.models import MCPServer

logger: Final = logging.getLogger(__name__)

_MCP_AVAILABLE: bool = False

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _MCP_AVAILABLE = True
except ImportError:
    MultiServerMCPClient = None


def is_mcp_available() -> bool:
    return _MCP_AVAILABLE


class MCPClientManager:
    def __init__(self) -> None:
        self._client: MultiServerMCPClient | None = None
        self._server_configs: dict[str, dict[str, Any]] = {}
        self._tools_cache: dict[str, list[BaseTool]] = {}

    def _build_server_config(self, server: MCPServer) -> dict[str, Any]:
        config: dict[str, Any] = {"transport": server.transport}

        if server.transport == "stdio":
            if server.command:
                config["command"] = server.command
            if server.args:
                config["args"] = server.args
            if server.env_vars:
                config["env"] = server.env_vars
        elif server.transport in ("http", "streamable_http", "sse"):
            if server.url:
                config["url"] = server.url
            if server.headers:
                config["headers"] = server.headers

        return config

    def configure_servers(self, servers: list[MCPServer]) -> None:
        if not _MCP_AVAILABLE:
            logger.warning("MCP adapters not installed. Run: pip install langchain-mcp-adapters")
            return

        self._server_configs = {
            server.name: self._build_server_config(server) for server in servers if server.is_active
        }

        self._tools_cache.clear()

        if self._server_configs:
            self._client = MultiServerMCPClient(self._server_configs)
        else:
            self._client = None

    def add_server(self, server: MCPServer) -> None:
        if not _MCP_AVAILABLE or not server.is_active:
            return

        self._server_configs[server.name] = self._build_server_config(server)
        self._tools_cache.pop(server.name, None)
        self._client = MultiServerMCPClient(self._server_configs)

    def remove_server(self, server_name: str) -> None:
        self._server_configs.pop(server_name, None)
        self._tools_cache.pop(server_name, None)

        if self._server_configs:
            self._client = MultiServerMCPClient(self._server_configs)
        else:
            self._client = None

    async def get_tools(self, server_names: list[str] | None = None) -> list[BaseTool]:
        if not _MCP_AVAILABLE or self._client is None:
            return []

        try:
            all_tools: list[BaseTool] = await self._client.get_tools()

            if server_names is None:
                return all_tools

            filtered_tools: list[BaseTool] = []
            for tool in all_tools:
                tool_server = getattr(tool, "server_name", None)
                if tool_server and tool_server in server_names:
                    filtered_tools.append(tool)

            return filtered_tools

        except Exception as e:
            logger.error(f"Failed to get MCP tools: {type(e).__name__}: {e}")
            return []

    async def get_tools_for_server(self, server_name: str) -> list[BaseTool]:
        if server_name in self._tools_cache:
            return self._tools_cache[server_name]

        tools = await self.get_tools([server_name])
        self._tools_cache[server_name] = tools
        return tools

    def get_configured_servers(self) -> list[str]:
        return list(self._server_configs.keys())

    def is_server_configured(self, server_name: str) -> bool:
        return server_name in self._server_configs

    def clear_cache(self) -> None:
        self._tools_cache.clear()


_mcp_client_manager: MCPClientManager | None = None


def get_mcp_client_manager() -> MCPClientManager:
    global _mcp_client_manager
    if _mcp_client_manager is None:
        _mcp_client_manager = MCPClientManager()
    return _mcp_client_manager


async def create_mcp_client_for_servers(servers: list[MCPServer]) -> list[BaseTool]:
    """Create a temporary MCP client for specific servers and return their tools.

    This creates an ephemeral client for only the servers needed by a specific agent,
    rather than loading all MCP servers from the database.
    """
    if not _MCP_AVAILABLE or not servers:
        return []

    try:
        server_configs: dict[str, dict[str, Any]] = {}
        for server in servers:
            if not server.is_active:
                continue

            config: dict[str, Any] = {"transport": server.transport}

            if server.transport == "stdio":
                if server.command:
                    config["command"] = server.command
                if server.args:
                    config["args"] = server.args
                if server.env_vars:
                    config["env"] = server.env_vars
            elif server.transport in ("http", "streamable_http", "sse"):
                if server.url:
                    config["url"] = server.url
                if server.headers:
                    config["headers"] = server.headers

            server_configs[server.name] = config

        if not server_configs:
            return []

        client = MultiServerMCPClient(server_configs)
        tools: list[BaseTool] = await client.get_tools()

        logger.info(f"Loaded {len(tools)} MCP tools from servers: {list(server_configs.keys())}")
        return tools

    except Exception as e:
        logger.error(f"Failed to create MCP client for servers: {type(e).__name__}: {e}")
        return []


async def initialize_mcp_from_database() -> int:
    from src.storage import get_repository

    if not _MCP_AVAILABLE:
        logger.info("MCP adapters not installed, skipping MCP initialization")
        return 0

    repo = await get_repository()
    servers = await repo.list_mcp_servers(active_only=True)

    if not servers:
        return 0

    manager = get_mcp_client_manager()
    manager.configure_servers(servers)

    logger.info(f"Initialized MCP with {len(servers)} server(s): {[s.name for s in servers]}")
    return len(servers)
