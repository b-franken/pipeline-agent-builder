from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp.client import (
    MCPClientManager,
    get_mcp_client_manager,
    initialize_mcp_from_database,
    is_mcp_available,
)


class TestIsMCPAvailable:
    def test_mcp_available_when_installed(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            assert is_mcp_available() is True

    def test_mcp_not_available_when_not_installed(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", False):
            assert is_mcp_available() is False


class TestMCPClientManager:
    def test_init(self) -> None:
        manager = MCPClientManager()

        assert manager._client is None
        assert manager._server_configs == {}
        assert manager._tools_cache == {}

    def test_build_server_config_stdio(self) -> None:
        manager = MCPClientManager()
        mock_server = MagicMock()
        mock_server.transport = "stdio"
        mock_server.command = "npx"
        mock_server.args = ["-y", "@test/server"]
        mock_server.env_vars = {"API_KEY": "secret"}
        mock_server.url = None
        mock_server.headers = {}

        config = manager._build_server_config(mock_server)

        assert config["transport"] == "stdio"
        assert config["command"] == "npx"
        assert config["args"] == ["-y", "@test/server"]
        assert config["env"] == {"API_KEY": "secret"}

    def test_build_server_config_http(self) -> None:
        manager = MCPClientManager()
        mock_server = MagicMock()
        mock_server.transport = "http"
        mock_server.command = None
        mock_server.args = []
        mock_server.env_vars = {}
        mock_server.url = "https://mcp.example.com"
        mock_server.headers = {"Authorization": "Bearer token"}

        config = manager._build_server_config(mock_server)

        assert config["transport"] == "http"
        assert config["url"] == "https://mcp.example.com"
        assert config["headers"] == {"Authorization": "Bearer token"}

    def test_build_server_config_sse(self) -> None:
        manager = MCPClientManager()
        mock_server = MagicMock()
        mock_server.transport = "sse"
        mock_server.command = None
        mock_server.args = []
        mock_server.env_vars = {}
        mock_server.url = "https://sse.example.com/events"
        mock_server.headers = {}

        config = manager._build_server_config(mock_server)

        assert config["transport"] == "sse"
        assert config["url"] == "https://sse.example.com/events"

    def test_build_server_config_streamable_http(self) -> None:
        manager = MCPClientManager()
        mock_server = MagicMock()
        mock_server.transport = "streamable_http"
        mock_server.command = None
        mock_server.args = []
        mock_server.env_vars = {}
        mock_server.url = "https://stream.example.com"
        mock_server.headers = {}

        config = manager._build_server_config(mock_server)

        assert config["transport"] == "streamable_http"
        assert config["url"] == "https://stream.example.com"

    def test_configure_servers_mcp_not_available(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", False):
            manager = MCPClientManager()
            mock_server = MagicMock()
            mock_server.is_active = True

            manager.configure_servers([mock_server])

            assert manager._server_configs == {}
            assert manager._client is None

    def test_configure_servers_empty_list(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()

            manager.configure_servers([])

            assert manager._server_configs == {}
            assert manager._client is None

    def test_configure_servers_filters_inactive(self) -> None:
        with (
            patch("src.mcp.client._MCP_AVAILABLE", True),
            patch("src.mcp.client.MultiServerMCPClient"),
        ):
            manager = MCPClientManager()

            active_server = MagicMock()
            active_server.name = "active"
            active_server.is_active = True
            active_server.transport = "stdio"
            active_server.command = "npx"
            active_server.args = []
            active_server.env_vars = {}

            inactive_server = MagicMock()
            inactive_server.name = "inactive"
            inactive_server.is_active = False

            manager.configure_servers([active_server, inactive_server])

            assert "active" in manager._server_configs
            assert "inactive" not in manager._server_configs

    def test_configure_servers_clears_cache(self) -> None:
        with (
            patch("src.mcp.client._MCP_AVAILABLE", True),
            patch("src.mcp.client.MultiServerMCPClient"),
        ):
            manager = MCPClientManager()
            manager._tools_cache = {"old": [MagicMock()]}

            manager.configure_servers([])

            assert manager._tools_cache == {}

    def test_add_server_mcp_not_available(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", False):
            manager = MCPClientManager()
            mock_server = MagicMock()
            mock_server.is_active = True

            manager.add_server(mock_server)

            assert manager._server_configs == {}

    def test_add_server_inactive(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            mock_server = MagicMock()
            mock_server.is_active = False

            manager.add_server(mock_server)

            assert manager._server_configs == {}

    def test_add_server_success(self) -> None:
        with (
            patch("src.mcp.client._MCP_AVAILABLE", True),
            patch("src.mcp.client.MultiServerMCPClient") as mock_client_class,
        ):
            manager = MCPClientManager()
            mock_server = MagicMock()
            mock_server.name = "github"
            mock_server.is_active = True
            mock_server.transport = "stdio"
            mock_server.command = "npx"
            mock_server.args = ["-y", "@mcp/github"]
            mock_server.env_vars = {}

            manager.add_server(mock_server)

            assert "github" in manager._server_configs
            mock_client_class.assert_called()

    def test_add_server_clears_cache_for_server(self) -> None:
        with (
            patch("src.mcp.client._MCP_AVAILABLE", True),
            patch("src.mcp.client.MultiServerMCPClient"),
        ):
            manager = MCPClientManager()
            manager._tools_cache = {"github": [MagicMock()], "other": [MagicMock()]}

            mock_server = MagicMock()
            mock_server.name = "github"
            mock_server.is_active = True
            mock_server.transport = "stdio"
            mock_server.command = "npx"
            mock_server.args = []
            mock_server.env_vars = {}

            manager.add_server(mock_server)

            assert "github" not in manager._tools_cache
            assert "other" in manager._tools_cache

    def test_remove_server(self) -> None:
        manager = MCPClientManager()
        manager._server_configs = {"server1": {}, "server2": {}}
        manager._tools_cache = {"server1": [MagicMock()]}

        with patch("src.mcp.client.MultiServerMCPClient"):
            manager.remove_server("server1")

        assert "server1" not in manager._server_configs
        assert "server1" not in manager._tools_cache
        assert "server2" in manager._server_configs

    def test_remove_server_last_one(self) -> None:
        manager = MCPClientManager()
        manager._server_configs = {"only_server": {}}
        manager._client = MagicMock()

        manager.remove_server("only_server")

        assert manager._server_configs == {}
        assert manager._client is None

    def test_remove_server_nonexistent(self) -> None:
        with patch("src.mcp.client.MultiServerMCPClient"):
            manager = MCPClientManager()
            manager._server_configs = {"existing": {}}

            manager.remove_server("nonexistent")

            assert "existing" in manager._server_configs

    @pytest.mark.asyncio
    async def test_get_tools_mcp_not_available(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", False):
            manager = MCPClientManager()

            tools = await manager.get_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_no_client(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            manager._client = None

            tools = await manager.get_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_all(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            mock_client = MagicMock()
            mock_tool1 = MagicMock()
            mock_tool1.server_name = "server1"
            mock_tool2 = MagicMock()
            mock_tool2.server_name = "server2"
            mock_client.get_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])
            manager._client = mock_client

            tools = await manager.get_tools()

            assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_get_tools_filtered_by_server(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            mock_client = MagicMock()
            mock_tool1 = MagicMock()
            mock_tool1.server_name = "server1"
            mock_tool2 = MagicMock()
            mock_tool2.server_name = "server2"
            mock_client.get_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])
            manager._client = mock_client

            tools = await manager.get_tools(["server1"])

            assert len(tools) == 1
            assert tools[0].server_name == "server1"

    @pytest.mark.asyncio
    async def test_get_tools_handles_exception(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(side_effect=Exception("Connection failed"))
            manager._client = mock_client

            tools = await manager.get_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_for_server_cached(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            cached_tools = [MagicMock()]
            manager._tools_cache = {"github": cached_tools}

            tools = await manager.get_tools_for_server("github")

            assert tools == cached_tools

    @pytest.mark.asyncio
    async def test_get_tools_for_server_fetches_and_caches(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            mock_client = MagicMock()
            mock_tool = MagicMock()
            mock_tool.server_name = "github"
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])
            manager._client = mock_client

            tools = await manager.get_tools_for_server("github")

            assert len(tools) == 1
            assert "github" in manager._tools_cache

    def test_get_configured_servers(self) -> None:
        manager = MCPClientManager()
        manager._server_configs = {"server1": {}, "server2": {}, "server3": {}}

        servers = manager.get_configured_servers()

        assert set(servers) == {"server1", "server2", "server3"}

    def test_get_configured_servers_empty(self) -> None:
        manager = MCPClientManager()

        servers = manager.get_configured_servers()

        assert servers == []

    def test_is_server_configured_true(self) -> None:
        manager = MCPClientManager()
        manager._server_configs = {"github": {}}

        assert manager.is_server_configured("github") is True

    def test_is_server_configured_false(self) -> None:
        manager = MCPClientManager()
        manager._server_configs = {"github": {}}

        assert manager.is_server_configured("slack") is False

    def test_clear_cache(self) -> None:
        manager = MCPClientManager()
        manager._tools_cache = {
            "server1": [MagicMock()],
            "server2": [MagicMock()],
        }

        manager.clear_cache()

        assert manager._tools_cache == {}


class TestGetMCPClientManager:
    def test_returns_singleton(self) -> None:
        import src.mcp.client as mcp_module

        mcp_module._mcp_client_manager = None

        manager1 = get_mcp_client_manager()
        manager2 = get_mcp_client_manager()

        assert manager1 is manager2

    def test_creates_new_manager_if_none(self) -> None:
        import src.mcp.client as mcp_module

        mcp_module._mcp_client_manager = None

        manager = get_mcp_client_manager()

        assert manager is not None
        assert isinstance(manager, MCPClientManager)


class TestInitializeMCPFromDatabase:
    @pytest.mark.asyncio
    async def test_mcp_not_available(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", False):
            count = await initialize_mcp_from_database()

            assert count == 0

    @pytest.mark.asyncio
    async def test_no_servers_in_database(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            mock_repo = MagicMock()
            mock_repo.list_mcp_servers = AsyncMock(return_value=[])

            with patch("src.storage.get_repository", AsyncMock(return_value=mock_repo)):
                count = await initialize_mcp_from_database()

            assert count == 0

    @pytest.mark.asyncio
    async def test_initializes_servers(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            mock_server1 = MagicMock()
            mock_server1.name = "server1"
            mock_server1.is_active = True
            mock_server1.transport = "stdio"
            mock_server1.command = "npx"
            mock_server1.args = []
            mock_server1.env_vars = {}

            mock_server2 = MagicMock()
            mock_server2.name = "server2"
            mock_server2.is_active = True
            mock_server2.transport = "http"
            mock_server2.url = "https://example.com"
            mock_server2.headers = {}

            mock_repo = MagicMock()
            mock_repo.list_mcp_servers = AsyncMock(return_value=[mock_server1, mock_server2])

            mock_manager = MagicMock()
            mock_manager.configure_servers = MagicMock()

            with (
                patch("src.storage.get_repository", AsyncMock(return_value=mock_repo)),
                patch("src.mcp.client.MultiServerMCPClient"),
                patch("src.mcp.client.get_mcp_client_manager", return_value=mock_manager),
            ):
                count = await initialize_mcp_from_database()

            assert count == 2
            mock_manager.configure_servers.assert_called_once_with([mock_server1, mock_server2])


class TestMCPClientManagerToolFiltering:
    @pytest.mark.asyncio
    async def test_get_tools_excludes_tools_without_server_name(self) -> None:
        """Tools without server_name attribute should be excluded when filtering for security."""
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            mock_client = MagicMock()

            tool_with_server = MagicMock()
            tool_with_server.server_name = "server1"

            tool_without_server = MagicMock(spec=[])

            mock_client.get_tools = AsyncMock(return_value=[tool_with_server, tool_without_server])
            manager._client = mock_client

            tools = await manager.get_tools(["server1"])

            assert len(tools) == 1
            assert tools[0] == tool_with_server

    @pytest.mark.asyncio
    async def test_get_tools_multiple_servers(self) -> None:
        with patch("src.mcp.client._MCP_AVAILABLE", True):
            manager = MCPClientManager()
            mock_client = MagicMock()

            tool1 = MagicMock()
            tool1.server_name = "github"
            tool2 = MagicMock()
            tool2.server_name = "slack"
            tool3 = MagicMock()
            tool3.server_name = "filesystem"

            mock_client.get_tools = AsyncMock(return_value=[tool1, tool2, tool3])
            manager._client = mock_client

            tools = await manager.get_tools(["github", "slack"])

            assert len(tools) == 2
            server_names = [t.server_name for t in tools]
            assert "github" in server_names
            assert "slack" in server_names
            assert "filesystem" not in server_names
