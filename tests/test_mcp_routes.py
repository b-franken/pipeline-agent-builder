from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.list_mcp_servers = AsyncMock(return_value=[])
    repo.get_mcp_server = AsyncMock(return_value=None)
    repo.get_mcp_server_by_name = AsyncMock(return_value=None)
    repo.create_mcp_server = AsyncMock()
    repo.update_mcp_server = AsyncMock(return_value=None)
    repo.delete_mcp_server = AsyncMock(return_value=False)
    return repo


@pytest.fixture
def mock_mcp_manager():
    manager = MagicMock()
    manager.get_configured_servers.return_value = []
    manager.get_tools = AsyncMock(return_value=[])
    manager.get_tools_for_server = AsyncMock(return_value=[])
    manager.is_server_configured.return_value = False
    manager.add_server = MagicMock()
    manager.remove_server = MagicMock()
    manager.configure_servers = MagicMock()
    return manager


@pytest.fixture
def client(mock_repository, mock_mcp_manager):
    from src.api.mcp_routes import router as mcp_router

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncGenerator[None]:
        yield

    app = FastAPI(lifespan=mock_lifespan)
    app.include_router(mcp_router, prefix="/api")

    with (
        patch("src.api.mcp_routes.get_repository", AsyncMock(return_value=mock_repository)),
        patch("src.api.mcp_routes.get_mcp_client_manager", return_value=mock_mcp_manager),
        patch("src.api.mcp_routes.is_mcp_available", return_value=True),
        TestClient(app) as test_client,
    ):
        yield test_client


class TestMCPStatus:
    def test_get_mcp_status_available(self, client: TestClient, mock_mcp_manager: MagicMock) -> None:
        mock_mcp_manager.get_configured_servers.return_value = ["server1", "server2"]
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_mcp_manager.get_tools = AsyncMock(return_value=[mock_tool])

        response = client.get("/api/mcp/status")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert "configured_servers" in data
        assert "total_tools" in data

    def test_get_mcp_status_no_servers(self, client: TestClient, mock_mcp_manager: MagicMock) -> None:
        mock_mcp_manager.get_configured_servers.return_value = []

        response = client.get("/api/mcp/status")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["configured_servers"] == []
        assert data["total_tools"] == 0


class TestMCPServersCRUD:
    def test_list_mcp_servers_empty(self, client: TestClient) -> None:
        response = client.get("/api/mcp/servers")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_mcp_servers_with_data(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-123"
        mock_server.name = "test-server"
        mock_server.transport = "stdio"
        mock_server.description = "Test server"
        mock_server.command = "npx"
        mock_server.args = ["-y", "@test/server"]
        mock_server.url = None
        mock_server.headers = {}
        mock_server.is_active = True
        mock_repository.list_mcp_servers = AsyncMock(return_value=[mock_server])

        response = client.get("/api/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test-server"
        assert data[0]["transport"] == "stdio"

    def test_create_mcp_server_stdio(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_server = MagicMock()
        mock_server.id = "new-server-id"
        mock_server.name = "github"
        mock_server.transport = "stdio"
        mock_server.description = "GitHub MCP server"
        mock_server.command = "npx"
        mock_server.args = ["-y", "@modelcontextprotocol/server-github"]
        mock_server.url = None
        mock_server.headers = {}
        mock_server.is_active = True
        mock_repository.create_mcp_server = AsyncMock(return_value=mock_server)

        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "github",
                "transport": "stdio",
                "description": "GitHub MCP server",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "github"
        assert data["transport"] == "stdio"

    def test_create_mcp_server_stdio_without_command(self, client: TestClient) -> None:
        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "invalid-server",
                "transport": "stdio",
            },
        )

        assert response.status_code == 400
        assert "command" in response.json()["detail"].lower()

    def test_create_mcp_server_http_without_url(self, client: TestClient) -> None:
        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "invalid-http",
                "transport": "http",
            },
        )

        assert response.status_code == 400
        assert "url" in response.json()["detail"].lower()

    def test_create_mcp_server_duplicate_name(self, client: TestClient, mock_repository: MagicMock) -> None:
        existing_server = MagicMock()
        existing_server.name = "existing"
        mock_repository.get_mcp_server_by_name = AsyncMock(return_value=existing_server)

        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "existing",
                "transport": "stdio",
                "command": "npx",
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_create_mcp_server_invalid_transport(self, client: TestClient) -> None:
        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "test",
                "transport": "invalid_transport",
            },
        )

        assert response.status_code == 422

    def test_get_mcp_server_not_found(self, client: TestClient) -> None:
        response = client.get("/api/mcp/servers/nonexistent-id")

        assert response.status_code == 404

    def test_get_mcp_server_found(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-123"
        mock_server.name = "test-server"
        mock_server.transport = "stdio"
        mock_server.description = ""
        mock_server.command = "npx"
        mock_server.args = []
        mock_server.url = None
        mock_server.headers = {}
        mock_server.is_active = True
        mock_repository.get_mcp_server = AsyncMock(return_value=mock_server)

        response = client.get("/api/mcp/servers/server-123")

        assert response.status_code == 200
        assert response.json()["id"] == "server-123"

    def test_update_mcp_server_not_found(self, client: TestClient) -> None:
        response = client.patch(
            "/api/mcp/servers/nonexistent",
            json={"description": "Updated"},
        )

        assert response.status_code == 404

    def test_update_mcp_server_success(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        updated_server = MagicMock()
        updated_server.id = "server-123"
        updated_server.name = "updated-server"
        updated_server.transport = "stdio"
        updated_server.description = "Updated description"
        updated_server.command = "npx"
        updated_server.args = []
        updated_server.url = None
        updated_server.headers = {}
        updated_server.is_active = True
        mock_repository.update_mcp_server = AsyncMock(return_value=updated_server)

        response = client.patch(
            "/api/mcp/servers/server-123",
            json={"description": "Updated description"},
        )

        assert response.status_code == 200
        assert response.json()["description"] == "Updated description"

    def test_update_mcp_server_deactivate(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        updated_server = MagicMock()
        updated_server.id = "server-123"
        updated_server.name = "test-server"
        updated_server.transport = "stdio"
        updated_server.description = ""
        updated_server.command = "npx"
        updated_server.args = []
        updated_server.url = None
        updated_server.headers = {}
        updated_server.is_active = False
        mock_repository.update_mcp_server = AsyncMock(return_value=updated_server)

        response = client.patch(
            "/api/mcp/servers/server-123",
            json={"is_active": False},
        )

        assert response.status_code == 200
        assert response.json()["is_active"] is False
        mock_mcp_manager.remove_server.assert_called_once_with("test-server")

    def test_delete_mcp_server_not_found(self, client: TestClient) -> None:
        response = client.delete("/api/mcp/servers/nonexistent")

        assert response.status_code == 404

    def test_delete_mcp_server_success(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        mock_server = MagicMock()
        mock_server.name = "test-server"
        mock_repository.get_mcp_server = AsyncMock(return_value=mock_server)
        mock_repository.delete_mcp_server = AsyncMock(return_value=True)

        response = client.delete("/api/mcp/servers/server-123")

        assert response.status_code == 204
        mock_mcp_manager.remove_server.assert_called_once_with("test-server")


class TestMCPServerTools:
    def test_get_server_tools_not_available(self, client: TestClient) -> None:
        with patch("src.api.mcp_routes.is_mcp_available", return_value=False):
            response = client.get("/api/mcp/servers/server-123/tools")

        assert response.status_code == 503
        assert "not installed" in response.json()["detail"].lower()

    def test_get_server_tools_server_not_found(self, client: TestClient) -> None:
        response = client.get("/api/mcp/servers/nonexistent/tools")

        assert response.status_code == 404

    def test_get_server_tools_success(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-123"
        mock_server.name = "github"
        mock_repository.get_mcp_server = AsyncMock(return_value=mock_server)
        mock_mcp_manager.is_server_configured.return_value = True

        mock_tool1 = MagicMock()
        mock_tool1.name = "create_issue"
        mock_tool1.description = "Create a GitHub issue"
        mock_tool2 = MagicMock()
        mock_tool2.name = "list_repos"
        mock_tool2.description = "List repositories"
        mock_mcp_manager.get_tools_for_server = AsyncMock(return_value=[mock_tool1, mock_tool2])

        response = client.get("/api/mcp/servers/server-123/tools")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "create_issue"
        assert data[1]["name"] == "list_repos"

    def test_get_server_tools_error(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-123"
        mock_server.name = "broken-server"
        mock_repository.get_mcp_server = AsyncMock(return_value=mock_server)
        mock_mcp_manager.is_server_configured.return_value = True
        mock_mcp_manager.get_tools_for_server = AsyncMock(side_effect=Exception("Connection failed"))

        response = client.get("/api/mcp/servers/server-123/tools")

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to get tools from MCP server"


class TestMCPServerTest:
    def test_test_mcp_server_not_available(self, client: TestClient) -> None:
        with patch("src.api.mcp_routes.is_mcp_available", return_value=False):
            response = client.post("/api/mcp/servers/server-123/test")

        assert response.status_code == 503

    def test_test_mcp_server_not_found(self, client: TestClient) -> None:
        response = client.post("/api/mcp/servers/nonexistent/test")

        assert response.status_code == 404

    def test_test_mcp_server_success(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-123"
        mock_server.name = "test-server"
        mock_repository.get_mcp_server = AsyncMock(return_value=mock_server)
        mock_mcp_manager.is_server_configured.return_value = True

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_mcp_manager.get_tools_for_server = AsyncMock(return_value=[mock_tool])

        response = client.post("/api/mcp/servers/server-123/test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tool_count"] == 1
        assert "test_tool" in data["tools"]

    def test_test_mcp_server_failure(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-123"
        mock_server.name = "failing-server"
        mock_repository.get_mcp_server = AsyncMock(return_value=mock_server)
        mock_mcp_manager.is_server_configured.return_value = True
        mock_mcp_manager.get_tools_for_server = AsyncMock(side_effect=ConnectionError("Failed to connect"))

        response = client.post("/api/mcp/servers/server-123/test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "Failed to connect to MCP server"


class TestMCPRegistryPopular:
    def test_get_popular_servers_all(self, client: TestClient) -> None:
        response = client.get("/api/mcp/registry/popular")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        for server in data:
            assert "id" in server
            assert "name" in server
            assert "description" in server
            assert "package" in server
            assert "transport" in server
            assert "command" in server
            assert "args" in server
            assert "env_vars" in server
            assert "category" in server
            assert "icon" in server

    def test_get_popular_servers_by_category(self, client: TestClient) -> None:
        response = client.get("/api/mcp/registry/popular?category=database")

        assert response.status_code == 200
        data = response.json()
        assert all(s["category"] == "database" for s in data)

    def test_get_popular_servers_unknown_category(self, client: TestClient) -> None:
        response = client.get("/api/mcp/registry/popular?category=nonexistent")

        assert response.status_code == 200
        assert response.json() == []

    def test_popular_servers_contain_expected_entries(self, client: TestClient) -> None:
        response = client.get("/api/mcp/registry/popular")
        data = response.json()

        server_ids = [s["id"] for s in data]
        assert "github" in server_ids
        assert "filesystem" in server_ids
        assert "postgres" in server_ids
        assert "memory" in server_ids


class TestMCPRegistryCategories:
    def test_get_categories(self, client: TestClient) -> None:
        response = client.get("/api/mcp/registry/categories")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        for cat in data:
            assert "id" in cat
            assert "name" in cat
            assert "count" in cat
            assert cat["count"] > 0

    def test_categories_include_expected_values(self, client: TestClient) -> None:
        response = client.get("/api/mcp/registry/categories")
        data = response.json()

        category_ids = [c["id"] for c in data]
        assert "developer" in category_ids
        assert "database" in category_ids
        assert "web" in category_ids


class TestMCPRegistryFetch:
    def test_get_registry_success(self, client: TestClient) -> None:
        mock_response_data = {
            "servers": [
                {
                    "name": "test-server",
                    "description": "A test server",
                    "version": "1.0.0",
                    "packages": [{"registryType": "npm", "name": "@test/server", "version": "1.0.0"}],
                    "remotes": [],
                }
            ]
        }

        mock_return = {**mock_response_data, "cached": False, "cache_age_seconds": 0}
        with patch("src.api.mcp_routes._fetch_registry", AsyncMock(return_value=mock_return)):
            response = client.get("/api/mcp/registry")

        assert response.status_code == 200
        data = response.json()
        assert "servers" in data
        assert "total" in data
        assert "cached" in data

    def test_get_registry_with_search(self, client: TestClient) -> None:
        mock_response_data = {"servers": [{"name": "github-server", "description": "GitHub integration"}]}

        mock_return = {**mock_response_data, "cached": False, "cache_age_seconds": 0}
        with patch("src.api.mcp_routes._fetch_registry", AsyncMock(return_value=mock_return)):
            response = client.get("/api/mcp/registry?search=github")

        assert response.status_code == 200

    def test_get_registry_with_limit(self, client: TestClient) -> None:
        mock_return = {"servers": [], "cached": False, "cache_age_seconds": 0}
        with patch("src.api.mcp_routes._fetch_registry", AsyncMock(return_value=mock_return)):
            response = client.get("/api/mcp/registry?limit=10")

        assert response.status_code == 200

    def test_get_registry_limit_validation(self, client: TestClient) -> None:
        response = client.get("/api/mcp/registry?limit=0")
        assert response.status_code == 422

        response = client.get("/api/mcp/registry?limit=101")
        assert response.status_code == 422


class TestMCPRegistryCaching:
    @pytest.mark.asyncio
    async def test_fetch_registry_caching(self) -> None:
        """Test that registry responses are cached properly."""
        import src.api.mcp_routes as mcp_routes
        from src.api.mcp_routes import _fetch_registry

        mcp_routes._registry_cache = {}
        mcp_routes._registry_cache_time = None

        mock_response = MagicMock()
        mock_response.json.return_value = {"servers": [{"name": "test"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result1 = await _fetch_registry()
            assert result1["cached"] is False

            result2 = await _fetch_registry()
            assert result2["cached"] is True


class TestMCPRoutesEdgeCases:
    def test_create_server_with_env_vars(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-with-env"
        mock_server.name = "brave-search"
        mock_server.transport = "stdio"
        mock_server.description = ""
        mock_server.command = "npx"
        mock_server.args = ["-y", "@modelcontextprotocol/server-brave-search"]
        mock_server.url = None
        mock_server.headers = {}
        mock_server.is_active = True
        mock_repository.create_mcp_server = AsyncMock(return_value=mock_server)

        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "brave-search",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "env_vars": {"BRAVE_API_KEY": "test-key"},
            },
        )

        assert response.status_code == 201

    def test_create_server_http_transport(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_server = MagicMock()
        mock_server.id = "http-server"
        mock_server.name = "http-mcp"
        mock_server.transport = "http"
        mock_server.description = ""
        mock_server.command = None
        mock_server.args = []
        mock_server.url = "https://mcp.example.com"
        mock_server.headers = {"Authorization": "Bearer token"}
        mock_server.is_active = True
        mock_repository.create_mcp_server = AsyncMock(return_value=mock_server)

        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "http-mcp",
                "transport": "http",
                "url": "https://mcp.example.com",
                "headers": {"Authorization": "Bearer token"},
            },
        )

        assert response.status_code == 201

    def test_create_server_sse_transport(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_server = MagicMock()
        mock_server.id = "sse-server"
        mock_server.name = "sse-mcp"
        mock_server.transport = "sse"
        mock_server.description = ""
        mock_server.command = None
        mock_server.args = []
        mock_server.url = "https://sse.example.com/events"
        mock_server.headers = {}
        mock_server.is_active = True
        mock_repository.create_mcp_server = AsyncMock(return_value=mock_server)

        response = client.post(
            "/api/mcp/servers",
            json={
                "name": "sse-mcp",
                "transport": "sse",
                "url": "https://sse.example.com/events",
            },
        )

        assert response.status_code == 201

    def test_server_tools_adds_unconfigured_server(
        self, client: TestClient, mock_repository: MagicMock, mock_mcp_manager: MagicMock
    ) -> None:
        mock_server = MagicMock()
        mock_server.id = "server-123"
        mock_server.name = "new-server"
        mock_repository.get_mcp_server = AsyncMock(return_value=mock_server)
        mock_mcp_manager.is_server_configured.return_value = False
        mock_mcp_manager.get_tools_for_server = AsyncMock(return_value=[])

        response = client.get("/api/mcp/servers/server-123/tools")

        assert response.status_code == 200
        mock_mcp_manager.add_server.assert_called_once_with(mock_server)
