from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_workflow():
    workflow = MagicMock()
    workflow.ainvoke = AsyncMock(
        return_value={
            "messages": [MagicMock(content="Test response", name="coder", __class__=type("AIMessage", (), {}))],
            "current_agent": "coder",
        }
    )
    return workflow


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.broadcast = AsyncMock()
    manager.connect = AsyncMock()
    manager.disconnect = AsyncMock()
    return manager


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.list_agent_configs = AsyncMock(return_value=[])
    repo.get_agent_config = AsyncMock(return_value=None)
    repo.create_agent_config = AsyncMock()
    repo.get_setting = AsyncMock(return_value=None)
    repo.set_setting = AsyncMock()
    repo.delete_setting = AsyncMock(return_value=True)
    repo.list_documents = AsyncMock(return_value=[])
    repo.get_recent_tasks = AsyncMock(return_value=[])
    repo.clear_all = AsyncMock(return_value={"tasks": 0, "context": 0, "facts": 0, "documents": 0})
    repo.clear_tasks = AsyncMock(return_value=0)
    repo.clear_context = AsyncMock(return_value=0)
    repo.clear_facts = AsyncMock(return_value=0)
    repo.clear_documents = AsyncMock(return_value=0)
    repo.export_all = AsyncMock(
        return_value={"exported_at": "2025-01-01", "tasks": [], "context": [], "facts": [], "documents": []}
    )
    return repo


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.get_agent_names.return_value = ["supervisor", "coder", "researcher"]
    mock_agent = MagicMock()
    mock_agent.name = "coder"
    mock_agent.description = "Writes code"
    registry.get.return_value = mock_agent
    return registry


@pytest.fixture
def client(mock_workflow, mock_manager, mock_repository, mock_registry):
    with (
        patch("src.storage.get_repository", AsyncMock(return_value=mock_repository)),
        patch("src.registry.get_registry", return_value=mock_registry),
        patch("src.graph.workflow.create_workflow", return_value=mock_workflow),
    ):
        from src.api.app import create_app

        app = create_app()
        app.state.manager = mock_manager
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "checks" in data
        assert "database" in data["checks"]


class TestAgentsEndpoint:
    def test_list_agents(self, client: TestClient, mock_registry: MagicMock) -> None:
        response = client.get("/api/agents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_agent_configs(self, client: TestClient) -> None:
        response = client.get("/api/agents/config")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_agent_config_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        mock_registry.get.return_value = None
        response = client.get("/api/agents/config/nonexistent_agent")
        assert response.status_code == 404

    def test_create_agent_invalid_id(self, client: TestClient) -> None:
        response = client.post(
            "/api/agents/config",
            json={
                "id": "InvalidID",
                "name": "Test Agent",
                "description": "Test",
                "system_prompt": "You are a test agent",
            },
        )
        assert response.status_code == 422

    def test_create_agent_valid(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.id = "test_agent"
        mock_config.name = "Test Agent"
        mock_config.description = "Test description"
        mock_config.system_prompt = "You are a test"
        mock_config.role = "worker"
        mock_config.capabilities = []
        mock_config.enabled_tools = []
        mock_config.model_override = None
        mock_config.is_active = True
        mock_config.is_builtin = False
        mock_repository.create_agent_config.return_value = mock_config
        mock_repository.get_agent_config.return_value = None

        response = client.post(
            "/api/agents/config",
            json={
                "id": "test_agent",
                "name": "Test Agent",
                "description": "Test description",
                "system_prompt": "You are a test",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_agent"


class TestSettingsEndpoint:
    def test_get_settings(self, client: TestClient) -> None:
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data
        assert "model" in data
        assert "available_providers" in data

    def test_update_settings_invalid_provider(self, client: TestClient) -> None:
        response = client.put("/api/settings", json={"provider": "invalid_provider"})
        assert response.status_code == 400

    def test_update_settings_valid(self, client: TestClient, mock_repository: MagicMock) -> None:
        response = client.put("/api/settings", json={"provider": "openai", "model": "gpt-4o"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "provider" in data["updated"]
        assert "model" in data["updated"]

    def test_get_models_for_provider(self, client: TestClient) -> None:
        response = client.get("/api/settings/models/openai")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_models_for_invalid_provider(self, client: TestClient) -> None:
        response = client.get("/api/settings/models/invalid")
        assert response.status_code == 404


class TestDataEndpoints:
    def test_list_documents(self, client: TestClient) -> None:
        response = client.get("/api/data/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_export_data(self, client: TestClient) -> None:
        response = client.get("/api/data/export")
        assert response.status_code == 200
        data = response.json()
        assert "exported_at" in data
        assert "tasks" in data
        assert "context" in data

    def test_clear_data_all(self, client: TestClient) -> None:
        response = client.post("/api/data/clear", json={"target": "all"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_clear_data_tasks(self, client: TestClient) -> None:
        response = client.post("/api/data/clear", json={"target": "tasks"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_clear_data_context(self, client: TestClient) -> None:
        response = client.post("/api/data/clear", json={"target": "context"})
        assert response.status_code == 200

    def test_clear_data_facts(self, client: TestClient) -> None:
        response = client.post("/api/data/clear", json={"target": "facts"})
        assert response.status_code == 200

    def test_clear_data_documents(self, client: TestClient) -> None:
        response = client.post("/api/data/clear", json={"target": "documents"})
        assert response.status_code == 200

    def test_delete_document_not_found(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_repository.deactivate_document = AsyncMock(return_value=False)
        with patch("src.knowledge.DocumentIngester") as mock_ingester:
            mock_ingester.return_value.delete_document.return_value = False
            response = client.delete("/api/data/documents/nonexistent")
            assert response.status_code == 404


class TestToolsEndpoint:
    def test_list_available_tools(self, client: TestClient) -> None:
        response = client.get("/api/tools/available")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for tool in data:
            assert "id" in tool
            assert "name" in tool
            assert "description" in tool


class TestTaskEndpoint:
    def test_task_request_empty_task(self, client: TestClient) -> None:
        response = client.post("/api/task", json={"task": ""})
        assert response.status_code == 422

    def test_task_request_too_long(self, client: TestClient) -> None:
        response = client.post("/api/task", json={"task": "x" * 10001})
        assert response.status_code == 422


class TestImportEndpoint:
    def test_import_data(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_repository.import_all = AsyncMock(
            return_value={
                "agents": 1,
                "pipelines": 0,
                "teams": 0,
                "mcp_servers": 0,
                "context": 0,
                "facts": 0,
                "settings": 0,
            }
        )
        response = client.post(
            "/api/data/import",
            json={
                "data": {
                    "agents": [
                        {
                            "id": "test_agent",
                            "name": "Test Agent",
                            "description": "Test",
                            "system_prompt": "You are test",
                        }
                    ]
                },
                "merge": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["counts"]["agents"] == 1

    def test_import_data_empty(self, client: TestClient, mock_repository: MagicMock) -> None:
        mock_repository.import_all = AsyncMock(
            return_value={
                "agents": 0,
                "pipelines": 0,
                "teams": 0,
                "mcp_servers": 0,
                "context": 0,
                "facts": 0,
                "settings": 0,
            }
        )
        response = client.post("/api/data/import", json={"data": {}, "merge": True})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["counts"]["agents"] == 0


class TestIngestEndpoint:
    def test_ingest_without_file(self, client: TestClient) -> None:
        response = client.post("/api/data/ingest")
        assert response.status_code == 422
