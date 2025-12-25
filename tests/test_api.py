import pytest
from pydantic import ValidationError

from src.api.routes import (
    AVAILABLE_MODELS,
    AgentConfigResponse,
    AgentInfo,
    ClearRequest,
    ClearResponse,
    CreateAgentRequest,
    DocumentInfo,
    IngestResponse,
    SettingsResponse,
    TaskRequest,
    TaskResponse,
    UpdateAgentRequest,
    UpdateSettingsRequest,
)


class TestTaskModels:
    def test_task_request_valid(self) -> None:
        req = TaskRequest(task="Test task")
        assert req.task == "Test task"
        assert req.thread_id is None

    def test_task_request_with_thread_id(self) -> None:
        req = TaskRequest(task="Test", thread_id="thread-123")
        assert req.thread_id == "thread-123"

    def test_task_request_empty_task_fails(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(task="")

    def test_task_request_max_length(self) -> None:
        long_task = "x" * 10001
        with pytest.raises(ValidationError):
            TaskRequest(task=long_task)

    def test_task_response_valid(self) -> None:
        resp = TaskResponse(
            task_id="abc123",
            result="Done",
            agent="coder",
            messages=[{"role": "human", "content": "Hi", "name": None}],
        )
        assert resp.task_id == "abc123"
        assert resp.result == "Done"
        assert resp.agent == "coder"

    def test_agent_info_valid(self) -> None:
        info = AgentInfo(name="Coder", description="Writes code")
        assert info.name == "Coder"
        assert info.description == "Writes code"
        assert info.status == "idle"


class TestClearModels:
    def test_clear_request_defaults(self) -> None:
        req = ClearRequest()
        assert req.target == "all"
        assert req.category is None
        assert req.hard_delete is False

    def test_clear_request_specific_target(self) -> None:
        req = ClearRequest(target="tasks", hard_delete=True)
        assert req.target == "tasks"
        assert req.hard_delete is True

    def test_clear_request_invalid_target(self) -> None:
        with pytest.raises(ValidationError):
            ClearRequest(target="invalid")  # type: ignore[arg-type]

    def test_clear_response_valid(self) -> None:
        resp = ClearResponse(
            success=True,
            message="Cleared 5 items",
            counts={"tasks": 3, "facts": 2},
        )
        assert resp.success is True
        assert resp.counts["tasks"] == 3


class TestDocumentModels:
    def test_document_info_valid(self) -> None:
        info = DocumentInfo(
            id="doc123",
            filename="test.pdf",
            content_type="application/pdf",
            chunk_count=5,
            is_active=True,
        )
        assert info.id == "doc123"
        assert info.chunk_count == 5

    def test_ingest_response_valid(self) -> None:
        resp = IngestResponse(
            success=True,
            document_id="doc123",
            filename="test.pdf",
            chunks_created=5,
            message="Success",
        )
        assert resp.success is True
        assert resp.chunks_created == 5


class TestSettingsModels:
    def test_settings_response_valid(self) -> None:
        resp = SettingsResponse(
            provider="openai",
            model="gpt-4o",
            available_providers=["openai", "anthropic"],
            available_models=[{"id": "gpt-4o", "name": "GPT-4o"}],
            has_openai_key=True,
            has_anthropic_key=False,
            has_google_key=False,
            ollama_host="http://localhost:11434",
        )
        assert resp.provider == "openai"
        assert resp.has_openai_key is True

    def test_update_settings_request_empty(self) -> None:
        req = UpdateSettingsRequest()
        assert req.provider is None
        assert req.model is None

    def test_update_settings_request_partial(self) -> None:
        req = UpdateSettingsRequest(provider="anthropic", model="claude-3-5-sonnet")
        assert req.provider == "anthropic"
        assert req.model == "claude-3-5-sonnet"


class TestAgentConfigModels:
    def test_agent_config_response_valid(self) -> None:
        resp = AgentConfigResponse(
            id="coder",
            name="Coder",
            description="Writes code",
            system_prompt="You are a coder",
            role="worker",
            capabilities=["python", "javascript"],
            enabled_tools=["execute_code"],
            model_override=None,
            is_active=True,
            is_builtin=True,
        )
        assert resp.id == "coder"
        assert resp.capabilities == ["python", "javascript"]

    def test_create_agent_request_valid(self) -> None:
        req = CreateAgentRequest(
            id="custom_agent",
            name="Custom Agent",
            description="A custom agent",
            system_prompt="You are custom",
        )
        assert req.id == "custom_agent"
        assert req.role == "worker"
        assert req.capabilities == []

    def test_create_agent_request_invalid_id_uppercase(self) -> None:
        with pytest.raises(ValidationError):
            CreateAgentRequest(
                id="CustomAgent",
                name="Custom",
                description="Desc",
                system_prompt="Prompt",
            )

    def test_create_agent_request_invalid_id_starts_with_number(self) -> None:
        with pytest.raises(ValidationError):
            CreateAgentRequest(
                id="1agent",
                name="Agent",
                description="Desc",
                system_prompt="Prompt",
            )

    def test_create_agent_request_valid_id_with_underscore(self) -> None:
        req = CreateAgentRequest(
            id="custom_agent_v2",
            name="Custom Agent V2",
            description="Desc",
            system_prompt="Prompt",
        )
        assert req.id == "custom_agent_v2"

    def test_update_agent_request_empty(self) -> None:
        req = UpdateAgentRequest()
        assert req.name is None
        assert req.system_prompt is None

    def test_update_agent_request_partial(self) -> None:
        req = UpdateAgentRequest(
            name="New Name",
            is_active=False,
        )
        assert req.name == "New Name"
        assert req.is_active is False


class TestAvailableModels:
    def test_openai_models_exist(self) -> None:
        assert "openai" in AVAILABLE_MODELS
        models = AVAILABLE_MODELS["openai"]
        model_ids = [m["id"] for m in models]
        assert "gpt-4o" in model_ids
        assert "gpt-4o-mini" in model_ids

    def test_anthropic_models_exist(self) -> None:
        assert "anthropic" in AVAILABLE_MODELS
        models = AVAILABLE_MODELS["anthropic"]
        model_ids = [m["id"] for m in models]
        assert "claude-sonnet-4-20250514" in model_ids

    def test_google_models_exist(self) -> None:
        assert "google" in AVAILABLE_MODELS
        models = AVAILABLE_MODELS["google"]
        model_ids = [m["id"] for m in models]
        assert "gemini-2.0-flash" in model_ids

    def test_ollama_models_exist(self) -> None:
        assert "ollama" in AVAILABLE_MODELS
        models = AVAILABLE_MODELS["ollama"]
        model_ids = [m["id"] for m in models]
        assert "llama3.3" in model_ids

    def test_all_providers_have_models(self) -> None:
        for provider, models in AVAILABLE_MODELS.items():
            assert len(models) > 0, f"Provider {provider} has no models"
            for model in models:
                assert "id" in model
                assert "name" in model
