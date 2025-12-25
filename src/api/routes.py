import logging
import uuid
from typing import Final, Literal

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import settings as app_settings
from src.trace import reset_tracer
from src.types import AgentState

logger: Final = logging.getLogger(__name__)


def _fix_document_links(content: str) -> str:
    """Fix document download links in response content.

    Transforms various incorrect link formats to proper /api/documents/download/ URLs:
    - sandbox:/api/documents/download/file.pdf -> /api/documents/download/file.pdf
    - sandbox:/app/data/documents/file.pdf -> /api/documents/download/file.pdf
    - https://*/api/documents/download/file.pdf -> /api/documents/download/file.pdf
    - /app/data/documents/file.pdf -> /api/documents/download/file.pdf
    """
    import re

    doc_extensions = r"[\w\-_]+\.(pdf|xlsx|csv)"

    content = re.sub(
        rf"sandbox:/(?:api/documents/download/|app/data/documents/)({doc_extensions})",
        r"/api/documents/download/\1",
        content,
    )

    content = re.sub(
        rf"https?://[^/]+/api/documents/download/({doc_extensions})", r"/api/documents/download/\1", content
    )

    content = re.sub(rf"/app/data/documents/({doc_extensions})", r"/api/documents/download/\1", content)

    return content


router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

AVAILABLE_MODELS: Final[dict[str, list[dict[str, str]]]] = {
    "openai": [
        {"id": "gpt-4o", "name": "GPT-4o (recommended)"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini (fast)"},
        {"id": "o1", "name": "o1 (reasoning)"},
        {"id": "o1-mini", "name": "o1 Mini (reasoning, fast)"},
    ],
    "anthropic": [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4 (recommended)"},
        {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku (fast)"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
    ],
    "google": [
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (recommended)"},
        {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite (fast)"},
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
    ],
    "ollama": [
        {"id": "llama3.3", "name": "Llama 3.3 70B (recommended)"},
        {"id": "llama3.2", "name": "Llama 3.2"},
        {"id": "qwen2.5", "name": "Qwen 2.5"},
        {"id": "mistral", "name": "Mistral"},
        {"id": "mixtral", "name": "Mixtral 8x7B"},
        {"id": "deepseek-r1", "name": "DeepSeek R1 (reasoning)"},
        {"id": "phi4", "name": "Phi-4"},
    ],
}


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=10000)
    thread_id: str | None = None
    pipeline_id: str | None = None


class TaskResponse(BaseModel):
    task_id: str
    result: str
    agent: str
    messages: list[dict[str, str | None]]


class AgentInfo(BaseModel):
    name: str
    description: str
    status: str = "idle"


@router.post("/task", response_model=TaskResponse)
@limiter.limit("10/minute")
async def run_task(request: Request, body: TaskRequest) -> TaskResponse:
    manager = request.app.state.manager
    task_id = str(uuid.uuid4())[:8]
    thread_id = body.thread_id or task_id

    tracer = reset_tracer(task_id=task_id, thread_id=thread_id)
    tracer.set_broadcast(manager.broadcast)

    if body.pipeline_id:
        from src.graph.pipeline_builder import PipelineValidationError, get_pipeline_workflow

        try:
            pipeline_workflow = await get_pipeline_workflow(body.pipeline_id)
        except PipelineValidationError as e:
            raise HTTPException(status_code=400, detail=f"Pipeline validation failed: {'; '.join(e.errors)}") from None

        if not pipeline_workflow:
            raise HTTPException(status_code=404, detail=f"Pipeline not found: {body.pipeline_id}")
        workflow = pipeline_workflow
    else:
        workflow = await request.app.state.get_async_workflow()

    await manager.broadcast(
        {
            "type": "task_start",
            "task_id": task_id,
            "agent": "system",
            "message": f"Task started: {body.task[:100]}",
        }
    )

    initial_state: AgentState = {
        "messages": [HumanMessage(content=body.task)],
        "current_agent": "supervisor",
        "context": {"task_id": task_id},
        "human_feedback": None,
        "iteration_count": 0,
        "execution_trace": [],
        "loop_counts": {},
    }

    result = await workflow.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result.get("messages", [])
    response_content = ""
    response_agent = "agent"

    for msg in reversed(messages):
        if not msg.content:
            continue
        content = str(msg.content)
        if content.startswith("[Supervisor]"):
            continue
        if isinstance(msg, (AIMessage, ToolMessage)):
            response_content = content
            response_agent = getattr(msg, "name", None) or "agent"
            break

    response_content = _fix_document_links(response_content)

    await manager.broadcast(
        {
            "type": "task_end",
            "task_id": task_id,
            "agent": "system",
            "message": "Task completed",
            "result": response_content[:200],
        }
    )

    return TaskResponse(
        task_id=task_id,
        result=response_content,
        agent=response_agent,
        messages=[
            {
                "role": type(m).__name__.replace("Message", "").lower(),
                "content": _fix_document_links(str(m.content)) if m.content else None,
                "name": getattr(m, "name", None),
            }
            for m in messages
            if m.content
        ],
    )


@router.get("/agents", response_model=list[AgentInfo])
async def list_agents() -> list[AgentInfo]:
    from src.registry import get_registry
    from src.storage import get_repository

    registry = get_registry()
    repo = await get_repository()

    stored_configs = {a.id: a for a in await repo.list_agent_configs(active_only=False)}

    agents: list[AgentInfo] = []
    for name in registry.get_agent_names():
        agent_def = registry.get(name)
        if agent_def:
            if name in stored_configs:
                cfg = stored_configs[name]
                agents.append(
                    AgentInfo(
                        name=cfg.name,
                        description=cfg.description,
                    )
                )
            else:
                agents.append(
                    AgentInfo(
                        name=agent_def.name.replace("_", " ").title(),
                        description=agent_def.description,
                    )
                )

    for cfg in stored_configs.values():
        if cfg.id not in registry.get_agent_names() and cfg.is_active:
            agents.append(
                AgentInfo(
                    name=cfg.name,
                    description=cfg.description,
                )
            )

    return agents


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    from src.registry import get_registry
    from src.storage import get_repository

    manager = websocket.app.state.manager
    await manager.connect(websocket, client_id)

    repo = await get_repository()
    registry = get_registry()

    agent_names = registry.get_agent_names()
    agents = []
    for name in agent_names:
        agent_def = registry.get(name)
        if agent_def:
            agents.append({"id": agent_def.name, "description": agent_def.description})

    pipelines = await repo.list_pipelines(active_only=True)
    pipeline_list = [{"id": p.id, "name": p.name, "is_default": p.is_default} for p in pipelines]

    recent_tasks = await repo.get_recent_tasks(limit=10)
    task_list = []
    for t in recent_tasks:
        created_at = t.created_at.isoformat() if t.created_at else None
        task_list.append({"id": t.id, "status": t.status, "created_at": created_at})

    await websocket.send_json(
        {
            "type": "snapshot",
            "agents": agents,
            "pipelines": pipeline_list,
            "tasks": task_list,
        }
    )

    try:
        while True:
            _ = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "message": "connected"})
    except WebSocketDisconnect:
        await manager.disconnect(client_id)


class ClearRequest(BaseModel):
    target: Literal["all", "tasks", "context", "facts", "documents", "knowledge"] = "all"
    category: str | None = None
    hard_delete: bool = False


class ClearResponse(BaseModel):
    success: bool
    message: str
    counts: dict[str, int]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_type: str
    chunk_count: int
    is_active: bool


class IngestResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    chunks_created: int
    message: str


@router.post("/data/clear", response_model=ClearResponse)
async def clear_data(body: ClearRequest) -> ClearResponse:
    from src.knowledge import DocumentIngester
    from src.storage import get_repository

    repo = await get_repository()
    counts: dict[str, int] = {}

    try:
        if body.target == "all":
            counts = await repo.clear_all(hard_delete=body.hard_delete)
            try:
                ingester = DocumentIngester()
                kb_count = ingester.clear_all()
                counts["knowledge_base"] = kb_count
            except Exception:
                counts["knowledge_base"] = 0

        elif body.target == "tasks":
            counts["tasks"] = await repo.clear_tasks()

        elif body.target == "context":
            counts["context"] = await repo.clear_context(category=body.category)

        elif body.target == "facts":
            counts["facts"] = await repo.clear_facts(
                category=body.category,
                hard_delete=body.hard_delete,
            )

        elif body.target == "documents":
            counts["documents"] = await repo.clear_documents(hard_delete=body.hard_delete)

        elif body.target == "knowledge":
            ingester = DocumentIngester()
            counts["knowledge_base"] = ingester.clear_all()
            counts["documents"] = await repo.clear_documents(hard_delete=body.hard_delete)

        total = sum(counts.values())
        return ClearResponse(
            success=True,
            message=f"Cleared {total} items",
            counts=counts,
        )

    except Exception as e:
        logger.exception("Error clearing data")
        return ClearResponse(
            success=False,
            message=f"Error clearing data: {e!s}",
            counts=counts,
        )


@router.get("/data/export")
async def export_data() -> JSONResponse:
    from src.storage import get_repository

    repo = await get_repository()
    data = await repo.export_all()

    return JSONResponse(content=data)


class ImportRequest(BaseModel):
    data: dict[str, object]
    merge: bool = True


class ImportResponse(BaseModel):
    success: bool
    counts: dict[str, int]


@router.post("/data/import", response_model=ImportResponse)
async def import_data(body: ImportRequest) -> ImportResponse:
    from src.registry import mark_registry_dirty
    from src.storage import get_repository

    repo = await get_repository()
    counts = await repo.import_all(body.data, merge=body.merge)
    mark_registry_dirty()

    return ImportResponse(success=True, counts=counts)


@router.get("/data/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    from src.storage import get_repository

    repo = await get_repository()
    docs = await repo.list_documents()

    return [
        DocumentInfo(
            id=d.id,
            filename=d.filename,
            content_type=d.content_type,
            chunk_count=d.chunk_count,
            is_active=d.is_active,
        )
        for d in docs
    ]


@router.post("/data/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    path: str | None = Form(None),
) -> IngestResponse:
    from src.knowledge import DocumentIngester
    from src.storage import get_repository

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    display_filename = path or file.filename

    try:
        content = await file.read()
        ingester = DocumentIngester()
        result = ingester.ingest_upload(
            file_data=content,
            filename=display_filename,
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Ingestion failed")

        repo = await get_repository()
        content_type = file.content_type or "application/octet-stream"
        content_hash = str(result.metadata.get("content_hash", ""))

        await repo.register_document(
            doc_id=result.document_id,
            filename=display_filename,
            content_type=content_type,
            content_hash=content_hash,
            chunk_count=result.chunks_created,
            total_chars=result.total_tokens_estimate * 4,
        )

        return IngestResponse(
            success=True,
            document_id=result.document_id,
            filename=display_filename,
            chunks_created=result.chunks_created,
            message=f"Successfully ingested {display_filename}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!s}") from e


@router.delete("/data/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict[str, str | bool]:
    from src.knowledge import DocumentIngester
    from src.storage import get_repository

    ingester = DocumentIngester()
    repo = await get_repository()

    kb_deleted = ingester.delete_document(doc_id)
    meta_deleted = await repo.deactivate_document(doc_id)

    if not kb_deleted and not meta_deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "success": True,
        "document_id": doc_id,
        "message": "Document deleted",
    }


class SettingsResponse(BaseModel):
    provider: str
    model: str
    available_providers: list[str]
    available_models: list[dict[str, str]]
    has_openai_key: bool
    has_anthropic_key: bool
    has_google_key: bool
    ollama_host: str


class UpdateSettingsRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    ollama_host: str | None = None


@router.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    from src.storage import get_repository

    repo = await get_repository()

    provider = await repo.get_setting("provider") or app_settings.provider
    model = await repo.get_setting("model") or app_settings.default_model

    return SettingsResponse(
        provider=provider,
        model=model,
        available_providers=["openai", "anthropic", "google", "ollama"],
        available_models=AVAILABLE_MODELS.get(provider, []),
        has_openai_key=bool(await repo.get_setting("openai_api_key") or app_settings.openai_api_key),
        has_anthropic_key=bool(await repo.get_setting("anthropic_api_key") or app_settings.anthropic_api_key),
        has_google_key=bool(await repo.get_setting("google_api_key") or app_settings.google_api_key),
        ollama_host=await repo.get_setting("ollama_host") or app_settings.ollama_base_url,
    )


@router.put("/settings")
async def update_settings(body: UpdateSettingsRequest) -> dict[str, str | bool | list[str]]:
    from src.graph.workflow import invalidate_workflow
    from src.llm import sync_runtime_settings
    from src.storage import get_repository

    repo = await get_repository()
    updated: list[str] = []

    if body.provider:
        if body.provider not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {body.provider}")
        await repo.set_setting("provider", body.provider)
        updated.append("provider")

    if body.model:
        await repo.set_setting("model", body.model)
        updated.append("model")

    if body.openai_api_key is not None:
        if body.openai_api_key:
            await repo.set_setting("openai_api_key", body.openai_api_key, encrypted=True)
        else:
            await repo.delete_setting("openai_api_key")
        updated.append("openai_api_key")

    if body.anthropic_api_key is not None:
        if body.anthropic_api_key:
            await repo.set_setting("anthropic_api_key", body.anthropic_api_key, encrypted=True)
        else:
            await repo.delete_setting("anthropic_api_key")
        updated.append("anthropic_api_key")

    if body.google_api_key is not None:
        if body.google_api_key:
            await repo.set_setting("google_api_key", body.google_api_key, encrypted=True)
        else:
            await repo.delete_setting("google_api_key")
        updated.append("google_api_key")

    if body.ollama_host:
        await repo.set_setting("ollama_host", body.ollama_host)
        updated.append("ollama_host")

    if updated:
        await sync_runtime_settings()
        invalidate_workflow()

    return {
        "success": True,
        "updated": updated,
        "message": f"Updated {len(updated)} setting(s)",
    }


@router.get("/settings/models/{provider}")
async def get_models_for_provider(provider: str) -> list[dict[str, str]]:
    if provider not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    return AVAILABLE_MODELS[provider]


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    role: str
    capabilities: list[str]
    enabled_tools: list[str]
    mcp_server_ids: list[str] = Field(default_factory=list)
    model_override: str | None
    is_active: bool
    is_builtin: bool


class CreateAgentRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(..., min_length=1, max_length=100)
    description: str
    system_prompt: str
    role: str = "worker"
    capabilities: list[str] = Field(default_factory=list)
    enabled_tools: list[str] = Field(default_factory=list)
    model_override: str | None = None
    mcp_server_ids: list[str] = Field(default_factory=list)


class UpdateAgentRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    capabilities: list[str] | None = None
    enabled_tools: list[str] | None = None
    model_override: str | None = None
    mcp_server_ids: list[str] | None = None
    is_active: bool | None = None


@router.get("/agents/config", response_model=list[AgentConfigResponse])
async def list_agent_configs() -> list[AgentConfigResponse]:
    from src.registry import get_registry
    from src.storage import get_repository

    repo = await get_repository()
    registry = get_registry()

    stored_configs = {a.id: a for a in await repo.list_agent_configs(active_only=False)}

    agents: list[AgentConfigResponse] = []
    default_builtin_ids = {"assistant", "researcher", "coder", "writer", "reflective_writer"}

    for name in registry.get_agent_names():
        agent_def = registry.get(name)
        if not agent_def:
            continue

        if name in stored_configs:
            cfg = stored_configs[name]
            is_builtin = cfg.is_builtin or name in default_builtin_ids
            agents.append(
                AgentConfigResponse(
                    id=cfg.id,
                    name=cfg.name,
                    description=cfg.description,
                    system_prompt=cfg.system_prompt,
                    role=cfg.role,
                    capabilities=cfg.capabilities,
                    enabled_tools=cfg.enabled_tools,
                    mcp_server_ids=cfg.mcp_server_ids or [],
                    model_override=cfg.model_override,
                    is_active=cfg.is_active,
                    is_builtin=is_builtin,
                )
            )
        else:
            agents.append(
                AgentConfigResponse(
                    id=name,
                    name=agent_def.name.replace("_", " ").title(),
                    description=agent_def.description,
                    system_prompt="",
                    role="supervisor" if name == "supervisor" else "worker",
                    capabilities=[],
                    enabled_tools=[],
                    mcp_server_ids=[],
                    model_override=None,
                    is_active=True,
                    is_builtin=name in default_builtin_ids,
                )
            )

    for cfg in stored_configs.values():
        if cfg.id not in registry.get_agent_names():
            agents.append(
                AgentConfigResponse(
                    id=cfg.id,
                    name=cfg.name,
                    description=cfg.description,
                    system_prompt=cfg.system_prompt,
                    role=cfg.role,
                    capabilities=cfg.capabilities,
                    enabled_tools=cfg.enabled_tools,
                    mcp_server_ids=cfg.mcp_server_ids or [],
                    model_override=cfg.model_override,
                    is_active=cfg.is_active,
                    is_builtin=False,
                )
            )

    return agents


@router.get("/agents/config/{agent_id}", response_model=AgentConfigResponse)
async def get_agent_config(agent_id: str) -> AgentConfigResponse:
    from src.registry import get_registry
    from src.storage import get_repository

    repo = await get_repository()
    registry = get_registry()

    cfg = await repo.get_agent_config(agent_id)
    if cfg:
        return AgentConfigResponse(
            id=cfg.id,
            name=cfg.name,
            description=cfg.description,
            system_prompt=cfg.system_prompt,
            role=cfg.role,
            capabilities=cfg.capabilities,
            enabled_tools=cfg.enabled_tools,
            mcp_server_ids=cfg.mcp_server_ids or [],
            model_override=cfg.model_override,
            is_active=cfg.is_active,
            is_builtin=cfg.is_builtin,
        )

    agent_def = registry.get(agent_id)
    if agent_def:
        return AgentConfigResponse(
            id=agent_id,
            name=agent_def.name.replace("_", " ").title(),
            description=agent_def.description,
            system_prompt="",
            role="supervisor" if agent_id == "supervisor" else "worker",
            capabilities=[],
            enabled_tools=[],
            mcp_server_ids=[],
            model_override=None,
            is_active=True,
            is_builtin=True,
        )

    raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")


@router.post("/agents/config", response_model=AgentConfigResponse)
async def create_agent(body: CreateAgentRequest) -> AgentConfigResponse:
    from src.registry import get_registry, register_dynamic_agent_async
    from src.storage import get_repository

    repo = await get_repository()
    registry = get_registry()

    if body.id in registry.get_agent_names():
        raise HTTPException(
            status_code=400,
            detail=f"Agent ID '{body.id}' conflicts with builtin agent. Use PUT to customize it.",
        )

    existing = await repo.get_agent_config(body.id)
    if existing:
        raise HTTPException(status_code=400, detail=f"Agent '{body.id}' already exists")

    cfg = await repo.create_agent_config(
        agent_id=body.id,
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        role=body.role,
        capabilities=body.capabilities,
        enabled_tools=body.enabled_tools,
        model_override=body.model_override,
        is_builtin=False,
        mcp_server_ids=body.mcp_server_ids,
    )

    await register_dynamic_agent_async(
        agent_id=cfg.id,
        name=cfg.name,
        description=cfg.description,
        system_prompt=cfg.system_prompt,
        model_override=cfg.model_override,
        enabled_tools=cfg.enabled_tools,
        mcp_server_ids=cfg.mcp_server_ids,
    )

    return AgentConfigResponse(
        id=cfg.id,
        name=cfg.name,
        description=cfg.description,
        system_prompt=cfg.system_prompt,
        role=cfg.role,
        capabilities=cfg.capabilities,
        enabled_tools=cfg.enabled_tools,
        mcp_server_ids=cfg.mcp_server_ids or [],
        model_override=cfg.model_override,
        is_active=cfg.is_active,
        is_builtin=False,
    )


@router.put("/agents/config/{agent_id}", response_model=AgentConfigResponse)
async def update_agent(agent_id: str, body: UpdateAgentRequest) -> AgentConfigResponse:
    from src.registry import get_registry, register_dynamic_agent_async
    from src.storage import get_repository

    repo = await get_repository()
    registry = get_registry()

    cfg = await repo.get_agent_config(agent_id)
    agent_def = registry.get(agent_id)

    if not cfg and not agent_def:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    if cfg:
        base_name = cfg.name
        base_desc = cfg.description
        base_prompt = cfg.system_prompt
        base_caps = cfg.capabilities
        base_tools = cfg.enabled_tools
        base_model = cfg.model_override
        base_mcp = cfg.mcp_server_ids
        is_builtin = cfg.is_builtin
    elif agent_def:
        base_name = agent_def.name.replace("_", " ").title()
        base_desc = agent_def.description
        base_prompt = ""
        base_caps = []
        base_tools = []
        base_model = None
        base_mcp = []
        is_builtin = True
    else:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    updated_cfg = await repo.create_agent_config(
        agent_id=agent_id,
        name=body.name if body.name is not None else base_name,
        description=body.description if body.description is not None else base_desc,
        system_prompt=body.system_prompt if body.system_prompt is not None else base_prompt,
        role="supervisor" if agent_id == "supervisor" else "worker",
        capabilities=body.capabilities if body.capabilities is not None else base_caps,
        enabled_tools=body.enabled_tools if body.enabled_tools is not None else base_tools,
        model_override=body.model_override if body.model_override is not None else base_model,
        is_builtin=is_builtin,
        mcp_server_ids=body.mcp_server_ids if body.mcp_server_ids is not None else base_mcp,
    )

    if body.is_active is not None:
        await repo.toggle_agent(agent_id, body.is_active)
        refreshed_cfg = await repo.get_agent_config(agent_id)
        if refreshed_cfg is not None:
            updated_cfg = refreshed_cfg

    await register_dynamic_agent_async(
        agent_id=updated_cfg.id,
        name=updated_cfg.name,
        description=updated_cfg.description,
        system_prompt=updated_cfg.system_prompt,
        model_override=updated_cfg.model_override,
        enabled_tools=updated_cfg.enabled_tools,
        mcp_server_ids=updated_cfg.mcp_server_ids,
    )

    return AgentConfigResponse(
        id=updated_cfg.id,
        name=updated_cfg.name,
        description=updated_cfg.description,
        system_prompt=updated_cfg.system_prompt,
        role=updated_cfg.role,
        capabilities=updated_cfg.capabilities,
        enabled_tools=updated_cfg.enabled_tools,
        mcp_server_ids=updated_cfg.mcp_server_ids or [],
        model_override=updated_cfg.model_override,
        is_active=updated_cfg.is_active,
        is_builtin=updated_cfg.is_builtin,
    )


@router.delete("/agents/config/{agent_id}")
async def delete_agent(agent_id: str) -> dict[str, str | bool]:
    from src.registry import get_registry, mark_registry_dirty, unregister_dynamic_agent
    from src.storage import get_repository

    repo = await get_repository()
    registry = get_registry()

    cfg = await repo.get_agent_config(agent_id)

    if not cfg:
        if agent_id in registry.get_agent_names():
            return {
                "success": True,
                "message": f"Agent '{agent_id}' is using defaults (nothing to reset)",
            }
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    if cfg.is_builtin:
        await repo.delete_agent_config(agent_id)
        mark_registry_dirty()
        return {
            "success": True,
            "message": f"Agent '{agent_id}' reset to defaults",
        }

    await repo.delete_agent_config(agent_id)
    unregister_dynamic_agent(agent_id)
    return {
        "success": True,
        "message": f"Agent '{agent_id}' deleted",
    }


@router.get("/tools/available")
async def list_available_tools() -> list[dict[str, str]]:
    from src.tools.plugins import get_plugin_registry

    registry = get_plugin_registry()
    tools_list: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for plugin in registry.get_all_plugins():
        for tool in registry.get_tools(plugin.name):
            if tool.name in seen_ids:
                continue
            seen_ids.add(tool.name)
            tools_list.append(
                {
                    "id": tool.name,
                    "name": tool.name.replace("_", " ").title(),
                    "description": tool.description or "",
                }
            )

    return tools_list


class PipelineResponse(BaseModel):
    id: str
    name: str
    description: str
    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    settings: dict[str, object]
    is_active: bool
    is_default: bool


async def _enrich_pipeline_nodes(
    nodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    from src.storage import get_repository

    if not nodes:
        return nodes

    repo = await get_repository()

    agent_configs = await repo.list_agent_configs(active_only=False)
    agent_map = {cfg.id: cfg for cfg in agent_configs}

    teams = await repo.list_teams()
    team_map = {t.id: t for t in teams}

    enriched_nodes: list[dict[str, object]] = []
    for node in nodes:
        enriched = dict(node)
        node_type = str(node.get("type", ""))
        data = node.get("data")

        if not isinstance(data, dict):
            enriched_nodes.append(enriched)
            continue

        enriched_data = dict(data)

        if node_type == "team":
            team_id = data.get("teamId")
            if team_id and team_id in team_map:
                team = team_map[team_id]
                enriched_data["teamName"] = team.name

                agent_ids = team.agent_ids or []
                enriched_agents = []
                for agent_id in agent_ids:
                    if agent_id in agent_map:
                        cfg = agent_map[agent_id]
                        enriched_agents.append(
                            {
                                "id": agent_id,
                                "name": cfg.name,
                                "description": cfg.description or "",
                            }
                        )
                    else:
                        enriched_agents.append(
                            {
                                "id": agent_id,
                                "name": agent_id,
                                "description": "",
                            }
                        )
                enriched_data["agents"] = enriched_agents
                enriched_data["agentIds"] = agent_ids
                enriched_data["leadAgentId"] = team.lead_agent_id

        elif node_type == "agent":
            agent_id = data.get("agentId") or data.get("agentName") or str(node.get("id", ""))
            if agent_id and agent_id in agent_map:
                cfg = agent_map[agent_id]
                enriched_data["label"] = cfg.name
                enriched_data["description"] = cfg.description or ""
                enriched_data["agentName"] = agent_id

        enriched["data"] = enriched_data
        enriched_nodes.append(enriched)

    return enriched_nodes


class CreatePipelineRequest(BaseModel):
    name: str
    description: str = ""
    nodes: list[dict[str, object]] = Field(default_factory=list)
    edges: list[dict[str, object]] = Field(default_factory=list)
    settings: dict[str, object] = Field(default_factory=dict)
    is_default: bool = False


class UpdatePipelineRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    nodes: list[dict[str, object]] | None = None
    edges: list[dict[str, object]] | None = None
    settings: dict[str, object] | None = None
    is_active: bool | None = None
    is_default: bool | None = None


@router.get("/pipelines", response_model=list[PipelineResponse])
async def list_pipelines() -> list[PipelineResponse]:
    from src.storage import get_repository

    repo = await get_repository()
    pipelines = await repo.list_pipelines(active_only=False)

    responses = []
    for p in pipelines:
        enriched_nodes = await _enrich_pipeline_nodes(p.nodes or [])
        responses.append(
            PipelineResponse(
                id=p.id,
                name=p.name,
                description=p.description,
                nodes=enriched_nodes,
                edges=p.edges or [],
                settings=p.settings or {},
                is_active=p.is_active,
                is_default=p.is_default,
            )
        )
    return responses


@router.get("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str) -> PipelineResponse:
    from src.storage import get_repository

    repo = await get_repository()
    pipeline = await repo.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    enriched_nodes = await _enrich_pipeline_nodes(pipeline.nodes or [])

    return PipelineResponse(
        id=pipeline.id,
        name=pipeline.name,
        description=pipeline.description,
        nodes=enriched_nodes,
        edges=pipeline.edges or [],
        settings=pipeline.settings or {},
        is_active=pipeline.is_active,
        is_default=pipeline.is_default,
    )


@router.post("/pipelines", response_model=PipelineResponse)
async def create_pipeline(body: CreatePipelineRequest) -> PipelineResponse:
    from src.storage import get_repository

    repo = await get_repository()
    pipeline = await repo.create_pipeline(
        name=body.name,
        description=body.description,
        nodes=body.nodes,
        edges=body.edges,
        settings=body.settings,
        is_default=body.is_default,
    )

    enriched_nodes = await _enrich_pipeline_nodes(pipeline.nodes or [])

    return PipelineResponse(
        id=pipeline.id,
        name=pipeline.name,
        description=pipeline.description,
        nodes=enriched_nodes,
        edges=pipeline.edges or [],
        settings=pipeline.settings or {},
        is_active=pipeline.is_active,
        is_default=pipeline.is_default,
    )


@router.put("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(pipeline_id: str, body: UpdatePipelineRequest) -> PipelineResponse:
    from src.storage import get_repository

    repo = await get_repository()
    pipeline = await repo.update_pipeline(
        pipeline_id=pipeline_id,
        name=body.name,
        description=body.description,
        nodes=body.nodes,
        edges=body.edges,
        settings=body.settings,
        is_active=body.is_active,
        is_default=body.is_default,
    )

    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    enriched_nodes = await _enrich_pipeline_nodes(pipeline.nodes or [])

    return PipelineResponse(
        id=pipeline.id,
        name=pipeline.name,
        description=pipeline.description,
        nodes=enriched_nodes,
        edges=pipeline.edges or [],
        settings=pipeline.settings or {},
        is_active=pipeline.is_active,
        is_default=pipeline.is_default,
    )


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str) -> dict[str, str | bool]:
    from src.storage import get_repository

    repo = await get_repository()
    deleted = await repo.delete_pipeline(pipeline_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    return {"success": True, "message": f"Pipeline '{pipeline_id}' deleted"}


class PipelineRunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=10000)
    thread_id: str | None = None


@router.post("/pipelines/{pipeline_id}/run", response_model=TaskResponse)
@limiter.limit("10/minute")
async def run_pipeline(request: Request, pipeline_id: str, body: PipelineRunRequest) -> TaskResponse:
    from src.graph.pipeline_builder import PipelineValidationError, get_pipeline_workflow

    manager = request.app.state.manager
    task_id = str(uuid.uuid4())[:8]
    thread_id = body.thread_id or task_id

    tracer = reset_tracer(task_id=task_id, thread_id=thread_id)
    tracer.set_broadcast(manager.broadcast)

    try:
        pipeline_workflow = await get_pipeline_workflow(pipeline_id)
    except PipelineValidationError as e:
        raise HTTPException(status_code=400, detail=f"Pipeline validation failed: {'; '.join(e.errors)}") from None

    if not pipeline_workflow:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    await manager.broadcast(
        {
            "type": "task_start",
            "task_id": task_id,
            "agent": "system",
            "message": f"Pipeline task started: {body.task[:100]}",
        }
    )

    initial_state: AgentState = {
        "messages": [HumanMessage(content=body.task)],
        "current_agent": "pipeline",
        "context": {"task_id": task_id, "pipeline_id": pipeline_id},
        "human_feedback": None,
        "iteration_count": 0,
        "execution_trace": [],
        "loop_counts": {},
    }

    result = await pipeline_workflow.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result.get("messages", [])
    response_content = ""
    response_agent = "agent"

    for msg in reversed(messages):
        if not msg.content:
            continue
        content = str(msg.content)
        if isinstance(msg, (AIMessage, ToolMessage)):
            response_content = content
            response_agent = getattr(msg, "name", None) or "agent"
            break

    response_content = _fix_document_links(response_content)

    await manager.broadcast(
        {
            "type": "task_end",
            "task_id": task_id,
            "agent": "system",
            "message": "Pipeline task completed",
            "result": response_content[:200],
        }
    )

    return TaskResponse(
        task_id=task_id,
        result=response_content,
        agent=response_agent,
        messages=[
            {
                "role": type(m).__name__.replace("Message", "").lower(),
                "content": _fix_document_links(str(m.content)) if m.content else None,
                "name": getattr(m, "name", None),
            }
            for m in messages
            if m.content
        ],
    )


@router.post("/pipelines/{pipeline_id}/validate")
async def validate_pipeline(pipeline_id: str) -> dict[str, bool | int | list[str]]:
    from src.registry import get_registry
    from src.storage import get_repository

    repo = await get_repository()
    registry = get_registry()

    pipeline = await repo.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    errors: list[str] = []
    warnings: list[str] = []

    nodes = pipeline.nodes or []
    edges = pipeline.edges or []

    agent_nodes = [n for n in nodes if n.get("type") == "agent"]
    team_nodes = [n for n in nodes if n.get("type") == "team"]
    start_nodes = [n for n in nodes if n.get("type") == "start"]
    end_nodes = [n for n in nodes if n.get("type") == "end"]

    if not start_nodes:
        errors.append("Pipeline missing start node")
    if not end_nodes:
        warnings.append("Pipeline missing end node (will auto-terminate)")
    if not agent_nodes and not team_nodes:
        errors.append("Pipeline has no agent or team nodes")

    registered_agents = registry.get_agent_names()
    for node in agent_nodes:
        data = node.get("data", {})
        agent_name = data.get("agentName") or data.get("agentId") or data.get("name") or data.get("label")
        if agent_name and agent_name not in registered_agents:
            errors.append(f"Unknown agent: {agent_name}")

    for node in team_nodes:
        data = node.get("data", {})
        agent_ids = data.get("agentIds", [])
        for agent_id in agent_ids:
            if agent_id not in registered_agents:
                errors.append(f"Unknown agent in team: {agent_id}")

    node_ids = {n.get("id") for n in nodes if n.get("id")}
    for edge in edges:
        if edge.get("source") not in node_ids:
            errors.append(f"Edge source not found: {edge.get('source')}")
        if edge.get("target") not in node_ids:
            errors.append(f"Edge target not found: {edge.get('target')}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "agent_count": len(agent_nodes),
        "team_count": len(team_nodes),
    }


class TeamResponse(BaseModel):
    id: str
    name: str
    description: str
    agent_ids: list[str]
    lead_agent_id: str | None
    settings: dict[str, object]
    is_active: bool


class CreateTeamRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    agent_ids: list[str] = []
    lead_agent_id: str | None = None
    settings: dict[str, object] = {}


class UpdateTeamRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    agent_ids: list[str] | None = None
    lead_agent_id: str | None = None
    settings: dict[str, object] | None = None
    is_active: bool | None = None


@router.get("/teams", response_model=list[TeamResponse])
async def list_teams() -> list[TeamResponse]:
    from src.storage import get_repository

    repo = await get_repository()
    teams = await repo.list_teams()

    return [
        TeamResponse(
            id=t.id,
            name=t.name,
            description=t.description,
            agent_ids=t.agent_ids or [],
            lead_agent_id=t.lead_agent_id,
            settings=t.settings or {},
            is_active=t.is_active,
        )
        for t in teams
    ]


@router.get("/teams/{team_id}", response_model=TeamResponse)
async def get_team(team_id: str) -> TeamResponse:
    from src.storage import get_repository

    repo = await get_repository()
    team = await repo.get_team(team_id)

    if not team:
        raise HTTPException(status_code=404, detail=f"Team not found: {team_id}")

    return TeamResponse(
        id=team.id,
        name=team.name,
        description=team.description,
        agent_ids=team.agent_ids or [],
        lead_agent_id=team.lead_agent_id,
        settings=team.settings or {},
        is_active=team.is_active,
    )


@router.post("/teams", response_model=TeamResponse)
async def create_team(body: CreateTeamRequest) -> TeamResponse:
    from src.storage import get_repository

    repo = await get_repository()
    team = await repo.create_team(
        name=body.name,
        description=body.description,
        agent_ids=body.agent_ids,
        lead_agent_id=body.lead_agent_id,
        settings=body.settings,
    )

    return TeamResponse(
        id=team.id,
        name=team.name,
        description=team.description,
        agent_ids=team.agent_ids or [],
        lead_agent_id=team.lead_agent_id,
        settings=team.settings or {},
        is_active=team.is_active,
    )


@router.put("/teams/{team_id}", response_model=TeamResponse)
async def update_team(team_id: str, body: UpdateTeamRequest) -> TeamResponse:
    from src.storage import get_repository

    repo = await get_repository()
    team = await repo.update_team(
        team_id=team_id,
        name=body.name,
        description=body.description,
        agent_ids=body.agent_ids,
        lead_agent_id=body.lead_agent_id,
        settings=body.settings,
        is_active=body.is_active,
    )

    if not team:
        raise HTTPException(status_code=404, detail=f"Team not found: {team_id}")

    return TeamResponse(
        id=team.id,
        name=team.name,
        description=team.description,
        agent_ids=team.agent_ids or [],
        lead_agent_id=team.lead_agent_id,
        settings=team.settings or {},
        is_active=team.is_active,
    )


@router.delete("/teams/{team_id}")
async def delete_team(team_id: str) -> dict[str, str | bool]:
    from src.storage import get_repository

    repo = await get_repository()
    deleted = await repo.delete_team(team_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Team not found: {team_id}")

    return {"success": True, "message": f"Team '{team_id}' deleted"}


@router.get("/documents/download/{filename:path}")
async def download_document(filename: str) -> FileResponse:
    """Download a generated document (PDF, Excel, CSV)."""
    import os
    import re
    from pathlib import Path

    output_dir = Path(os.getenv("DOCUMENT_OUTPUT_DIR", "./data/documents")).resolve()

    if not re.fullmatch(r"[a-zA-Z0-9_\-\.]+", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    allowed_files = {f.name for f in output_dir.iterdir() if f.is_file()} if output_dir.exists() else set()
    if filename not in allowed_files:
        raise HTTPException(status_code=404, detail="Document not found")

    validated_path = output_dir / filename

    ext = validated_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(validated_path),
        filename=filename,
        media_type=media_type,
    )


@router.get("/documents/list")
async def list_generated_documents() -> list[dict[str, str | int | float]]:
    """List all generated documents in the output directory."""
    import os
    from pathlib import Path

    output_dir = Path(os.getenv("DOCUMENT_OUTPUT_DIR", "./data/documents"))

    if not output_dir.exists():
        return []

    documents: list[dict[str, str | int | float]] = []
    for file_path in output_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".xlsx", ".csv"]:
            stat = file_path.stat()
            documents.append(
                {
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "created_at": stat.st_ctime,
                    "download_url": f"/api/documents/download/{file_path.name}",
                }
            )

    documents.sort(key=lambda x: float(x["created_at"]), reverse=True)
    return documents
