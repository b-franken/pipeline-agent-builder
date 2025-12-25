"""FastAPI application with rate limiting and middleware."""

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(levelname)s:%(name)s:%(message)s",
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.mcp_routes import router as mcp_router
from src.api.routes import router
from src.api.websocket import ConnectionManager


async def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    detail = str(exc.detail) if hasattr(exc, "detail") else str(exc)
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": detail},
    )


limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    import logging

    from src.config import settings
    from src.graph.workflow import get_async_workflow, get_workflow, invalidate_workflow
    from src.llm import sync_runtime_settings
    from src.registry import set_workflow_invalidation_callback, sync_agents_from_database

    logger = logging.getLogger(__name__)

    if settings.setup_langsmith():
        logger.info(f"LangSmith tracing enabled for project: {settings.langsmith_project}")
    else:
        logger.debug("LangSmith tracing not configured (set LANGSMITH_API_KEY and LANGCHAIN_TRACING_V2=true)")

    set_workflow_invalidation_callback(invalidate_workflow)
    await sync_runtime_settings()
    await sync_agents_from_database()

    try:
        from src.knowledge import DocumentIngester

        ingester = DocumentIngester()
        doc_count = await ingester.sync_from_database()
        if doc_count > 0:
            logger.info(f"Synced {doc_count} documents from database")
    except Exception as e:
        logger.warning(f"Failed to sync documents: {e}")

    try:
        from src.mcp.client import get_mcp_client_manager
        from src.storage import get_repository

        repo = await get_repository()
        servers = await repo.list_mcp_servers(active_only=True)
        if servers:
            manager_mcp = get_mcp_client_manager()
            manager_mcp.configure_servers(servers)
            logger.info(f"Loaded {len(servers)} MCP servers from database")
    except ImportError:
        logger.debug("MCP module not available")
    except Exception as e:
        logger.warning(f"Failed to initialize MCP servers: {e}")

    app.state.get_workflow = get_workflow
    app.state.get_async_workflow = get_async_workflow
    app.state.manager = manager
    yield
    set_workflow_invalidation_callback(None)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Any) -> Any:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        return response

    app.include_router(router, prefix="/api")
    app.include_router(mcp_router, prefix="/api")

    @app.get("/health")
    async def health() -> dict[str, object]:
        from src.llm import get_runtime_settings
        from src.mcp.client import is_mcp_available
        from src.storage import get_repository

        checks: dict[str, bool] = {}
        degraded_reasons: list[str] = []

        try:
            repo = await get_repository()
            await repo.get_recent_tasks(limit=1)
            checks["database"] = True
        except Exception:
            checks["database"] = False
            degraded_reasons.append("database unavailable")

        runtime = get_runtime_settings()
        checks["runtime_settings"] = len(runtime) > 0
        if not checks["runtime_settings"]:
            degraded_reasons.append("no runtime settings loaded")

        checks["mcp_available"] = is_mcp_available()

        all_healthy = checks["database"]
        status = "healthy" if all_healthy else "degraded"

        return {
            "status": status,
            "checks": checks,
            "degraded_reasons": degraded_reasons,
        }

    return app


app = create_app()
