import asyncio
import atexit
import os
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

DATA_DIR = Path("./data")

_checkpointer_cache: dict[str, BaseCheckpointSaver[Any]] = {}
_postgres_conn: Any = None
_async_checkpointer_cache: dict[str, BaseCheckpointSaver[Any]] = {}
_async_postgres_conn: Any = None
_async_lock: asyncio.Lock | None = None


def _resolve_backend() -> str:
    if os.environ.get("POSTGRES_URI") or os.environ.get("USE_POSTGRES"):
        return "postgres"
    if os.environ.get("USE_SQLITE", "true").lower() == "true":
        return "sqlite"
    return "memory"


def _make_cache_key(backend: str, postgres_uri: str | None) -> str:
    uri_part = postgres_uri or os.environ.get("POSTGRES_URI", "")
    return f"{backend}:{uri_part}"


def create_checkpointer(
    backend: str | None = None,
    postgres_uri: str | None = None,
) -> BaseCheckpointSaver[Any]:
    resolved_backend = backend or _resolve_backend()
    cache_key = _make_cache_key(resolved_backend, postgres_uri)

    if cache_key in _checkpointer_cache:
        return _checkpointer_cache[cache_key]

    checkpointer: BaseCheckpointSaver[Any]
    if resolved_backend == "memory":
        checkpointer = MemorySaver()
    elif resolved_backend == "postgres":
        checkpointer = _create_postgres_checkpointer(postgres_uri)
    elif resolved_backend == "sqlite":
        checkpointer = _create_sqlite_checkpointer()
    else:
        raise ValueError(f"Unknown backend: {resolved_backend}")

    _checkpointer_cache[cache_key] = checkpointer
    return checkpointer


def _close_postgres_conn() -> None:
    global _postgres_conn
    if _postgres_conn is not None:
        _postgres_conn.close()
        _postgres_conn = None


def _create_postgres_checkpointer(uri: str | None = None) -> BaseCheckpointSaver[Any]:
    global _postgres_conn
    try:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
    except ImportError as e:
        raise ImportError("Install: pip install langgraph-checkpoint-postgres psycopg[binary]") from e

    connection_string = uri or os.environ.get("POSTGRES_URI")
    if not connection_string:
        raise ValueError("POSTGRES_URI environment variable required")

    _postgres_conn = psycopg.connect(connection_string, autocommit=True, row_factory=dict_row)
    atexit.register(_close_postgres_conn)

    checkpointer = PostgresSaver(_postgres_conn)
    checkpointer.setup()
    return cast(BaseCheckpointSaver[Any], checkpointer)


def _create_sqlite_checkpointer() -> BaseCheckpointSaver[Any]:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return MemorySaver()

    DATA_DIR.mkdir(exist_ok=True)
    db_path = os.environ.get("SQLITE_DB_PATH", str(DATA_DIR / "memory.db"))
    return cast(BaseCheckpointSaver[Any], SqliteSaver.from_conn_string(db_path))


def get_memory_saver() -> MemorySaver:
    return MemorySaver()


async def _close_async_postgres_conn() -> None:
    global _async_postgres_conn
    if _async_postgres_conn is not None:
        await _async_postgres_conn.close()
        _async_postgres_conn = None


def _get_async_lock() -> asyncio.Lock:
    global _async_lock
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    return _async_lock


async def create_async_checkpointer(
    backend: str | None = None,
    postgres_uri: str | None = None,
) -> BaseCheckpointSaver[Any]:
    global _async_postgres_conn

    resolved_backend = backend or _resolve_backend()
    cache_key = _make_cache_key(resolved_backend, postgres_uri)

    if cache_key in _async_checkpointer_cache:
        return _async_checkpointer_cache[cache_key]

    async with _get_async_lock():
        if cache_key in _async_checkpointer_cache:
            return _async_checkpointer_cache[cache_key]

        checkpointer: BaseCheckpointSaver[Any]
        if resolved_backend == "memory":
            checkpointer = MemorySaver()
        elif resolved_backend == "postgres":
            try:
                import psycopg
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from psycopg.rows import dict_row
            except ImportError as e:
                raise ImportError("Install: pip install langgraph-checkpoint-postgres psycopg[binary]") from e

            connection_string = postgres_uri or os.environ.get("POSTGRES_URI")
            if not connection_string:
                raise ValueError("POSTGRES_URI environment variable required")

            _async_postgres_conn = await psycopg.AsyncConnection.connect(
                connection_string, autocommit=True, row_factory=dict_row
            )
            async_saver = AsyncPostgresSaver(_async_postgres_conn)
            await async_saver.setup()
            checkpointer = cast(BaseCheckpointSaver[Any], async_saver)
        elif resolved_backend == "sqlite":
            checkpointer = _create_sqlite_checkpointer()
        else:
            raise ValueError(f"Unknown backend: {resolved_backend}")

        _async_checkpointer_cache[cache_key] = checkpointer
        return _async_checkpointer_cache[cache_key]
