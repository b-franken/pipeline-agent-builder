"""LLM factory with multi-provider support and rate limiting.

Supports runtime configuration from database settings.
Priority: explicit args > database settings > environment/config.py
"""

import logging
import threading
from typing import Any, Final, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from pydantic import BaseModel, SecretStr

from src.config import settings

logger: Final = logging.getLogger(__name__)

Provider = Literal["openai", "ollama", "google", "anthropic"]

_rate_limiters: dict[str, InMemoryRateLimiter] = {}
_rate_limiter_lock = threading.Lock()


def _get_rate_limiter(provider: str) -> InMemoryRateLimiter | None:
    if provider == "ollama":
        return None

    with _rate_limiter_lock:
        if provider not in _rate_limiters:
            _rate_limiters[provider] = InMemoryRateLimiter(
                requests_per_second=2.0,
                check_every_n_seconds=0.1,
                max_bucket_size=20,
            )
        return _rate_limiters[provider]


_runtime_settings_cache: dict[str, str] = {}
_settings_lock = threading.Lock()


def get_runtime_settings() -> dict[str, str]:
    """Get cached runtime settings from database."""
    with _settings_lock:
        return dict(_runtime_settings_cache)


def update_runtime_settings(new_settings: dict[str, str]) -> None:
    """Update the runtime settings cache."""
    with _settings_lock:
        _runtime_settings_cache.clear()
        _runtime_settings_cache.update(new_settings)


def clear_runtime_settings() -> None:
    """Clear the runtime settings cache."""
    with _settings_lock:
        _runtime_settings_cache.clear()


async def sync_runtime_settings() -> dict[str, str]:
    """Sync runtime settings from database to cache."""
    try:
        from src.storage import get_repository

        repo = await get_repository()

        provider = await repo.get_setting("provider")
        model = await repo.get_setting("model")
        openai_key = await repo.get_setting("openai_api_key", decrypt=True)
        anthropic_key = await repo.get_setting("anthropic_api_key", decrypt=True)
        google_key = await repo.get_setting("google_api_key", decrypt=True)
        ollama_host = await repo.get_setting("ollama_host")

        new_settings: dict[str, str] = {}
        if provider:
            new_settings["provider"] = provider
        if model:
            new_settings["model"] = model
        if openai_key:
            new_settings["openai_api_key"] = openai_key
        if anthropic_key:
            new_settings["anthropic_api_key"] = anthropic_key
        if google_key:
            new_settings["google_api_key"] = google_key
        if ollama_host:
            new_settings["ollama_host"] = ollama_host

        update_runtime_settings(new_settings)
        logger.info("Synced %d runtime settings from database", len(new_settings))
        return new_settings
    except Exception:
        logger.exception("Failed to sync runtime settings from database")
        return {}


class ResolvedSettings:
    __slots__ = ("anthropic_key", "google_key", "model", "ollama_host", "openai_key", "provider")

    def __init__(
        self,
        provider: str,
        model: str,
        openai_key: str | None,
        anthropic_key: str | None,
        google_key: str | None,
        ollama_host: str | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.google_key = google_key
        self.ollama_host = ollama_host


def get_resolved_settings(
    model: str | None = None,
    provider: Provider | None = None,
) -> ResolvedSettings:
    runtime = get_runtime_settings()

    resolved_provider = provider or runtime.get("provider") or settings.provider
    resolved_model = model or runtime.get("model") or settings.default_model

    openai_key = runtime.get("openai_api_key") or settings.openai_api_key
    anthropic_key = runtime.get("anthropic_api_key") or settings.anthropic_api_key
    google_key = runtime.get("google_api_key") or settings.google_api_key
    ollama_host = runtime.get("ollama_host") or settings.ollama_base_url

    return ResolvedSettings(
        provider=resolved_provider,
        model=resolved_model,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        google_key=google_key,
        ollama_host=ollama_host,
    )


def create_llm(
    model: str | None = None,
    temperature: float | None = None,
    provider: Provider | None = None,
    use_rate_limiter: bool = True,
) -> tuple[BaseChatModel, str]:
    resolved = get_resolved_settings(model, provider)

    if model:
        resolved.model = model

    temp: float = temperature if temperature is not None else settings.temperature
    rate_limiter = _get_rate_limiter(resolved.provider) if use_rate_limiter else None

    if resolved.provider == "openai":
        from langchain_openai import ChatOpenAI

        return (
            ChatOpenAI(
                model=resolved.model,
                temperature=temp,
                api_key=SecretStr(resolved.openai_key) if resolved.openai_key else None,
                rate_limiter=rate_limiter,
            ),
            resolved.model,
        )

    if resolved.provider == "ollama":
        from langchain_ollama import ChatOllama

        return (
            ChatOllama(
                model=resolved.model,
                temperature=temp,
                base_url=resolved.ollama_host,
            ),
            resolved.model,
        )

    if resolved.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return (
            ChatGoogleGenerativeAI(
                model=resolved.model,
                temperature=temp,
                google_api_key=resolved.google_key,
                rate_limiter=rate_limiter,
            ),
            resolved.model,
        )

    if resolved.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return (
            ChatAnthropic(
                model_name=resolved.model,
                temperature=temp,
                timeout=None,
                stop=None,
                api_key=SecretStr(resolved.anthropic_key) if resolved.anthropic_key else SecretStr(""),
                rate_limiter=rate_limiter,
            ),
            resolved.model,
        )

    raise ValueError(f"Unknown provider: {resolved.provider}")


def create_llm_with_structured_output[T: BaseModel](
    schema: type[T],
    model: str | None = None,
    temperature: float | None = None,
    provider: Provider | None = None,
) -> tuple[Runnable[Any, T], str]:
    llm, model_name = create_llm(model, temperature, provider)
    return cast("Runnable[Any, T]", llm.with_structured_output(schema)), model_name
