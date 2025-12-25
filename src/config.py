import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import config as user_config

    _PROVIDER = getattr(user_config, "PROVIDER", "openai")
    _MODEL = getattr(user_config, "MODEL", "gpt-4o")
    _OPENAI_API_KEY = getattr(user_config, "OPENAI_API_KEY", "")
    _GOOGLE_API_KEY = getattr(user_config, "GOOGLE_API_KEY", "")
    _ANTHROPIC_API_KEY = getattr(user_config, "ANTHROPIC_API_KEY", "")
    _OLLAMA_BASE_URL = getattr(user_config, "OLLAMA_BASE_URL", "http://localhost:11434")
except ImportError:
    _PROVIDER = "openai"
    _MODEL = "gpt-4o"
    _OPENAI_API_KEY = ""
    _GOOGLE_API_KEY = ""
    _ANTHROPIC_API_KEY = ""
    _OLLAMA_BASE_URL = "http://localhost:11434"

_PROVIDER = os.getenv("PROVIDER", _PROVIDER)
_MODEL = os.getenv("DEFAULT_MODEL", _MODEL)
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", _OPENAI_API_KEY)
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", _GOOGLE_API_KEY)
_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", _ANTHROPIC_API_KEY)
_OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", _OLLAMA_BASE_URL)


@dataclass
class Settings:
    provider: str = field(default=_PROVIDER)
    default_model: str = field(default=_MODEL)
    temperature: float = field(default=0.7)
    max_iterations: int = field(default=10)
    supervisor_recursion_limit: int = field(default=25)

    use_postgres: bool = field(default_factory=lambda: os.getenv("USE_POSTGRES", "false").lower() == "true")
    postgres_uri: str = field(default_factory=lambda: os.getenv("POSTGRES_URI", ""))
    sqlite_db_path: Path = field(default_factory=lambda: Path(os.getenv("SQLITE_DB_PATH", "./data/memory.db")))

    chroma_host: str | None = field(default_factory=lambda: os.getenv("CHROMA_HOST"))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))

    openai_api_key: str = field(default=_OPENAI_API_KEY)
    google_api_key: str = field(default=_GOOGLE_API_KEY)
    anthropic_api_key: str = field(default=_ANTHROPIC_API_KEY)
    ollama_base_url: str = field(default=_OLLAMA_BASE_URL)

    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    ollama_embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_timeout: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_TIMEOUT", "30")))
    embedding_max_retries: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_MAX_RETRIES", "3")))

    e2b_api_key: str = field(default_factory=lambda: os.getenv("E2B_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "ai-agent-dashboard"))
    langsmith_tracing_enabled: bool = field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )

    tool_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("TOOL_TIMEOUT_SECONDS", "60")))

    context_max_messages: int = field(default_factory=lambda: int(os.getenv("CONTEXT_MAX_MESSAGES", "20")))
    context_summary_threshold: int = field(default_factory=lambda: int(os.getenv("CONTEXT_SUMMARY_THRESHOLD", "15")))
    context_keep_recent: int = field(default_factory=lambda: int(os.getenv("CONTEXT_KEEP_RECENT", "6")))

    @property
    def is_docker(self) -> bool:
        return self.use_postgres or self.chroma_host is not None

    @property
    def checkpointer_backend(self) -> str:
        if self.use_postgres and self.postgres_uri:
            return "postgres"
        return "sqlite"

    def ensure_data_dir(self) -> None:
        if not self.use_postgres:
            self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_api_key(self) -> str:
        if self.provider == "openai":
            return self.openai_api_key
        elif self.provider == "google":
            return self.google_api_key
        elif self.provider == "anthropic":
            return self.anthropic_api_key
        return ""

    @property
    def embedding_provider(self) -> str:
        if self.provider == "ollama":
            return "ollama"
        return "openai"

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing_enabled and bool(self.langsmith_api_key)

    def setup_langsmith(self) -> bool:
        if not self.langsmith_enabled:
            return False

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        return True


settings = Settings()
