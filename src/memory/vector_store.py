import logging
import os
from pathlib import Path
from typing import Any, Final

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import Field

logger: Final = logging.getLogger(__name__)

DATA_DIR = Path("./data/chroma")


class VectorMemory:
    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_directory: str | None = None,
        embedding_provider: str = "openai",
        chroma_host: str | None = None,
        chroma_port: int | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(DATA_DIR)
        self._vectorstore: Any = None
        self._embedding_provider = embedding_provider
        self._chroma_host = chroma_host or os.environ.get("CHROMA_HOST")
        self._chroma_port = chroma_port or int(os.environ.get("CHROMA_PORT", "8000"))
        self._embedding_model: str | None = None

    @property
    def embedding_model(self) -> str:
        if self._embedding_model is None:
            from src.config import settings

            if self._embedding_provider == "openai":
                self._embedding_model = settings.embedding_model
            else:
                self._embedding_model = settings.ollama_embed_model
        return self._embedding_model

    @property
    def is_client_mode(self) -> bool:
        return self._chroma_host is not None

    def _get_embeddings(self) -> Any:
        """Create embeddings instance with proper configuration from settings."""
        from src.config import settings

        if self._embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            from pydantic import SecretStr

            api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                ollama_available = bool(settings.ollama_base_url)
                if ollama_available:
                    logger.warning(
                        "No OpenAI API key configured for embeddings. "
                        f"Falling back to Ollama at {settings.ollama_base_url}"
                    )
                    from langchain_ollama import OllamaEmbeddings

                    return OllamaEmbeddings(
                        model=settings.ollama_embed_model,
                        base_url=settings.ollama_base_url,
                    )
                raise ValueError(
                    "No OpenAI API key configured for embeddings. "
                    "Set OPENAI_API_KEY environment variable or configure Ollama as fallback."
                )

            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=SecretStr(api_key),
                timeout=settings.embedding_timeout,
                max_retries=settings.embedding_max_retries,
            )
        elif self._embedding_provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            if not settings.ollama_base_url:
                raise ValueError("Ollama embedding provider selected but OLLAMA_BASE_URL is not configured.")

            return OllamaEmbeddings(
                model=settings.ollama_embed_model,
                base_url=settings.ollama_base_url,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self._embedding_provider}")

    def _get_chroma_client(self) -> Any:
        import chromadb
        from chromadb.config import Settings

        return chromadb.HttpClient(
            host=self._chroma_host,
            port=self._chroma_port,
            settings=Settings(anonymized_telemetry=False),
        )

    def _get_vectorstore(self) -> Any:
        if self._vectorstore is None:
            from langchain_chroma import Chroma

            if self.is_client_mode:
                self._vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self._get_embeddings(),
                    client=self._get_chroma_client(),
                )
            else:
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                self._vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self._get_embeddings(),
                    persist_directory=self.persist_directory,
                )
        return self._vectorstore

    def store(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        enriched_metadata = {
            **(metadata or {}),
            "embedding_model": self.embedding_model,
            "embedding_provider": self._embedding_provider,
        }
        doc = Document(page_content=content, metadata=enriched_metadata)
        ids = self._get_vectorstore().add_documents([doc])
        return ids[0] if ids else ""

    def search(self, query: str, k: int = 5) -> list[str]:
        docs = self._get_vectorstore().similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        results = self._get_vectorstore().similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]

    def clear(self) -> None:
        vs = self._get_vectorstore()
        collection = vs._collection
        if collection.count() > 0:
            collection.delete(where={})


_memory_registry: dict[str, VectorMemory] = {}


def get_memory(
    collection_name: str = "agent_memory",
    embedding_provider: str = "openai",
) -> VectorMemory:
    global _memory_registry
    cache_key = f"{collection_name}:{embedding_provider}"
    if cache_key not in _memory_registry:
        _memory_registry[cache_key] = VectorMemory(
            collection_name=collection_name,
            embedding_provider=embedding_provider,
        )
    return _memory_registry[cache_key]


def clear_memory_registry() -> None:
    global _memory_registry
    _memory_registry.clear()


class MemoryStoreTool(BaseTool):
    name: str = "store_memory"
    description: str = "Store important information for later recall."
    memory: VectorMemory = Field(default_factory=get_memory)

    def _run(self, content: str) -> str:
        doc_id = self.memory.store(content)
        return f"Stored in memory (id: {doc_id})"

    async def _arun(self, content: str) -> str:
        return self._run(content)


class MemorySearchTool(BaseTool):
    name: str = "search_memory"
    description: str = "Search previous conversations and stored information."
    memory: VectorMemory = Field(default_factory=get_memory)

    def _run(self, query: str) -> str:
        results = self.memory.search(query, k=5)
        if not results:
            return "No relevant memories found."
        return "\n---\n".join(results)

    async def _arun(self, query: str) -> str:
        return self._run(query)


def create_memory_tools(memory: VectorMemory | None = None) -> list[BaseTool]:
    mem = memory or get_memory()
    return [MemoryStoreTool(memory=mem), MemorySearchTool(memory=mem)]
