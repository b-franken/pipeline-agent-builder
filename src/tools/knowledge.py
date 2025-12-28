from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Final

from langchain_core.tools import BaseTool
from pydantic import Field

from src.knowledge.retriever import KnowledgeRetriever, get_retriever
from src.memory.vector_store import VectorMemory, get_memory

if TYPE_CHECKING:
    from src.storage.repository import StorageRepository

logger: Final = logging.getLogger(__name__)


class UnifiedSearchTool(BaseTool):
    name: str = "search"
    description: str = (
        "Search for relevant information across all sources. "
        "This searches both the knowledge base (documents, code) and "
        "memory (past conversations, learned facts)."
    )
    knowledge_retriever: KnowledgeRetriever = Field(default_factory=get_retriever)
    memory: VectorMemory = Field(default_factory=get_memory)

    def _run(self, query: str) -> str:
        results: list[str] = []

        try:
            kb_results = self.knowledge_retriever.search(query, k=5)
            if kb_results:
                results.append("=== Knowledge Base ===")
                for i, r in enumerate(kb_results, 1):
                    content = r.content[:1500] + "..." if len(r.content) > 1500 else r.content
                    results.append(f"[{i}] (source: {r.source}, score: {r.score:.2f})\n{content}")
        except Exception:
            logger.exception("Knowledge base search failed for query: %s", query[:100])

        try:
            mem_results = self.memory.search_with_scores(query, k=5)
            if mem_results:
                results.append("\n=== Memory ===")
                for i, (content, score) in enumerate(mem_results, 1):
                    content_preview = content[:1500] + "..." if len(content) > 1500 else content
                    results.append(f"[{i}] (score: {score:.2f})\n{content_preview}")
        except Exception:
            logger.exception("Memory search failed for query: %s", query[:100])

        if not results:
            return "No relevant information found."

        return "\n\n".join(results)

    async def _arun(self, query: str) -> str:
        return self._run(query)


async def _store_fact_to_sql(
    fact: str,
    source_agent: str | None = None,
    source_task_id: str | None = None,
) -> str | None:
    try:
        from src.storage import get_repository

        repo: StorageRepository = await get_repository()
        record = await repo.store_fact(
            fact=fact,
            category="agent_learned",
            source_agent=source_agent,
            source_task_id=source_task_id,
            confidence=1.0,
        )
        return record.id
    except Exception:
        logger.exception("Failed to store fact in SQL database")
        return None


class StoreFactTool(BaseTool):
    name: str = "store_fact"
    description: str = "Store an important fact, learning, or user preference for later recall."
    memory: VectorMemory = Field(default_factory=get_memory)
    source_agent: str | None = Field(default=None)
    source_task_id: str | None = Field(default=None)

    def _run(self, fact: str) -> str:
        vector_id = self.memory.store(fact)
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                _store_fact_to_sql(fact, self.source_agent, self.source_task_id),
                loop,
            )
            sql_id = future.result(timeout=5.0)
        except RuntimeError:
            sql_id = asyncio.run(_store_fact_to_sql(fact, self.source_agent, self.source_task_id))
        except Exception:
            logger.exception("Failed to store fact in SQL")
            sql_id = None

        if sql_id:
            return f"Stored fact (vector: {vector_id[:8]}..., sql: {sql_id[:8]}...)"
        return f"Stored fact (vector: {vector_id[:8]}...)"

    async def _arun(self, fact: str) -> str:
        vector_id = self.memory.store(fact)
        sql_id = await _store_fact_to_sql(fact, self.source_agent, self.source_task_id)
        if sql_id:
            return f"Stored fact (vector: {vector_id[:8]}..., sql: {sql_id[:8]}...)"
        return f"Stored fact (vector: {vector_id[:8]}...)"


def create_knowledge_tools(
    collection_name: str = "knowledge_base",
    memory_collection: str = "agent_memory",
    include_unified_search: bool = True,
    include_store: bool = True,
    source_agent: str | None = None,
    source_task_id: str | None = None,
) -> list[BaseTool]:
    retriever = KnowledgeRetriever(collection_name=collection_name)
    memory = get_memory(collection_name=memory_collection)

    tools: list[BaseTool] = []

    if include_unified_search:
        tools.append(
            UnifiedSearchTool(
                knowledge_retriever=retriever,
                memory=memory,
            )
        )

    if include_store:
        tools.append(
            StoreFactTool(
                memory=memory,
                source_agent=source_agent,
                source_task_id=source_task_id,
            )
        )

    return tools
