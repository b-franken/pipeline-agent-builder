"""Knowledge retrieval with re-ranking support."""

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field


@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: dict[str, Any]
    source: str


class KnowledgeRetriever:
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        vector_store: Any = None,
        use_reranking: bool = False,
        distance_threshold: float = 1.5,
    ) -> None:
        self._collection_name = collection_name
        self._vector_store = vector_store
        self._use_reranking = use_reranking
        self._distance_threshold = distance_threshold
        self._reranker: Any = None

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            from src.memory.vector_store import VectorMemory

            self._vector_store = VectorMemory(collection_name=self._collection_name)
        return self._vector_store

    def _get_reranker(self) -> Any:
        if self._reranker is None and self._use_reranking:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except ImportError:
                self._use_reranking = False
                return None
        return self._reranker

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        include_scores: bool = True,
    ) -> list[RetrievalResult]:
        vs = self._get_vector_store()
        fetch_k = k * 3 if self._use_reranking else k

        if filter_metadata:
            results = vs._get_vectorstore().similarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter_metadata,
            )
        else:
            results = vs._get_vectorstore().similarity_search_with_score(
                query,
                k=fetch_k,
            )

        retrieval_results = []
        for doc, distance in results:
            if distance > self._distance_threshold:
                continue
            similarity = 1.0 / (1.0 + distance)
            retrieval_results.append(
                RetrievalResult(
                    content=doc.page_content,
                    score=similarity,
                    metadata=doc.metadata,
                    source=doc.metadata.get("source", "unknown"),
                )
            )

        if self._use_reranking and len(retrieval_results) > 0:
            retrieval_results = self._rerank(query, retrieval_results, k)

        return retrieval_results[:k]

    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        k: int,
    ) -> list[RetrievalResult]:
        reranker = self._get_reranker()
        if reranker is None:
            return results

        pairs = [(query, r.content) for r in results]
        scores = reranker.predict(pairs)

        for result, score in zip(results, scores, strict=False):
            result.score = float(score)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def search_by_document(
        self,
        document_id: str,
        query: str | None = None,
        k: int = 10,
    ) -> list[RetrievalResult]:
        return self.search(
            query=query or "",
            k=k,
            filter_metadata={"document_id": document_id},
        )

    def get_context(
        self,
        query: str,
        max_tokens: int = 8000,
        k: int = 10,
    ) -> str:
        results = self.search(query, k=k)
        if not results:
            return ""

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4

        for result in results:
            if total_chars + len(result.content) > max_chars:
                break
            context_parts.append(f"[Source: {result.source}]\n{result.content}")
            total_chars += len(result.content)

        return "\n\n---\n\n".join(context_parts)


_retriever_registry: dict[str, KnowledgeRetriever] = {}


def get_retriever(collection_name: str = "knowledge_base") -> KnowledgeRetriever:
    global _retriever_registry
    if collection_name not in _retriever_registry:
        _retriever_registry[collection_name] = KnowledgeRetriever(collection_name=collection_name)
    return _retriever_registry[collection_name]


def clear_retriever_registry() -> None:
    global _retriever_registry
    _retriever_registry.clear()


class KnowledgeSearchTool(BaseTool):
    name: str = "search_knowledge"
    description: str = (
        "Search the knowledge base for relevant information. "
        "Use this to find answers from uploaded documents, code, and notes."
    )
    retriever: KnowledgeRetriever = Field(default_factory=get_retriever)

    def _run(self, query: str) -> str:
        results = self.retriever.search(query, k=8)
        if not results:
            return "No relevant information found in the knowledge base."

        output_parts = []
        for i, result in enumerate(results, 1):
            content = result.content[:2000]
            if len(result.content) > 2000:
                content += "..."
            output_parts.append(f"[{i}] (score: {result.score:.2f}, source: {result.source})\n{content}")
        return "\n\n".join(output_parts)

    async def _arun(self, query: str) -> str:
        return self._run(query)


def create_knowledge_tool(collection_name: str = "knowledge_base") -> KnowledgeSearchTool:
    retriever = KnowledgeRetriever(collection_name=collection_name)
    return KnowledgeSearchTool(retriever=retriever)
