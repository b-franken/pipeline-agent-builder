import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest

HAS_CHROMADB = importlib.util.find_spec("langchain_chroma") is not None

pytestmark = pytest.mark.skipif(not HAS_CHROMADB, reason="langchain_chroma not installed")


class MockRetrievalResult:
    def __init__(self, content: str, source: str, score: float) -> None:
        self.content = content
        self.source = source
        self.score = score


@pytest.fixture(autouse=True)
def mock_chromadb():
    mock_chroma = MagicMock()
    with patch.dict(sys.modules, {"chromadb": mock_chroma}):
        yield mock_chroma


class TestUnifiedSearchTool:
    def test_tool_name(self, mock_chromadb: MagicMock) -> None:
        with (
            patch("src.tools.knowledge.get_retriever") as mock_get_ret,
            patch("src.tools.knowledge.get_memory") as mock_get_mem,
        ):
            mock_get_ret.return_value = MagicMock()
            mock_get_mem.return_value = MagicMock()
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=MagicMock(), memory=MagicMock())
            assert tool.name == "search"

    def test_tool_description(self, mock_chromadb: MagicMock) -> None:
        with (
            patch("src.tools.knowledge.get_retriever") as mock_get_ret,
            patch("src.tools.knowledge.get_memory") as mock_get_mem,
        ):
            mock_get_ret.return_value = MagicMock()
            mock_get_mem.return_value = MagicMock()
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=MagicMock(), memory=MagicMock())
            assert "knowledge base" in tool.description.lower()
            assert "memory" in tool.description.lower()

    def test_search_returns_knowledge_base_results(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockRetrievalResult(
                content="Test content from KB",
                source="test.txt",
                score=0.95,
            )
        ]
        mock_memory = MagicMock()
        mock_memory.search_with_scores.return_value = []

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = tool._run("test query")

        assert "Knowledge Base" in result
        assert "Test content from KB" in result
        assert "test.txt" in result
        assert "0.95" in result

    def test_search_returns_memory_results(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_memory = MagicMock()
        mock_memory.search_with_scores.return_value = [("Memory content here", 0.85)]

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = tool._run("test query")

        assert "Memory" in result
        assert "Memory content here" in result
        assert "0.85" in result

    def test_search_no_results(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_memory = MagicMock()
        mock_memory.search_with_scores.return_value = []

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = tool._run("test query")

        assert result == "No relevant information found."

    def test_search_truncates_long_content(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        long_content = "x" * 2000
        mock_retriever.search.return_value = [
            MockRetrievalResult(
                content=long_content,
                source="large.txt",
                score=0.90,
            )
        ]
        mock_memory = MagicMock()
        mock_memory.search_with_scores.return_value = []

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = tool._run("test query")

        assert "..." in result
        assert len(result) < len(long_content) + 500

    def test_search_handles_knowledge_base_exception(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = Exception("KB error")
        mock_memory = MagicMock()
        mock_memory.search_with_scores.return_value = [("Memory result", 0.85)]

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = tool._run("test query")

        assert "Memory" in result
        assert "Memory result" in result

    def test_search_handles_memory_exception(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            MockRetrievalResult(
                content="KB result",
                source="test.txt",
                score=0.90,
            )
        ]
        mock_memory = MagicMock()
        mock_memory.search_with_scores.side_effect = Exception("Memory error")

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = tool._run("test query")

        assert "Knowledge Base" in result
        assert "KB result" in result

    def test_search_handles_both_exceptions(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = Exception("KB error")
        mock_memory = MagicMock()
        mock_memory.search_with_scores.side_effect = Exception("Memory error")

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = tool._run("test query")

        assert result == "No relevant information found."

    @pytest.mark.asyncio
    async def test_arun_calls_run(self, mock_chromadb: MagicMock) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_memory = MagicMock()
        mock_memory.search_with_scores.return_value = []

        with patch("src.tools.knowledge.get_retriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import UnifiedSearchTool

            tool = UnifiedSearchTool(knowledge_retriever=mock_retriever, memory=mock_memory)
            result = await tool._arun("test query")

        assert result == "No relevant information found."


class TestStoreFactTool:
    def test_tool_name(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import StoreFactTool

            mock_memory = MagicMock()
            tool = StoreFactTool(memory=mock_memory)
            assert tool.name == "store_fact"

    def test_tool_description(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import StoreFactTool

            mock_memory = MagicMock()
            tool = StoreFactTool(memory=mock_memory)
            assert "fact" in tool.description.lower()

    def test_store_fact(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import StoreFactTool

            mock_memory = MagicMock()
            mock_memory.store.return_value = "doc_12345678_abcd"

            tool = StoreFactTool(memory=mock_memory)
            result = tool._run("Python is awesome")

            mock_memory.store.assert_called_once_with("Python is awesome")
            assert "Stored fact" in result
            assert "doc_1234" in result

    @pytest.mark.asyncio
    async def test_arun_calls_run(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import StoreFactTool

            mock_memory = MagicMock()
            mock_memory.store.return_value = "doc_12345678"

            tool = StoreFactTool(memory=mock_memory)
            result = await tool._arun("Test fact")

            assert "Stored fact" in result


class TestCreateKnowledgeTools:
    def test_creates_both_tools_by_default(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.KnowledgeRetriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import create_knowledge_tools

            tools = create_knowledge_tools()
            assert len(tools) == 2
            tool_names = [t.name for t in tools]
            assert "search" in tool_names
            assert "store_fact" in tool_names

    def test_creates_only_search_tool(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.KnowledgeRetriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import create_knowledge_tools

            tools = create_knowledge_tools(include_store=False)
            assert len(tools) == 1
            assert tools[0].name == "search"

    def test_creates_only_store_tool(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.KnowledgeRetriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import create_knowledge_tools

            tools = create_knowledge_tools(include_unified_search=False)
            assert len(tools) == 1
            assert tools[0].name == "store_fact"

    def test_creates_no_tools(self, mock_chromadb: MagicMock) -> None:
        with patch("src.tools.knowledge.KnowledgeRetriever"), patch("src.tools.knowledge.get_memory"):
            from src.tools.knowledge import create_knowledge_tools

            tools = create_knowledge_tools(include_unified_search=False, include_store=False)
            assert len(tools) == 0

    def test_uses_custom_collection_names(self, mock_chromadb: MagicMock) -> None:
        with (
            patch("src.tools.knowledge.KnowledgeRetriever") as mock_retriever_cls,
            patch("src.tools.knowledge.get_memory") as mock_get_memory,
        ):
            from src.tools.knowledge import create_knowledge_tools

            create_knowledge_tools(collection_name="custom_kb", memory_collection="custom_memory")
            mock_retriever_cls.assert_called_once_with(collection_name="custom_kb")
            mock_get_memory.assert_called_once_with(collection_name="custom_memory")
