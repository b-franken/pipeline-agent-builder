from unittest.mock import MagicMock, patch

import pytest

from src.tools.factory import create_tools_for_agent


class TestCreateToolsForAgent:
    def test_empty_tool_ids_returns_empty_list(self) -> None:
        result = create_tools_for_agent([])
        assert result == []

    def test_unknown_tool_id_is_skipped(self) -> None:
        with patch("src.tools.factory.logger"):
            result = create_tools_for_agent(["unknown_tool_xyz"])
            assert result == []

    def test_duplicate_tool_ids_only_loads_once(self) -> None:
        with patch("src.tools.factory._create_web_search") as mock_search:
            mock_tool = MagicMock()
            mock_tool.name = "web_search"
            mock_search.return_value = mock_tool

            result = create_tools_for_agent(["web_search", "web_search"])

            mock_search.assert_called_once()
            assert len(result) == 1

    def test_loads_search_tools(self) -> None:
        with patch("src.tools.factory._create_web_search") as mock_search:
            mock_tool = MagicMock()
            mock_tool.name = "web_search"
            mock_search.return_value = mock_tool

            result = create_tools_for_agent(["web_search"])

            assert len(result) == 1
            assert result[0].name == "web_search"

    def test_loads_tavily_search_tool(self) -> None:
        with patch("src.tools.factory._create_tavily_search") as mock_tavily:
            mock_tool = MagicMock()
            mock_tool.name = "tavily_search"
            mock_tavily.return_value = mock_tool

            result = create_tools_for_agent(["tavily_search"])

            mock_tavily.assert_called_once()
            assert len(result) == 1

    def test_loads_code_tools(self) -> None:
        with patch("src.tools.factory._create_execute_python_safe") as mock_code:
            mock_tool = MagicMock()
            mock_tool.name = "execute_python_safe"
            mock_code.return_value = mock_tool

            result = create_tools_for_agent(["execute_python"])

            mock_code.assert_called_once()
            assert len(result) == 1

    def test_loads_format_code_tool(self) -> None:
        with patch("src.tools.factory._create_format_code") as mock_format:
            mock_tool = MagicMock()
            mock_tool.name = "format_code"
            mock_format.return_value = mock_tool

            result = create_tools_for_agent(["format_code"])

            mock_format.assert_called_once()
            assert len(result) == 1

    def test_loads_analyze_code_tool(self) -> None:
        with patch("src.tools.factory._create_analyze_code") as mock_analyze:
            mock_tool = MagicMock()
            mock_tool.name = "analyze_code"
            mock_analyze.return_value = mock_tool

            result = create_tools_for_agent(["analyze_code"])

            mock_analyze.assert_called_once()
            assert len(result) == 1

    def test_loads_knowledge_tools(self) -> None:
        with patch("src.tools.factory._create_search_knowledge") as mock_knowledge:
            mock_tool = MagicMock()
            mock_tool.name = "search_knowledge"
            mock_knowledge.return_value = mock_tool

            result = create_tools_for_agent(
                ["search_knowledge"],
                source_agent="test_agent",
                source_task_id="task_123",
            )

            mock_knowledge.assert_called_once_with("test_agent", "task_123")
            assert len(result) == 1

    def test_loads_store_fact_tool(self) -> None:
        with patch("src.tools.factory._create_store_fact") as mock_store:
            mock_tool = MagicMock()
            mock_tool.name = "store_fact"
            mock_store.return_value = mock_tool

            result = create_tools_for_agent(
                ["store_fact"],
                source_agent="test_agent",
                source_task_id="task_123",
            )

            mock_store.assert_called_once_with("test_agent", "task_123")
            assert len(result) == 1

    def test_loads_memory_tools(self) -> None:
        with patch("src.tools.factory._create_store_memory") as mock_memory:
            mock_tool = MagicMock()
            mock_tool.name = "store_memory"
            mock_memory.return_value = mock_tool

            result = create_tools_for_agent(["store_memory"])

            mock_memory.assert_called_once()
            assert len(result) == 1

    def test_loads_search_memory_tool(self) -> None:
        with patch("src.tools.factory._create_search_memory") as mock_search:
            mock_tool = MagicMock()
            mock_tool.name = "search_memory"
            mock_search.return_value = mock_tool

            result = create_tools_for_agent(["search_memory"])

            mock_search.assert_called_once()
            assert len(result) == 1

    def test_loads_document_tools(self) -> None:
        with patch("src.tools.factory._create_generate_pdf") as mock_pdf:
            mock_tool = MagicMock()
            mock_tool.name = "generate_pdf"
            mock_pdf.return_value = mock_tool

            result = create_tools_for_agent(["generate_pdf"])

            mock_pdf.assert_called_once()
            assert len(result) == 1

    def test_loads_generate_excel_tool(self) -> None:
        with patch("src.tools.factory._create_generate_excel") as mock_excel:
            mock_tool = MagicMock()
            mock_tool.name = "generate_excel"
            mock_excel.return_value = mock_tool

            result = create_tools_for_agent(["generate_excel"])

            mock_excel.assert_called_once()
            assert len(result) == 1

    def test_loads_generate_csv_tool(self) -> None:
        with patch("src.tools.factory._create_generate_csv") as mock_csv:
            mock_tool = MagicMock()
            mock_tool.name = "generate_csv"
            mock_csv.return_value = mock_tool

            result = create_tools_for_agent(["generate_csv"])

            mock_csv.assert_called_once()
            assert len(result) == 1

    def test_loads_list_templates_tool(self) -> None:
        with patch("src.tools.factory._create_list_document_templates") as mock_list:
            mock_tool = MagicMock()
            mock_tool.name = "list_document_templates"
            mock_list.return_value = mock_tool

            result = create_tools_for_agent(["list_document_templates"])

            mock_list.assert_called_once()
            assert len(result) == 1

    def test_handles_exception_gracefully(self) -> None:
        with (
            patch("src.tools.factory._create_web_search") as mock_search,
            patch("src.tools.factory.logger"),
        ):
            mock_search.side_effect = RuntimeError("Test error")

            result = create_tools_for_agent(["web_search"])

            assert result == []

    def test_multiple_different_tool_types(self) -> None:
        with (
            patch("src.tools.factory._create_web_search") as mock_search,
            patch("src.tools.factory._create_execute_python_safe") as mock_code,
        ):
            mock_search_tool = MagicMock()
            mock_search_tool.name = "web_search"
            mock_search.return_value = mock_search_tool

            mock_code_tool = MagicMock()
            mock_code_tool.name = "execute_python_safe"
            mock_code.return_value = mock_code_tool

            result = create_tools_for_agent(["web_search", "execute_python"])

            assert len(result) == 2
            mock_search.assert_called_once()
            mock_code.assert_called_once()

    def test_search_alias_maps_to_search_knowledge(self) -> None:
        with patch("src.tools.factory._create_search_knowledge") as mock_search:
            mock_tool = MagicMock()
            mock_tool.name = "search"
            mock_search.return_value = mock_tool

            result = create_tools_for_agent(["search"])

            mock_search.assert_called_once()
            assert len(result) == 1

    def test_execute_python_safe_alias(self) -> None:
        with patch("src.tools.factory._create_execute_python_safe") as mock_code:
            mock_tool = MagicMock()
            mock_tool.name = "execute_python_safe"
            mock_code.return_value = mock_tool

            result = create_tools_for_agent(["execute_python_safe"])

            mock_code.assert_called_once()
            assert len(result) == 1


class TestCreateToolsForAgentAsync:
    @pytest.mark.asyncio
    async def test_async_without_mcp(self) -> None:
        from src.tools.factory import create_tools_for_agent_async

        with patch("src.tools.factory._create_web_search") as mock_search:
            mock_tool = MagicMock()
            mock_tool.name = "web_search"
            mock_search.return_value = mock_tool

            result = await create_tools_for_agent_async(
                agent_id="test_agent",
                enabled_tool_ids=["web_search"],
            )

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_async_with_mcp_not_available(self) -> None:
        from src.tools.factory import create_tools_for_agent_async

        with (
            patch("src.tools.factory._create_web_search") as mock_search,
            patch("src.mcp.client.is_mcp_available", return_value=False),
        ):
            mock_tool = MagicMock()
            mock_tool.name = "web_search"
            mock_search.return_value = mock_tool

            result = await create_tools_for_agent_async(
                agent_id="test_agent",
                enabled_tool_ids=["web_search"],
                mcp_server_ids=["server1"],
            )

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_async_with_mcp_available(self) -> None:
        from unittest.mock import AsyncMock

        from src.storage.models import MCPServer
        from src.tools.factory import create_tools_for_agent_async

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "mcp_tool"

        mock_server = MCPServer(
            id="server1",
            name="test-server",
            transport="stdio",
            command="npx",
            is_active=True,
        )

        mock_repo = MagicMock()
        mock_repo.get_mcp_server = AsyncMock(return_value=mock_server)

        with (
            patch("src.tools.factory._create_web_search") as mock_search,
            patch("src.mcp.client.is_mcp_available", return_value=True),
            patch("src.storage.get_repository", AsyncMock(return_value=mock_repo)),
            patch(
                "src.mcp.client.create_mcp_client_for_servers",
                AsyncMock(return_value=[mock_mcp_tool]),
            ) as mock_create_client,
        ):
            mock_tool = MagicMock()
            mock_tool.name = "web_search"
            mock_search.return_value = mock_tool

            result = await create_tools_for_agent_async(
                agent_id="test_agent",
                enabled_tool_ids=["web_search"],
                mcp_server_ids=["server1"],
            )

            assert len(result) == 2
            mock_create_client.assert_called_once_with([mock_server])

    @pytest.mark.asyncio
    async def test_async_with_empty_mcp_server_ids(self) -> None:
        from src.tools.factory import create_tools_for_agent_async

        with patch("src.tools.factory._create_web_search") as mock_search:
            mock_tool = MagicMock()
            mock_tool.name = "web_search"
            mock_search.return_value = mock_tool

            result = await create_tools_for_agent_async(
                agent_id="test_agent",
                enabled_tool_ids=["web_search"],
                mcp_server_ids=[],
            )

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_async_mcp_exception_handled(self) -> None:
        from unittest.mock import AsyncMock

        from src.tools.factory import create_tools_for_agent_async

        with (
            patch("src.tools.factory._create_web_search") as mock_search,
            patch("src.mcp.client.is_mcp_available", return_value=True),
            patch(
                "src.storage.get_repository",
                AsyncMock(side_effect=RuntimeError("DB error")),
            ),
            patch("src.tools.factory.logger"),
        ):
            mock_tool = MagicMock()
            mock_tool.name = "web_search"
            mock_search.return_value = mock_tool

            result = await create_tools_for_agent_async(
                agent_id="test_agent",
                enabled_tool_ids=["web_search"],
                mcp_server_ids=["server1"],
            )

            assert len(result) == 1
