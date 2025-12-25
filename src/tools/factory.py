import logging
from typing import Final

from langchain_core.tools import BaseTool

logger: Final = logging.getLogger(__name__)


def _create_web_search() -> BaseTool:
    from src.tools.search import create_search_tool

    return create_search_tool()


def _create_tavily_search() -> BaseTool:
    from src.tools.search import create_tavily_search_tool

    return create_tavily_search_tool()


def _create_execute_python_safe() -> BaseTool:
    from src.tools.code import execute_python_safe

    return execute_python_safe


def _create_format_code() -> BaseTool:
    from src.tools.code import format_code

    return format_code


def _create_analyze_code() -> BaseTool:
    from src.tools.code import analyze_code

    return analyze_code


def _create_search_knowledge(source_agent: str | None, source_task_id: str | None) -> BaseTool:
    from src.tools.knowledge import UnifiedSearchTool

    return UnifiedSearchTool()


def _create_store_fact(source_agent: str | None, source_task_id: str | None) -> BaseTool:
    from src.memory.vector_store import get_memory
    from src.tools.knowledge import StoreFactTool

    return StoreFactTool(
        memory=get_memory(),
        source_agent=source_agent,
        source_task_id=source_task_id,
    )


def _create_store_memory() -> BaseTool:
    from src.memory import MemoryStoreTool

    return MemoryStoreTool()


def _create_search_memory() -> BaseTool:
    from src.memory import MemorySearchTool

    return MemorySearchTool()


def _create_generate_pdf() -> BaseTool:
    from src.tools.documents import GeneratePDFTool

    return GeneratePDFTool()


def _create_generate_excel() -> BaseTool:
    from src.tools.documents import GenerateExcelTool

    return GenerateExcelTool()


def _create_generate_csv() -> BaseTool:
    from src.tools.documents import GenerateCSVTool

    return GenerateCSVTool()


def _create_list_document_templates() -> BaseTool:
    from src.tools.documents import list_templates

    return list_templates


def create_tools_for_agent(
    enabled_tool_ids: list[str],
    source_agent: str | None = None,
    source_task_id: str | None = None,
) -> list[BaseTool]:
    if not enabled_tool_ids:
        return []

    tools: list[BaseTool] = []
    loaded_ids: set[str] = set()

    for tool_id in enabled_tool_ids:
        if tool_id in loaded_ids:
            continue

        try:
            tool = _create_tool_by_id(tool_id, source_agent, source_task_id)
            if tool:
                tools.append(tool)
                loaded_ids.add(tool_id)
        except Exception:
            logger.exception("Failed to load tool: %s", tool_id)

    return tools


def _create_tool_by_id(
    tool_id: str,
    source_agent: str | None,
    source_task_id: str | None,
) -> BaseTool | None:
    match tool_id:
        case "web_search":
            return _create_web_search()
        case "tavily_search":
            return _create_tavily_search()
        case "execute_python" | "execute_python_safe":
            return _create_execute_python_safe()
        case "format_code":
            return _create_format_code()
        case "analyze_code":
            return _create_analyze_code()
        case "search_knowledge" | "search":
            return _create_search_knowledge(source_agent, source_task_id)
        case "store_fact":
            return _create_store_fact(source_agent, source_task_id)
        case "store_memory":
            return _create_store_memory()
        case "search_memory":
            return _create_search_memory()
        case "generate_pdf":
            return _create_generate_pdf()
        case "generate_excel":
            return _create_generate_excel()
        case "generate_csv":
            return _create_generate_csv()
        case "list_document_templates":
            return _create_list_document_templates()
        case _:
            logger.warning("Unknown tool ID: %s", tool_id)
            return None


async def create_tools_for_agent_async(
    agent_id: str,
    enabled_tool_ids: list[str],
    mcp_server_ids: list[str] | None = None,
    source_task_id: str | None = None,
) -> list[BaseTool]:
    tools = create_tools_for_agent(
        enabled_tool_ids=enabled_tool_ids,
        source_agent=agent_id,
        source_task_id=source_task_id,
    )

    if mcp_server_ids:
        try:
            from src.mcp.client import create_mcp_client_for_servers, is_mcp_available
            from src.storage import get_repository

            if is_mcp_available():
                repo = await get_repository()
                servers_to_load = []
                for server_id in mcp_server_ids:
                    server = await repo.get_mcp_server(server_id)
                    if server and server.is_active:
                        servers_to_load.append(server)

                if servers_to_load:
                    mcp_tools = await create_mcp_client_for_servers(servers_to_load)
                    tools.extend(mcp_tools)
        except Exception:
            logger.exception("Failed to load MCP tools for agent: %s", agent_id)

    return tools
