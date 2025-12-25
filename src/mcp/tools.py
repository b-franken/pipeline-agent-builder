import logging
from typing import Final

from langchain_core.tools import BaseTool

from src.mcp.client import get_mcp_client_manager, is_mcp_available

logger: Final = logging.getLogger(__name__)


async def get_mcp_tools_for_servers(server_ids: list[str]) -> list[BaseTool]:
    if not server_ids or not is_mcp_available():
        return []

    from src.storage import get_repository

    repo = await get_repository()
    servers = await repo.get_mcp_servers_by_ids(server_ids)

    if not servers:
        return []

    manager = get_mcp_client_manager()

    for server in servers:
        if not manager.is_server_configured(server.name):
            manager.add_server(server)

    server_names = [s.name for s in servers]
    return await manager.get_tools(server_names)


async def get_mcp_tools_for_agent(agent_id: str) -> list[BaseTool]:
    if not is_mcp_available():
        return []

    from src.storage import get_repository

    repo = await get_repository()
    agent_config = await repo.get_agent_config(agent_id)

    if agent_config is None or not agent_config.mcp_server_ids:
        return []

    return await get_mcp_tools_for_servers(agent_config.mcp_server_ids)


async def get_mcp_tools_for_team(team_id: str) -> list[BaseTool]:
    if not is_mcp_available():
        return []

    from src.storage import get_repository

    repo = await get_repository()
    team = await repo.get_team(team_id)

    if team is None or not team.mcp_server_ids:
        return []

    return await get_mcp_tools_for_servers(team.mcp_server_ids)
