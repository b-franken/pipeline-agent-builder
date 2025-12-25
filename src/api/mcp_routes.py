import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.mcp.client import get_mcp_client_manager, is_mcp_available
from src.storage import get_repository

logger: Final = logging.getLogger(__name__)
router = APIRouter(prefix="/mcp", tags=["mcp"])

MCP_REGISTRY_URL: Final = "https://registry.modelcontextprotocol.io/v0/servers"
REGISTRY_CACHE_TTL: Final = timedelta(hours=1)

_registry_cache: dict[str, Any] = {}
_registry_cache_time: datetime | None = None


class MCPServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    transport: str = Field(..., pattern="^(stdio|http|streamable_http|sse)$")
    description: str = ""
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env_vars: dict[str, str] = Field(default_factory=dict)


class MCPServerUpdate(BaseModel):
    name: str | None = None
    transport: str | None = Field(default=None, pattern="^(stdio|http|streamable_http|sse)$")
    description: str | None = None
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    env_vars: dict[str, str] | None = None
    is_active: bool | None = None


class MCPServerResponse(BaseModel):
    id: str
    name: str
    transport: str
    description: str
    command: str | None
    args: list[str]
    url: str | None
    headers: dict[str, str]
    is_active: bool

    model_config = {"from_attributes": True}


class MCPToolInfo(BaseModel):
    name: str
    description: str
    server: str | None = None


class MCPStatusResponse(BaseModel):
    available: bool
    configured_servers: list[str]
    total_tools: int


@router.get("/status", response_model=MCPStatusResponse)
async def get_mcp_status() -> MCPStatusResponse:
    manager = get_mcp_client_manager()
    configured = manager.get_configured_servers()

    tool_count = 0
    if is_mcp_available() and configured:
        try:
            tools = await manager.get_tools()
            tool_count = len(tools)
        except Exception:
            pass

    return MCPStatusResponse(
        available=is_mcp_available(),
        configured_servers=configured,
        total_tools=tool_count,
    )


@router.get("/servers", response_model=list[MCPServerResponse])
async def list_mcp_servers(active_only: bool = False) -> list[MCPServerResponse]:
    repo = await get_repository()
    servers = await repo.list_mcp_servers(active_only=active_only)
    return [MCPServerResponse.model_validate(s) for s in servers]


@router.post("/servers", response_model=MCPServerResponse, status_code=201)
async def create_mcp_server(data: MCPServerCreate) -> MCPServerResponse:
    if data.transport == "stdio" and not data.command:
        raise HTTPException(status_code=400, detail="stdio transport requires command")
    if data.transport in ("http", "streamable_http", "sse") and not data.url:
        raise HTTPException(status_code=400, detail=f"{data.transport} transport requires url")

    repo = await get_repository()

    existing = await repo.get_mcp_server_by_name(data.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Server with name '{data.name}' already exists")

    server = await repo.create_mcp_server(
        name=data.name,
        transport=data.transport,
        description=data.description,
        command=data.command,
        args=data.args,
        url=data.url,
        headers=data.headers,
        env_vars=data.env_vars,
    )

    manager = get_mcp_client_manager()
    manager.add_server(server)

    return MCPServerResponse.model_validate(server)


@router.get("/servers/{server_id}", response_model=MCPServerResponse)
async def get_mcp_server(server_id: str) -> MCPServerResponse:
    repo = await get_repository()
    server = await repo.get_mcp_server(server_id)
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return MCPServerResponse.model_validate(server)


@router.patch("/servers/{server_id}", response_model=MCPServerResponse)
async def update_mcp_server(server_id: str, data: MCPServerUpdate) -> MCPServerResponse:
    repo = await get_repository()

    server = await repo.update_mcp_server(
        server_id=server_id,
        name=data.name,
        transport=data.transport,
        description=data.description,
        command=data.command,
        args=data.args,
        url=data.url,
        headers=data.headers,
        env_vars=data.env_vars,
        is_active=data.is_active,
    )

    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")

    manager = get_mcp_client_manager()
    if server.is_active:
        manager.add_server(server)
    else:
        manager.remove_server(server.name)

    return MCPServerResponse.model_validate(server)


@router.delete("/servers/{server_id}", status_code=204)
async def delete_mcp_server(server_id: str) -> None:
    repo = await get_repository()

    server = await repo.get_mcp_server(server_id)
    if server:
        manager = get_mcp_client_manager()
        manager.remove_server(server.name)

    deleted = await repo.delete_mcp_server(server_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="MCP server not found")


@router.get("/servers/{server_id}/tools", response_model=list[MCPToolInfo])
async def get_server_tools(server_id: str) -> list[MCPToolInfo]:
    if not is_mcp_available():
        raise HTTPException(status_code=503, detail="MCP adapters not installed")

    repo = await get_repository()
    server = await repo.get_mcp_server(server_id)
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")

    manager = get_mcp_client_manager()
    if not manager.is_server_configured(server.name):
        manager.add_server(server)

    try:
        tools = await manager.get_tools_for_server(server.name)
        return [
            MCPToolInfo(
                name=tool.name,
                description=tool.description or "",
                server=server.name,
            )
            for tool in tools
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {e}") from e


@router.post("/servers/{server_id}/test")
async def test_mcp_server(server_id: str) -> dict[str, Any]:
    if not is_mcp_available():
        raise HTTPException(status_code=503, detail="MCP adapters not installed")

    repo = await get_repository()
    server = await repo.get_mcp_server(server_id)
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")

    manager = get_mcp_client_manager()
    if not manager.is_server_configured(server.name):
        manager.add_server(server)

    try:
        tools = await manager.get_tools_for_server(server.name)
        return {
            "success": True,
            "server": server.name,
            "tool_count": len(tools),
            "tools": [t.name for t in tools],
        }
    except Exception as e:
        return {
            "success": False,
            "server": server.name,
            "error": f"{type(e).__name__}: {e}",
        }


POPULAR_MCP_SERVERS: Final[list[dict[str, Any]]] = [
    {
        "id": "github",
        "name": "GitHub",
        "description": "Manage repositories, issues, pull requests, and more",
        "package": "@modelcontextprotocol/server-github",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_vars": {"GITHUB_PERSONAL_ACCESS_TOKEN": ""},
        "category": "developer",
        "icon": "github",
    },
    {
        "id": "filesystem",
        "name": "Filesystem",
        "description": "Secure file operations with configurable access controls",
        "package": "@modelcontextprotocol/server-filesystem",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"],
        "env_vars": {},
        "category": "system",
        "icon": "folder",
    },
    {
        "id": "postgres",
        "name": "PostgreSQL",
        "description": "Read-only database access with schema inspection",
        "package": "@modelcontextprotocol/server-postgres",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
        "env_vars": {},
        "category": "database",
        "icon": "database",
    },
    {
        "id": "memory",
        "name": "Memory",
        "description": "Knowledge graph-based persistent memory",
        "package": "@modelcontextprotocol/server-memory",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env_vars": {},
        "category": "utility",
        "icon": "brain",
    },
    {
        "id": "fetch",
        "name": "Fetch",
        "description": "Web content fetching with robots.txt compliance",
        "package": "@modelcontextprotocol/server-fetch",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "env_vars": {},
        "category": "web",
        "icon": "globe",
    },
    {
        "id": "brave-search",
        "name": "Brave Search",
        "description": "Web and local search via Brave Search API",
        "package": "@modelcontextprotocol/server-brave-search",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env_vars": {"BRAVE_API_KEY": ""},
        "category": "search",
        "icon": "search",
    },
    {
        "id": "slack",
        "name": "Slack",
        "description": "Channel management and messaging for workspaces",
        "package": "@modelcontextprotocol/server-slack",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env_vars": {"SLACK_BOT_TOKEN": "", "SLACK_TEAM_ID": ""},
        "category": "communication",
        "icon": "message-square",
    },
    {
        "id": "puppeteer",
        "name": "Puppeteer",
        "description": "Browser automation and web scraping",
        "package": "@modelcontextprotocol/server-puppeteer",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "env_vars": {},
        "category": "automation",
        "icon": "monitor",
    },
    {
        "id": "sqlite",
        "name": "SQLite",
        "description": "Local database with business intelligence capabilities",
        "package": "@modelcontextprotocol/server-sqlite",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/path/to/db.sqlite"],
        "env_vars": {},
        "category": "database",
        "icon": "database",
    },
    {
        "id": "google-drive",
        "name": "Google Drive",
        "description": "File access and search for Google Drive",
        "package": "@modelcontextprotocol/server-gdrive",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-gdrive"],
        "env_vars": {"GDRIVE_CREDENTIALS_PATH": ""},
        "category": "storage",
        "icon": "hard-drive",
    },
    {
        "id": "sequential-thinking",
        "name": "Sequential Thinking",
        "description": "Dynamic problem-solving through thought sequences",
        "package": "@modelcontextprotocol/server-sequential-thinking",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "env_vars": {},
        "category": "reasoning",
        "icon": "git-branch",
    },
    {
        "id": "everart",
        "name": "EverArt",
        "description": "AI image generation using various models",
        "package": "@modelcontextprotocol/server-everart",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-everart"],
        "env_vars": {"EVERART_API_KEY": ""},
        "category": "creative",
        "icon": "image",
    },
    {
        "id": "terraform",
        "name": "Terraform",
        "description": "HashiCorp Terraform Registry and HCP Terraform workspace management",
        "package": "hashicorp/terraform-mcp-server",
        "transport": "stdio",
        "command": "docker",
        "args": ["run", "-i", "--rm", "hashicorp/terraform-mcp-server"],
        "env_vars": {"TFE_TOKEN": "", "TFE_ADDRESS": "https://app.terraform.io"},
        "category": "infrastructure",
        "icon": "server",
    },
    {
        "id": "azure",
        "name": "Azure",
        "description": "Microsoft Azure cloud services management and deployment",
        "package": "@azure/mcp",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@azure/mcp"],
        "env_vars": {},
        "category": "infrastructure",
        "icon": "cloud",
    },
    {
        "id": "azure-devops",
        "name": "Azure DevOps",
        "description": "Azure DevOps work items, repos, boards, and sprint management",
        "package": "@modelcontextprotocol/server-azure-devops",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-azure-devops"],
        "env_vars": {"AZURE_DEVOPS_ORG_URL": "", "AZURE_DEVOPS_PAT": ""},
        "category": "developer",
        "icon": "git-branch",
    },
    {
        "id": "aws-kb-retrieval",
        "name": "AWS Knowledge Base",
        "description": "Amazon Bedrock knowledge base retrieval and RAG",
        "package": "@modelcontextprotocol/server-aws-kb-retrieval",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-aws-kb-retrieval"],
        "env_vars": {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": "", "AWS_REGION": ""},
        "category": "infrastructure",
        "icon": "cloud",
    },
    {
        "id": "git",
        "name": "Git",
        "description": "Git repository operations - read, search, and analyze local repos",
        "package": "@modelcontextprotocol/server-git",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-git"],
        "env_vars": {},
        "category": "developer",
        "icon": "git-branch",
    },
    {
        "id": "gitlab",
        "name": "GitLab",
        "description": "GitLab API integration for projects, issues, and merge requests",
        "package": "@modelcontextprotocol/server-gitlab",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-gitlab"],
        "env_vars": {"GITLAB_TOKEN": "", "GITLAB_URL": "https://gitlab.com"},
        "category": "developer",
        "icon": "git-branch",
    },
    {
        "id": "sentry",
        "name": "Sentry",
        "description": "Sentry error tracking and performance monitoring",
        "package": "@modelcontextprotocol/server-sentry",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sentry"],
        "env_vars": {"SENTRY_AUTH_TOKEN": "", "SENTRY_ORG": ""},
        "category": "developer",
        "icon": "alert-triangle",
    },
]


class RegistryServerPackage(BaseModel):
    registry_type: str = Field(alias="registryType")
    name: str
    version: str | None = None


class RegistryServerRemote(BaseModel):
    transport_type: str = Field(alias="transportType")
    url: str | None = None


class RegistryServer(BaseModel):
    name: str
    description: str | None = None
    version: str | None = None
    title: str | None = None
    packages: list[RegistryServerPackage] = Field(default_factory=list)
    remotes: list[RegistryServerRemote] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class RegistryResponse(BaseModel):
    servers: list[RegistryServer]
    total: int
    cached: bool
    cache_age_seconds: int | None = None


class PopularServerResponse(BaseModel):
    id: str
    name: str
    description: str
    package: str
    transport: str
    command: str
    args: list[str]
    env_vars: dict[str, str]
    category: str
    icon: str


async def _fetch_registry(search: str | None = None, limit: int = 50) -> dict[str, Any]:
    """Fetch from official MCP registry with caching."""
    global _registry_cache, _registry_cache_time

    cache_key = f"{search or 'all'}:{limit}"
    now = datetime.now(UTC)

    if _registry_cache_time and (now - _registry_cache_time) < REGISTRY_CACHE_TTL and cache_key in _registry_cache:
        cache_age = int((now - _registry_cache_time).total_seconds())
        return {**_registry_cache[cache_key], "cached": True, "cache_age_seconds": cache_age}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params: dict[str, Any] = {"limit": limit}
            if search:
                params["q"] = search

            response = await client.get(MCP_REGISTRY_URL, params=params)
            response.raise_for_status()
            data = response.json()

            _registry_cache[cache_key] = data
            _registry_cache_time = now

            return {**data, "cached": False, "cache_age_seconds": 0}

    except httpx.HTTPStatusError as e:
        logger.warning(f"MCP Registry HTTP error: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"Registry returned {e.response.status_code}") from e
    except httpx.RequestError as e:
        logger.warning(f"MCP Registry request error: {e}")
        raise HTTPException(status_code=502, detail="Failed to connect to MCP registry") from e


@router.get("/registry", response_model=RegistryResponse)
async def get_mcp_registry(
    search: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> RegistryResponse:
    data = await _fetch_registry(search=search, limit=limit)

    servers_raw = data.get("servers", [])
    servers = []

    for entry in servers_raw:
        if not isinstance(entry, dict):
            continue
        s = entry.get("server", entry) if "server" in entry else entry
        if not isinstance(s, dict):
            continue
        try:
            servers.append(
                RegistryServer(
                    name=s.get("name", ""),
                    description=s.get("description"),
                    version=s.get("version"),
                    title=s.get("title"),
                    packages=[
                        RegistryServerPackage(
                            registryType=p.get("registryType", "npm"),
                            name=p.get("name", ""),
                            version=p.get("version"),
                        )
                        for p in s.get("packages", [])
                        if isinstance(p, dict)
                    ],
                    remotes=[
                        RegistryServerRemote(
                            transportType=r.get("type") or r.get("transportType") or "",
                            url=r.get("url"),
                        )
                        for r in s.get("remotes", [])
                        if isinstance(r, dict)
                    ],
                )
            )
        except Exception:
            continue

    return RegistryResponse(
        servers=servers,
        total=len(servers),
        cached=data.get("cached", False),
        cache_age_seconds=data.get("cache_age_seconds"),
    )


@router.get("/registry/popular", response_model=list[PopularServerResponse])
async def get_popular_mcp_servers(category: str | None = None) -> list[PopularServerResponse]:
    """Get curated list of popular MCP servers with pre-configured settings."""
    servers = POPULAR_MCP_SERVERS

    if category:
        servers = [s for s in servers if s["category"] == category]

    return [PopularServerResponse(**s) for s in servers]


@router.get("/registry/categories")
async def get_mcp_categories() -> list[dict[str, Any]]:
    """Get available MCP server categories."""
    categories = {}
    for server in POPULAR_MCP_SERVERS:
        cat = server["category"]
        if cat not in categories:
            categories[cat] = {"id": cat, "name": cat.title(), "count": 0}
        categories[cat]["count"] += 1

    return list(categories.values())
