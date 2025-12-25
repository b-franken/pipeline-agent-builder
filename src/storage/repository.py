import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Final

from packaging.version import Version
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.storage.models import (
    AgentConfig,
    Base,
    DocumentMetadata,
    LearnedFact,
    MCPServer,
    Pipeline,
    SystemSettings,
    TaskRecord,
    Team,
    UserContext,
)

logger: Final = logging.getLogger(__name__)

EXPORT_VERSION: Final = "1.1"
MINIMUM_SUPPORTED_VERSION: Final = "1.0"

_DEFAULT_POOL_SIZE: Final = 5
_DEFAULT_MAX_OVERFLOW: Final = 10
_DEFAULT_POOL_TIMEOUT: Final = 30.0


class StorageRepository:
    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_task(
        self,
        task: str,
        entry_agent: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TaskRecord:
        async with self.session() as session:
            record = TaskRecord(
                task=task,
                entry_agent=entry_agent,
                status="pending",
                task_metadata=metadata,
            )
            session.add(record)
            await session.flush()
            await session.refresh(record)
            return record

    async def start_task(self, task_id: str) -> TaskRecord | None:
        async with self.session() as session:
            result = await session.execute(select(TaskRecord).where(TaskRecord.id == task_id))
            record = result.scalar_one_or_none()
            if record is not None:
                record.status = "running"
                await session.flush()
                await session.refresh(record)
            return record

    async def complete_task(
        self,
        task_id: str,
        result: str,
        agents_used: list[str] | None = None,
        handoff_count: int = 0,
        iteration_count: int = 0,
    ) -> TaskRecord | None:
        async with self.session() as session:
            query_result = await session.execute(select(TaskRecord).where(TaskRecord.id == task_id))
            record = query_result.scalar_one_or_none()
            if record is not None:
                record.status = "completed"
                record.result = result
                record.completed_at = datetime.now(UTC)
                record.agents_used = agents_used
                record.handoff_count = handoff_count
                record.iteration_count = iteration_count
                if record.created_at and record.completed_at:
                    created = record.created_at
                    completed = record.completed_at
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=UTC)
                    if completed.tzinfo is None:
                        completed = completed.replace(tzinfo=UTC)
                    delta = completed - created
                    record.duration_seconds = delta.total_seconds()
                await session.flush()
                await session.refresh(record)
            return record

    async def fail_task(self, task_id: str, error: str) -> TaskRecord | None:
        async with self.session() as session:
            result = await session.execute(select(TaskRecord).where(TaskRecord.id == task_id))
            record = result.scalar_one_or_none()
            if record is not None:
                record.status = "failed"
                record.result = f"Error: {error}"
                record.completed_at = datetime.now(UTC)
                await session.flush()
                await session.refresh(record)
            return record

    async def get_task(self, task_id: str) -> TaskRecord | None:
        async with self.session() as session:
            result = await session.execute(select(TaskRecord).where(TaskRecord.id == task_id))
            return result.scalar_one_or_none()

    async def get_recent_tasks(self, limit: int = 10) -> list[TaskRecord]:
        async with self.session() as session:
            result = await session.execute(select(TaskRecord).order_by(TaskRecord.created_at.desc()).limit(limit))
            return list(result.scalars().all())

    async def set_context(
        self,
        key: str,
        value: str,
        category: str = "general",
    ) -> UserContext:
        async with self.session() as session:
            result = await session.execute(select(UserContext).where(UserContext.key == key))
            existing = result.scalar_one_or_none()
            if existing is not None:
                existing.value = value
                existing.category = category
                existing.updated_at = datetime.now(UTC)
                await session.flush()
                await session.refresh(existing)
                return existing
            context = UserContext(key=key, value=value, category=category)
            session.add(context)
            await session.flush()
            await session.refresh(context)
            return context

    async def get_context(self, key: str) -> str | None:
        async with self.session() as session:
            result = await session.execute(select(UserContext).where(UserContext.key == key))
            context = result.scalar_one_or_none()
            return context.value if context else None

    async def get_all_context(self, category: str | None = None) -> dict[str, str]:
        async with self.session() as session:
            query = select(UserContext)
            if category:
                query = query.where(UserContext.category == category)
            result = await session.execute(query)
            return {c.key: c.value for c in result.scalars().all()}

    async def delete_context(self, key: str) -> bool:
        async with self.session() as session:
            result = await session.execute(delete(UserContext).where(UserContext.key == key))
            rowcount = getattr(result, "rowcount", 0)
            return bool(rowcount and rowcount > 0)

    async def store_fact(
        self,
        fact: str,
        category: str = "general",
        source_agent: str | None = None,
        source_task_id: str | None = None,
        confidence: float = 1.0,
    ) -> LearnedFact:
        async with self.session() as session:
            record = LearnedFact(
                fact=fact,
                category=category,
                source_agent=source_agent,
                source_task_id=source_task_id,
                confidence=confidence,
            )
            session.add(record)
            await session.flush()
            await session.refresh(record)
            return record

    async def get_facts(
        self,
        category: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[LearnedFact]:
        async with self.session() as session:
            query = select(LearnedFact)
            if active_only:
                query = query.where(LearnedFact.is_active == True)  # noqa: E712
            if category:
                query = query.where(LearnedFact.category == category)
            query = query.order_by(LearnedFact.created_at.desc()).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def archive_fact(self, fact_id: str) -> bool:
        async with self.session() as session:
            result = await session.execute(update(LearnedFact).where(LearnedFact.id == fact_id).values(is_active=False))
            rowcount = getattr(result, "rowcount", 0)
            return bool(rowcount and rowcount > 0)

    async def register_document(
        self,
        doc_id: str,
        filename: str,
        content_type: str,
        content_hash: str,
        chunk_count: int,
        total_chars: int,
        collection_name: str = "knowledge_base",
    ) -> DocumentMetadata:
        async with self.session() as session:
            doc = DocumentMetadata(
                id=doc_id,
                filename=filename,
                content_type=content_type,
                content_hash=content_hash,
                chunk_count=chunk_count,
                total_chars=total_chars,
                collection_name=collection_name,
            )
            session.add(doc)
            await session.flush()
            await session.refresh(doc)
            return doc

    async def get_document(self, doc_id: str) -> DocumentMetadata | None:
        async with self.session() as session:
            result = await session.execute(select(DocumentMetadata).where(DocumentMetadata.id == doc_id))
            return result.scalar_one_or_none()

    async def find_document_by_hash(self, content_hash: str) -> DocumentMetadata | None:
        async with self.session() as session:
            result = await session.execute(
                select(DocumentMetadata).where(
                    DocumentMetadata.content_hash == content_hash,
                    DocumentMetadata.is_active == True,  # noqa: E712
                )
            )
            return result.scalar_one_or_none()

    async def list_documents(
        self,
        collection_name: str | None = None,
        active_only: bool = True,
    ) -> list[DocumentMetadata]:
        async with self.session() as session:
            query = select(DocumentMetadata)
            if active_only:
                query = query.where(DocumentMetadata.is_active == True)  # noqa: E712
            if collection_name:
                query = query.where(DocumentMetadata.collection_name == collection_name)
            query = query.order_by(DocumentMetadata.created_at.desc())
            result = await session.execute(query)
            return list(result.scalars().all())

    async def deactivate_document(self, doc_id: str) -> bool:
        async with self.session() as session:
            result = await session.execute(
                update(DocumentMetadata).where(DocumentMetadata.id == doc_id).values(is_active=False)
            )
            rowcount = getattr(result, "rowcount", 0)
            return bool(rowcount and rowcount > 0)

    async def clear_tasks(self) -> int:
        async with self.session() as session:
            result = await session.execute(delete(TaskRecord))
            return int(getattr(result, "rowcount", 0) or 0)

    async def clear_context(self, category: str | None = None) -> int:
        async with self.session() as session:
            query = delete(UserContext)
            if category:
                query = query.where(UserContext.category == category)
            result = await session.execute(query)
            return int(getattr(result, "rowcount", 0) or 0)

    async def clear_facts(
        self,
        category: str | None = None,
        hard_delete: bool = False,
    ) -> int:
        async with self.session() as session:
            if hard_delete:
                del_query = delete(LearnedFact)
                if category:
                    del_query = del_query.where(LearnedFact.category == category)
                result = await session.execute(del_query)
                return int(getattr(result, "rowcount", 0) or 0)
            upd_query = update(LearnedFact).values(is_active=False)
            if category:
                upd_query = upd_query.where(LearnedFact.category == category)
            upd_result = await session.execute(upd_query)
            return int(getattr(upd_result, "rowcount", 0) or 0)

    async def clear_documents(
        self,
        collection_name: str | None = None,
        hard_delete: bool = False,
    ) -> int:
        async with self.session() as session:
            if hard_delete:
                del_query = delete(DocumentMetadata)
                if collection_name:
                    del_query = del_query.where(DocumentMetadata.collection_name == collection_name)
                result = await session.execute(del_query)
                return int(getattr(result, "rowcount", 0) or 0)
            upd_query = update(DocumentMetadata).values(is_active=False)
            if collection_name:
                upd_query = upd_query.where(DocumentMetadata.collection_name == collection_name)
            upd_result = await session.execute(upd_query)
            return int(getattr(upd_result, "rowcount", 0) or 0)

    async def clear_all(self, hard_delete: bool = False) -> dict[str, int]:
        return {
            "tasks": await self.clear_tasks(),
            "context": await self.clear_context(),
            "facts": await self.clear_facts(hard_delete=hard_delete),
            "documents": await self.clear_documents(hard_delete=hard_delete),
        }

    async def export_all(self, include_secrets: bool = False) -> dict[str, object]:
        async with self.session() as session:
            tasks_result = await session.execute(select(TaskRecord))
            tasks = list(tasks_result.scalars().all())

            context_result = await session.execute(select(UserContext))
            context = list(context_result.scalars().all())

            facts_result = await session.execute(select(LearnedFact))
            facts = list(facts_result.scalars().all())

            docs_result = await session.execute(select(DocumentMetadata))
            documents = list(docs_result.scalars().all())

            agents_result = await session.execute(select(AgentConfig).where(AgentConfig.is_builtin.is_(False)))
            agents = list(agents_result.scalars().all())

            pipelines_result = await session.execute(select(Pipeline))
            pipelines = list(pipelines_result.scalars().all())

            teams_result = await session.execute(select(Team))
            teams = list(teams_result.scalars().all())

            mcp_result = await session.execute(select(MCPServer))
            mcp_servers = list(mcp_result.scalars().all())

            settings_result = await session.execute(select(SystemSettings).where(SystemSettings.encrypted.is_(False)))
            settings_list = list(settings_result.scalars().all())

        return {
            "version": EXPORT_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "tasks": [
                {
                    "id": t.id,
                    "task": t.task,
                    "result": t.result,
                    "status": t.status,
                    "entry_agent": t.entry_agent,
                    "agents_used": t.agents_used,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                    "duration_seconds": t.duration_seconds,
                }
                for t in tasks
            ],
            "context": [
                {
                    "key": c.key,
                    "value": c.value,
                    "category": c.category,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in context
            ],
            "facts": [
                {
                    "id": f.id,
                    "fact": f.fact,
                    "category": f.category,
                    "confidence": f.confidence,
                    "source_agent": f.source_agent,
                    "is_active": f.is_active,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in facts
            ],
            "documents": [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "content_type": d.content_type,
                    "chunk_count": d.chunk_count,
                    "is_active": d.is_active,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in documents
            ],
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "description": a.description,
                    "system_prompt": a.system_prompt,
                    "role": a.role,
                    "capabilities": a.capabilities,
                    "enabled_tools": a.enabled_tools,
                    "mcp_server_ids": a.mcp_server_ids,
                    "model_override": a.model_override,
                    "is_active": a.is_active,
                }
                for a in agents
            ],
            "pipelines": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "nodes": p.nodes,
                    "edges": p.edges,
                    "settings": p.settings,
                    "is_active": p.is_active,
                    "is_default": p.is_default,
                }
                for p in pipelines
            ],
            "teams": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "agent_ids": t.agent_ids,
                    "lead_agent_id": t.lead_agent_id,
                    "pipeline_id": t.pipeline_id,
                    "mcp_server_ids": t.mcp_server_ids,
                    "settings": t.settings,
                    "is_active": t.is_active,
                }
                for t in teams
            ],
            "mcp_servers": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "transport": m.transport,
                    "command": m.command,
                    "args": m.args,
                    "url": m.url,
                    "headers": m.headers if include_secrets else {},
                    "env_vars": m.env_vars if include_secrets else {},
                    "is_active": m.is_active,
                }
                for m in mcp_servers
            ],
            "settings": [
                {
                    "key": s.key,
                    "value": s.value,
                }
                for s in settings_list
            ],
        }

    def _extract_list(
        self,
        data: dict[str, object],
        key: str,
    ) -> list[dict[str, object]]:
        raw = data.get(key)
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _migrate_export_data(self, data: dict[str, object], from_version: Version) -> dict[str, object]:
        migrated = dict(data)
        current = from_version

        if current < Version("1.1"):
            for agent in self._extract_list(migrated, "agents"):
                if "mcp_server_ids" not in agent:
                    agent["mcp_server_ids"] = []
            for team in self._extract_list(migrated, "teams"):
                if "mcp_server_ids" not in team:
                    team["mcp_server_ids"] = []
            current = Version("1.1")

        migrated["version"] = str(current)
        return migrated

    async def import_all(
        self,
        data: dict[str, object],
        merge: bool = True,
    ) -> dict[str, int]:
        raw_version = data.get("version", "1.0")
        version_str = str(raw_version) if raw_version else "1.0"

        try:
            import_version = Version(version_str)
        except Exception:
            import_version = Version("1.0")

        min_version = Version(MINIMUM_SUPPORTED_VERSION)
        max_version = Version(EXPORT_VERSION)

        if import_version < min_version:
            raise ValueError(f"Export version {version_str} is too old. Minimum supported: {MINIMUM_SUPPORTED_VERSION}")

        if import_version > max_version:
            logger.warning(
                "Import version %s is newer than current %s, some fields may be ignored",
                version_str,
                EXPORT_VERSION,
            )

        if import_version < max_version:
            data = self._migrate_export_data(data, import_version)
            logger.info("Migrated export data from version %s to %s", version_str, EXPORT_VERSION)

        counts: dict[str, int] = {
            "agents": 0,
            "pipelines": 0,
            "teams": 0,
            "mcp_servers": 0,
            "context": 0,
            "facts": 0,
            "settings": 0,
        }

        async with self.session() as session:
            for agent_data in self._extract_list(data, "agents"):
                agent_id = agent_data.get("id")
                if not isinstance(agent_id, str):
                    continue

                existing = await session.execute(select(AgentConfig).where(AgentConfig.id == agent_id))
                name = str(agent_data.get("name", ""))
                description = str(agent_data.get("description", ""))
                system_prompt = str(agent_data.get("system_prompt", ""))
                role = str(agent_data.get("role", "worker"))
                capabilities_raw = agent_data.get("capabilities", [])
                capabilities: list[str] = capabilities_raw if isinstance(capabilities_raw, list) else []
                tools_raw = agent_data.get("enabled_tools", [])
                enabled_tools: list[str] = tools_raw if isinstance(tools_raw, list) else []
                mcp_raw = agent_data.get("mcp_server_ids", [])
                mcp_server_ids: list[str] = mcp_raw if isinstance(mcp_raw, list) else []
                model_raw = agent_data.get("model_override")
                model_override = str(model_raw) if model_raw else None
                is_active = bool(agent_data.get("is_active", True))

                if existing.scalar_one_or_none() and merge:
                    await session.execute(
                        update(AgentConfig)
                        .where(AgentConfig.id == agent_id)
                        .values(
                            name=name,
                            description=description,
                            system_prompt=system_prompt,
                            role=role,
                            capabilities=capabilities,
                            enabled_tools=enabled_tools,
                            mcp_server_ids=mcp_server_ids,
                            model_override=model_override,
                            is_active=is_active,
                        )
                    )
                else:
                    session.add(
                        AgentConfig(
                            id=agent_id,
                            name=name,
                            description=description,
                            system_prompt=system_prompt,
                            role=role,
                            capabilities=capabilities,
                            enabled_tools=enabled_tools,
                            mcp_server_ids=mcp_server_ids,
                            model_override=model_override,
                            is_active=is_active,
                            is_builtin=False,
                        )
                    )
                counts["agents"] += 1

            for pipeline_data in self._extract_list(data, "pipelines"):
                pipeline_id = pipeline_data.get("id")
                if not isinstance(pipeline_id, str):
                    continue

                existing = await session.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
                name = str(pipeline_data.get("name", ""))
                description = str(pipeline_data.get("description", ""))
                nodes_raw = pipeline_data.get("nodes", [])
                nodes: list[dict[str, object]] = nodes_raw if isinstance(nodes_raw, list) else []
                edges_raw = pipeline_data.get("edges", [])
                edges: list[dict[str, object]] = edges_raw if isinstance(edges_raw, list) else []
                settings_raw = pipeline_data.get("settings", {})
                settings: dict[str, object] = settings_raw if isinstance(settings_raw, dict) else {}
                is_active = bool(pipeline_data.get("is_active", True))
                is_default = bool(pipeline_data.get("is_default", False))

                if existing.scalar_one_or_none() and merge:
                    await session.execute(
                        update(Pipeline)
                        .where(Pipeline.id == pipeline_id)
                        .values(
                            name=name,
                            description=description,
                            nodes=nodes,
                            edges=edges,
                            settings=settings,
                            is_active=is_active,
                            is_default=is_default,
                        )
                    )
                else:
                    session.add(
                        Pipeline(
                            id=pipeline_id,
                            name=name,
                            description=description,
                            nodes=nodes,
                            edges=edges,
                            settings=settings,
                            is_active=is_active,
                            is_default=is_default,
                        )
                    )
                counts["pipelines"] += 1

            for team_data in self._extract_list(data, "teams"):
                team_id = team_data.get("id")
                if not isinstance(team_id, str):
                    continue

                existing = await session.execute(select(Team).where(Team.id == team_id))
                team_name = str(team_data.get("name", ""))
                team_description = str(team_data.get("description", ""))
                agent_ids_raw = team_data.get("agent_ids", [])
                team_agent_ids: list[str] = agent_ids_raw if isinstance(agent_ids_raw, list) else []
                lead_raw = team_data.get("lead_agent_id")
                team_lead_agent_id = str(lead_raw) if lead_raw else None
                pipeline_raw = team_data.get("pipeline_id")
                team_pipeline_id = str(pipeline_raw) if pipeline_raw else None
                team_mcp_raw = team_data.get("mcp_server_ids", [])
                team_mcp_server_ids: list[str] = team_mcp_raw if isinstance(team_mcp_raw, list) else []
                team_settings_raw = team_data.get("settings", {})
                team_settings: dict[str, object] = team_settings_raw if isinstance(team_settings_raw, dict) else {}
                team_is_active = bool(team_data.get("is_active", True))

                if existing.scalar_one_or_none() and merge:
                    await session.execute(
                        update(Team)
                        .where(Team.id == team_id)
                        .values(
                            name=team_name,
                            description=team_description,
                            agent_ids=team_agent_ids,
                            lead_agent_id=team_lead_agent_id,
                            pipeline_id=team_pipeline_id,
                            mcp_server_ids=team_mcp_server_ids,
                            settings=team_settings,
                            is_active=team_is_active,
                        )
                    )
                else:
                    session.add(
                        Team(
                            id=team_id,
                            name=team_name,
                            description=team_description,
                            agent_ids=team_agent_ids,
                            lead_agent_id=team_lead_agent_id,
                            pipeline_id=team_pipeline_id,
                            mcp_server_ids=team_mcp_server_ids,
                            settings=team_settings,
                            is_active=team_is_active,
                        )
                    )
                counts["teams"] += 1

            for mcp_data in self._extract_list(data, "mcp_servers"):
                mcp_id = mcp_data.get("id")
                if not isinstance(mcp_id, str):
                    continue

                existing = await session.execute(select(MCPServer).where(MCPServer.id == mcp_id))
                name = str(mcp_data.get("name", ""))
                description = str(mcp_data.get("description", ""))
                transport = str(mcp_data.get("transport", "stdio"))
                command_raw = mcp_data.get("command")
                command = str(command_raw) if command_raw else None
                args_raw = mcp_data.get("args", [])
                args: list[str] = args_raw if isinstance(args_raw, list) else []
                url_raw = mcp_data.get("url")
                url = str(url_raw) if url_raw else None
                headers_raw = mcp_data.get("headers", {})
                headers: dict[str, str] = headers_raw if isinstance(headers_raw, dict) else {}
                env_raw = mcp_data.get("env_vars", {})
                env_vars: dict[str, str] = env_raw if isinstance(env_raw, dict) else {}
                is_active = bool(mcp_data.get("is_active", True))

                if existing.scalar_one_or_none() and merge:
                    await session.execute(
                        update(MCPServer)
                        .where(MCPServer.id == mcp_id)
                        .values(
                            name=name,
                            description=description,
                            transport=transport,
                            command=command,
                            args=args,
                            url=url,
                            headers=headers,
                            env_vars=env_vars,
                            is_active=is_active,
                        )
                    )
                else:
                    session.add(
                        MCPServer(
                            id=mcp_id,
                            name=name,
                            description=description,
                            transport=transport,
                            command=command,
                            args=args,
                            url=url,
                            headers=headers,
                            env_vars=env_vars,
                            is_active=is_active,
                        )
                    )
                counts["mcp_servers"] += 1

            for ctx_data in self._extract_list(data, "context"):
                key = ctx_data.get("key")
                if not isinstance(key, str):
                    continue

                existing = await session.execute(select(UserContext).where(UserContext.key == key))
                value = str(ctx_data.get("value", ""))
                category = str(ctx_data.get("category", "general"))

                if existing.scalar_one_or_none() and merge:
                    await session.execute(
                        update(UserContext).where(UserContext.key == key).values(value=value, category=category)
                    )
                else:
                    session.add(UserContext(key=key, value=value, category=category))
                counts["context"] += 1

            for fact_data in self._extract_list(data, "facts"):
                fact_id = fact_data.get("id")
                if not isinstance(fact_id, str):
                    continue

                existing = await session.execute(select(LearnedFact).where(LearnedFact.id == fact_id))
                if existing.scalar_one_or_none():
                    continue

                fact = str(fact_data.get("fact", ""))
                category = str(fact_data.get("category", "general"))
                confidence_raw = fact_data.get("confidence", 1.0)
                confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 1.0
                source_raw = fact_data.get("source_agent")
                source_agent = str(source_raw) if source_raw else None
                is_active = bool(fact_data.get("is_active", True))

                session.add(
                    LearnedFact(
                        id=fact_id,
                        fact=fact,
                        category=category,
                        confidence=confidence,
                        source_agent=source_agent,
                        is_active=is_active,
                    )
                )
                counts["facts"] += 1

            for setting_data in self._extract_list(data, "settings"):
                setting_key = setting_data.get("key")
                setting_value = setting_data.get("value")
                if not isinstance(setting_key, str) or not isinstance(setting_value, str):
                    continue

                existing = await session.execute(select(SystemSettings).where(SystemSettings.key == setting_key))
                if existing.scalar_one_or_none() and merge:
                    await session.execute(
                        update(SystemSettings).where(SystemSettings.key == setting_key).values(value=setting_value)
                    )
                else:
                    session.add(
                        SystemSettings(
                            key=setting_key,
                            value=setting_value,
                            encrypted=False,
                        )
                    )
                counts["settings"] += 1

        return counts

    async def get_setting(self, key: str, decrypt: bool = True) -> str | None:
        """Get a setting value, automatically decrypting if it was stored encrypted.

        Args:
            key: The setting key
            decrypt: Whether to decrypt encrypted values (default True)

        Returns:
            The setting value (decrypted if applicable) or None if not found
        """
        from src.crypto import decrypt as decrypt_value

        async with self.session() as session:
            result = await session.execute(select(SystemSettings).where(SystemSettings.key == key))
            setting = result.scalar_one_or_none()
            if setting is None:
                return None
            if setting.encrypted and decrypt:
                try:
                    return decrypt_value(setting.value)
                except ValueError:
                    logger.warning(f"Failed to decrypt setting '{key}', returning None")
                    return None
            return setting.value

    async def set_setting(
        self,
        key: str,
        value: str,
        encrypted: bool = False,
    ) -> SystemSettings:
        """Set a setting value, encrypting if requested.

        Args:
            key: The setting key
            value: The plaintext value to store
            encrypted: Whether to encrypt the value before storing

        Returns:
            The created/updated SystemSettings record
        """
        from src.crypto import encrypt as encrypt_value

        stored_value = encrypt_value(value) if encrypted else value

        async with self.session() as session:
            result = await session.execute(select(SystemSettings).where(SystemSettings.key == key))
            setting = result.scalar_one_or_none()
            if setting is not None:
                setting.value = stored_value
                setting.encrypted = encrypted
                setting.updated_at = datetime.now(UTC)
                await session.flush()
                await session.refresh(setting)
                return setting
            setting = SystemSettings(key=key, value=stored_value, encrypted=encrypted)
            session.add(setting)
            await session.flush()
            await session.refresh(setting)
            return setting

    async def get_all_settings(self, decrypt: bool = True) -> dict[str, str]:
        """Get all settings, automatically decrypting encrypted values.

        Args:
            decrypt: Whether to decrypt encrypted values (default True)

        Returns:
            Dictionary of setting key -> value (decrypted if applicable)
        """
        from src.crypto import decrypt as decrypt_value

        async with self.session() as session:
            result = await session.execute(select(SystemSettings))
            settings_dict: dict[str, str] = {}
            for s in result.scalars().all():
                if s.encrypted and decrypt:
                    try:
                        settings_dict[s.key] = decrypt_value(s.value)
                    except ValueError:
                        logger.warning(f"Failed to decrypt setting '{s.key}', skipping")
                        continue
                else:
                    settings_dict[s.key] = s.value
            return settings_dict

    async def delete_setting(self, key: str) -> bool:
        async with self.session() as session:
            result = await session.execute(delete(SystemSettings).where(SystemSettings.key == key))
            rowcount = getattr(result, "rowcount", 0)
            return bool(rowcount and rowcount > 0)

    async def create_agent_config(
        self,
        agent_id: str,
        name: str,
        description: str,
        system_prompt: str,
        role: str = "worker",
        capabilities: list[str] | None = None,
        enabled_tools: list[str] | None = None,
        model_override: str | None = None,
        is_builtin: bool = False,
        mcp_server_ids: list[str] | None = None,
    ) -> AgentConfig:
        async with self.session() as session:
            result = await session.execute(select(AgentConfig).where(AgentConfig.id == agent_id))
            agent = result.scalar_one_or_none()
            if agent is not None:
                agent.name = name
                agent.description = description
                agent.system_prompt = system_prompt
                agent.role = role
                agent.capabilities = capabilities or []
                agent.enabled_tools = enabled_tools or []
                agent.model_override = model_override
                agent.is_builtin = is_builtin
                agent.mcp_server_ids = mcp_server_ids or []
                agent.updated_at = datetime.now(UTC)
                await session.flush()
                await session.refresh(agent)
                return agent
            agent = AgentConfig(
                id=agent_id,
                name=name,
                description=description,
                system_prompt=system_prompt,
                role=role,
                capabilities=capabilities or [],
                enabled_tools=enabled_tools or [],
                model_override=model_override,
                is_builtin=is_builtin,
                mcp_server_ids=mcp_server_ids or [],
            )
            session.add(agent)
            await session.flush()
            await session.refresh(agent)
            return agent

    async def get_agent_config(self, agent_id: str) -> AgentConfig | None:
        async with self.session() as session:
            result = await session.execute(select(AgentConfig).where(AgentConfig.id == agent_id))
            return result.scalar_one_or_none()

    async def list_agent_configs(self, active_only: bool = True) -> list[AgentConfig]:
        async with self.session() as session:
            query = select(AgentConfig)
            if active_only:
                query = query.where(AgentConfig.is_active == True)  # noqa: E712
            query = query.order_by(AgentConfig.name)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def update_agent_prompt(
        self,
        agent_id: str,
        system_prompt: str,
    ) -> AgentConfig | None:
        async with self.session() as session:
            result = await session.execute(select(AgentConfig).where(AgentConfig.id == agent_id))
            agent = result.scalar_one_or_none()
            if agent is not None:
                agent.system_prompt = system_prompt
                agent.updated_at = datetime.now(UTC)
                await session.flush()
                await session.refresh(agent)
            return agent

    async def toggle_agent(
        self,
        agent_id: str,
        is_active: bool,
    ) -> AgentConfig | None:
        async with self.session() as session:
            result = await session.execute(select(AgentConfig).where(AgentConfig.id == agent_id))
            agent = result.scalar_one_or_none()
            if agent is not None:
                agent.is_active = is_active
                agent.updated_at = datetime.now(UTC)
                await session.flush()
                await session.refresh(agent)
            return agent

    async def delete_agent_config(self, agent_id: str) -> bool:
        async with self.session() as session:
            result = await session.execute(select(AgentConfig).where(AgentConfig.id == agent_id))
            agent = result.scalar_one_or_none()
            if agent and not agent.is_builtin:
                await session.delete(agent)
                return True
            return False

    async def create_pipeline(
        self,
        name: str,
        description: str = "",
        nodes: list[dict[str, object]] | None = None,
        edges: list[dict[str, object]] | None = None,
        settings: dict[str, object] | None = None,
        is_default: bool = False,
    ) -> Pipeline:
        from uuid import uuid4

        async with self.session() as session:
            if is_default:
                await session.execute(
                    update(Pipeline).where(Pipeline.is_default == True).values(is_default=False)  # noqa: E712
                )

            pipeline = Pipeline(
                id=str(uuid4()),
                name=name,
                description=description,
                nodes=nodes or [],
                edges=edges or [],
                settings=settings or {},
                is_default=is_default,
            )
            session.add(pipeline)
            await session.flush()
            await session.refresh(pipeline)
            return pipeline

    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None:
        async with self.session() as session:
            result = await session.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
            return result.scalar_one_or_none()

    async def list_pipelines(self, active_only: bool = True) -> list[Pipeline]:
        async with self.session() as session:
            query = select(Pipeline)
            if active_only:
                query = query.where(Pipeline.is_active == True)  # noqa: E712
            query = query.order_by(Pipeline.name)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def update_pipeline(
        self,
        pipeline_id: str,
        name: str | None = None,
        description: str | None = None,
        nodes: list[dict[str, object]] | None = None,
        edges: list[dict[str, object]] | None = None,
        settings: dict[str, object] | None = None,
        is_active: bool | None = None,
        is_default: bool | None = None,
    ) -> Pipeline | None:
        async with self.session() as session:
            result = await session.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
            pipeline = result.scalar_one_or_none()
            if pipeline is None:
                return None

            if is_default is True:
                await session.execute(
                    update(Pipeline)
                    .where(Pipeline.is_default == True, Pipeline.id != pipeline_id)  # noqa: E712
                    .values(is_default=False)
                )

            if name is not None:
                pipeline.name = name
            if description is not None:
                pipeline.description = description
            if nodes is not None:
                pipeline.nodes = nodes
            if edges is not None:
                pipeline.edges = edges
            if settings is not None:
                pipeline.settings = settings
            if is_active is not None:
                pipeline.is_active = is_active
            if is_default is not None:
                pipeline.is_default = is_default
            pipeline.updated_at = datetime.now(UTC)
            await session.flush()
            await session.refresh(pipeline)
            return pipeline

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        async with self.session() as session:
            result = await session.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
            pipeline = result.scalar_one_or_none()
            if pipeline is not None:
                await session.delete(pipeline)
                return True
            return False

    async def get_default_pipeline(self) -> Pipeline | None:
        async with self.session() as session:
            result = await session.execute(
                select(Pipeline).where(Pipeline.is_default == True, Pipeline.is_active == True)  # noqa: E712
            )
            return result.scalar_one_or_none()

    async def create_team(
        self,
        name: str,
        description: str = "",
        agent_ids: list[str] | None = None,
        lead_agent_id: str | None = None,
        settings: dict[str, object] | None = None,
    ) -> Team:
        from uuid import uuid4

        async with self.session() as session:
            team = Team(
                id=str(uuid4()),
                name=name,
                description=description,
                agent_ids=agent_ids or [],
                lead_agent_id=lead_agent_id,
                settings=settings or {},
            )
            session.add(team)
            await session.flush()
            await session.refresh(team)
            return team

    async def get_team(self, team_id: str) -> Team | None:
        async with self.session() as session:
            result = await session.execute(select(Team).where(Team.id == team_id))
            return result.scalar_one_or_none()

    async def list_teams(self, active_only: bool = True) -> list[Team]:
        async with self.session() as session:
            query = select(Team)
            if active_only:
                query = query.where(Team.is_active == True)  # noqa: E712
            query = query.order_by(Team.name)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def update_team(
        self,
        team_id: str,
        name: str | None = None,
        description: str | None = None,
        agent_ids: list[str] | None = None,
        lead_agent_id: str | None = None,
        settings: dict[str, object] | None = None,
        is_active: bool | None = None,
    ) -> Team | None:
        async with self.session() as session:
            result = await session.execute(select(Team).where(Team.id == team_id))
            team = result.scalar_one_or_none()
            if team is None:
                return None
            if name is not None:
                team.name = name
            if description is not None:
                team.description = description
            if agent_ids is not None:
                team.agent_ids = agent_ids
            if lead_agent_id is not None:
                team.lead_agent_id = lead_agent_id
            if settings is not None:
                team.settings = settings
            if is_active is not None:
                team.is_active = is_active
            team.updated_at = datetime.now(UTC)
            await session.flush()
            await session.refresh(team)
            return team

    async def delete_team(self, team_id: str) -> bool:
        async with self.session() as session:
            result = await session.execute(select(Team).where(Team.id == team_id))
            team = result.scalar_one_or_none()
            if team is not None:
                await session.delete(team)
                return True
            return False

    async def create_mcp_server(
        self,
        name: str,
        transport: str,
        description: str = "",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> MCPServer:
        from uuid import uuid4

        async with self.session() as session:
            result = await session.execute(select(MCPServer).where(MCPServer.name == name))
            existing = result.scalar_one_or_none()
            if existing is not None:
                existing.transport = transport
                existing.description = description
                existing.command = command
                existing.args = args or []
                existing.url = url
                existing.headers = headers or {}
                existing.env_vars = env_vars or {}
                existing.updated_at = datetime.now(UTC)
                await session.flush()
                await session.refresh(existing)
                return existing
            server = MCPServer(
                id=str(uuid4()),
                name=name,
                transport=transport,
                description=description,
                command=command,
                args=args or [],
                url=url,
                headers=headers or {},
                env_vars=env_vars or {},
            )
            session.add(server)
            await session.flush()
            await session.refresh(server)
            return server

    async def get_mcp_server(self, server_id: str) -> MCPServer | None:
        async with self.session() as session:
            result = await session.execute(select(MCPServer).where(MCPServer.id == server_id))
            return result.scalar_one_or_none()

    async def get_mcp_server_by_name(self, name: str) -> MCPServer | None:
        async with self.session() as session:
            result = await session.execute(select(MCPServer).where(MCPServer.name == name))
            return result.scalar_one_or_none()

    async def list_mcp_servers(self, active_only: bool = True) -> list[MCPServer]:
        async with self.session() as session:
            query = select(MCPServer)
            if active_only:
                query = query.where(MCPServer.is_active == True)  # noqa: E712
            query = query.order_by(MCPServer.name)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def update_mcp_server(
        self,
        server_id: str,
        name: str | None = None,
        transport: str | None = None,
        description: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        is_active: bool | None = None,
    ) -> MCPServer | None:
        async with self.session() as session:
            result = await session.execute(select(MCPServer).where(MCPServer.id == server_id))
            server = result.scalar_one_or_none()
            if server is None:
                return None
            if name is not None:
                server.name = name
            if transport is not None:
                server.transport = transport
            if description is not None:
                server.description = description
            if command is not None:
                server.command = command
            if args is not None:
                server.args = args
            if url is not None:
                server.url = url
            if headers is not None:
                server.headers = headers
            if env_vars is not None:
                server.env_vars = env_vars
            if is_active is not None:
                server.is_active = is_active
            server.updated_at = datetime.now(UTC)
            await session.flush()
            await session.refresh(server)
            return server

    async def delete_mcp_server(self, server_id: str) -> bool:
        async with self.session() as session:
            result = await session.execute(select(MCPServer).where(MCPServer.id == server_id))
            server = result.scalar_one_or_none()
            if server is not None:
                await session.delete(server)
                return True
            return False

    async def get_mcp_servers_by_ids(self, server_ids: list[str]) -> list[MCPServer]:
        if not server_ids:
            return []
        async with self.session() as session:
            result = await session.execute(
                select(MCPServer).where(
                    MCPServer.id.in_(server_ids),
                    MCPServer.is_active == True,  # noqa: E712
                )
            )
            return list(result.scalars().all())

    async def close(self) -> None:
        await self._engine.dispose()


def _get_database_url() -> str:
    postgres_uri = os.environ.get("POSTGRES_URI")
    if postgres_uri:
        if postgres_uri.startswith("postgresql://"):
            return postgres_uri.replace("postgresql://", "postgresql+asyncpg://", 1)
        if postgres_uri.startswith("postgres://"):
            return postgres_uri.replace("postgres://", "postgresql+asyncpg://", 1)
        return postgres_uri

    data_dir = os.environ.get("DATA_DIR", "./data")
    os.makedirs(data_dir, exist_ok=True)
    return f"sqlite+aiosqlite:///{data_dir}/storage.db"


async def create_repository(
    database_url: str | None = None,
    pool_size: int = _DEFAULT_POOL_SIZE,
    max_overflow: int = _DEFAULT_MAX_OVERFLOW,
    pool_timeout: float = _DEFAULT_POOL_TIMEOUT,
) -> StorageRepository:
    url = database_url or _get_database_url()

    is_sqlite = url.startswith("sqlite")
    engine_kwargs: dict[str, object] = {"echo": False}

    if not is_sqlite:
        engine_kwargs.update(
            {
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": pool_timeout,
            }
        )

    engine = create_async_engine(url, **engine_kwargs)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    return StorageRepository(engine, session_factory)


_repository: StorageRepository | None = None
_repository_lock: asyncio.Lock | None = None


def _get_repository_lock() -> asyncio.Lock:
    global _repository_lock
    if _repository_lock is None:
        _repository_lock = asyncio.Lock()
    return _repository_lock


async def get_repository() -> StorageRepository:
    global _repository
    if _repository is not None:
        return _repository
    async with _get_repository_lock():
        if _repository is None:
            _repository = await create_repository()
        return _repository


async def close_repository() -> None:
    global _repository
    async with _get_repository_lock():
        if _repository is not None:
            await _repository.close()
            _repository = None
