from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from src.storage.repository import StorageRepository, create_repository


@pytest_asyncio.fixture
async def repo() -> AsyncGenerator[StorageRepository]:
    repository = await create_repository("sqlite+aiosqlite:///:memory:")
    yield repository
    await repository.close()


class TestTaskOperations:
    @pytest.mark.asyncio
    async def test_create_task(self, repo: StorageRepository) -> None:
        task = await repo.create_task("Test task", entry_agent="coder")
        assert task.task == "Test task"
        assert task.entry_agent == "coder"
        assert task.status == "pending"
        assert task.id is not None

    @pytest.mark.asyncio
    async def test_start_task(self, repo: StorageRepository) -> None:
        task = await repo.create_task("Test task")
        started = await repo.start_task(task.id)
        assert started is not None
        assert started.status == "running"

    @pytest.mark.asyncio
    async def test_complete_task(self, repo: StorageRepository) -> None:
        task = await repo.create_task("Test task")
        await repo.start_task(task.id)
        completed = await repo.complete_task(
            task.id,
            result="Done",
            agents_used=["coder", "writer"],
            handoff_count=2,
            iteration_count=5,
        )
        assert completed is not None
        assert completed.status == "completed"
        assert completed.result == "Done"
        assert completed.agents_used == ["coder", "writer"]
        assert completed.handoff_count == 2
        assert completed.iteration_count == 5

    @pytest.mark.asyncio
    async def test_fail_task(self, repo: StorageRepository) -> None:
        task = await repo.create_task("Test task")
        failed = await repo.fail_task(task.id, "Something went wrong")
        assert failed is not None
        assert failed.status == "failed"
        assert "Something went wrong" in failed.result

    @pytest.mark.asyncio
    async def test_get_task(self, repo: StorageRepository) -> None:
        task = await repo.create_task("Test task")
        retrieved = await repo.get_task(task.id)
        assert retrieved is not None
        assert retrieved.task == "Test task"

    @pytest.mark.asyncio
    async def test_get_task_nonexistent(self, repo: StorageRepository) -> None:
        retrieved = await repo.get_task("nonexistent-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_recent_tasks(self, repo: StorageRepository) -> None:
        await repo.create_task("Task 1")
        await repo.create_task("Task 2")
        await repo.create_task("Task 3")
        tasks = await repo.get_recent_tasks(limit=2)
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_clear_tasks(self, repo: StorageRepository) -> None:
        await repo.create_task("Task 1")
        await repo.create_task("Task 2")
        count = await repo.clear_tasks()
        assert count == 2
        tasks = await repo.get_recent_tasks()
        assert len(tasks) == 0


class TestContextOperations:
    @pytest.mark.asyncio
    async def test_set_and_get_context(self, repo: StorageRepository) -> None:
        await repo.set_context("user_name", "Alice", category="profile")
        value = await repo.get_context("user_name")
        assert value == "Alice"

    @pytest.mark.asyncio
    async def test_update_context(self, repo: StorageRepository) -> None:
        await repo.set_context("key", "value1")
        await repo.set_context("key", "value2")
        value = await repo.get_context("key")
        assert value == "value2"

    @pytest.mark.asyncio
    async def test_get_context_nonexistent(self, repo: StorageRepository) -> None:
        value = await repo.get_context("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_get_all_context(self, repo: StorageRepository) -> None:
        await repo.set_context("key1", "value1", category="cat1")
        await repo.set_context("key2", "value2", category="cat1")
        await repo.set_context("key3", "value3", category="cat2")
        all_ctx = await repo.get_all_context()
        assert len(all_ctx) == 3
        cat1_ctx = await repo.get_all_context(category="cat1")
        assert len(cat1_ctx) == 2

    @pytest.mark.asyncio
    async def test_delete_context(self, repo: StorageRepository) -> None:
        await repo.set_context("key", "value")
        deleted = await repo.delete_context("key")
        assert deleted is True
        value = await repo.get_context("key")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_context_nonexistent(self, repo: StorageRepository) -> None:
        deleted = await repo.delete_context("nonexistent")
        assert deleted is False


class TestFactOperations:
    @pytest.mark.asyncio
    async def test_store_fact(self, repo: StorageRepository) -> None:
        fact = await repo.store_fact(
            "Python is a programming language",
            category="tech",
            source_agent="researcher",
            confidence=0.95,
        )
        assert fact.fact == "Python is a programming language"
        assert fact.category == "tech"
        assert fact.confidence == 0.95

    @pytest.mark.asyncio
    async def test_get_facts(self, repo: StorageRepository) -> None:
        await repo.store_fact("Fact 1", category="cat1")
        await repo.store_fact("Fact 2", category="cat1")
        await repo.store_fact("Fact 3", category="cat2")
        all_facts = await repo.get_facts()
        assert len(all_facts) == 3
        cat1_facts = await repo.get_facts(category="cat1")
        assert len(cat1_facts) == 2

    @pytest.mark.asyncio
    async def test_archive_fact(self, repo: StorageRepository) -> None:
        fact = await repo.store_fact("Test fact")
        archived = await repo.archive_fact(fact.id)
        assert archived is True
        active_facts = await repo.get_facts(active_only=True)
        assert len(active_facts) == 0

    @pytest.mark.asyncio
    async def test_clear_facts_soft(self, repo: StorageRepository) -> None:
        await repo.store_fact("Fact 1")
        await repo.store_fact("Fact 2")
        count = await repo.clear_facts(hard_delete=False)
        assert count == 2
        all_facts = await repo.get_facts(active_only=False)
        assert len(all_facts) == 2
        active_facts = await repo.get_facts(active_only=True)
        assert len(active_facts) == 0

    @pytest.mark.asyncio
    async def test_clear_facts_hard(self, repo: StorageRepository) -> None:
        await repo.store_fact("Fact 1")
        await repo.store_fact("Fact 2")
        count = await repo.clear_facts(hard_delete=True)
        assert count == 2
        all_facts = await repo.get_facts(active_only=False)
        assert len(all_facts) == 0


class TestDocumentOperations:
    @pytest.mark.asyncio
    async def test_register_document(self, repo: StorageRepository) -> None:
        doc = await repo.register_document(
            doc_id="doc123",
            filename="test.pdf",
            content_type="application/pdf",
            content_hash="abc123",
            chunk_count=5,
            total_chars=1000,
        )
        assert doc.id == "doc123"
        assert doc.filename == "test.pdf"
        assert doc.chunk_count == 5

    @pytest.mark.asyncio
    async def test_get_document(self, repo: StorageRepository) -> None:
        await repo.register_document(
            doc_id="doc123",
            filename="test.pdf",
            content_type="application/pdf",
            content_hash="abc123",
            chunk_count=5,
            total_chars=1000,
        )
        doc = await repo.get_document("doc123")
        assert doc is not None
        assert doc.filename == "test.pdf"

    @pytest.mark.asyncio
    async def test_find_document_by_hash(self, repo: StorageRepository) -> None:
        await repo.register_document(
            doc_id="doc123",
            filename="test.pdf",
            content_type="application/pdf",
            content_hash="unique_hash",
            chunk_count=5,
            total_chars=1000,
        )
        doc = await repo.find_document_by_hash("unique_hash")
        assert doc is not None
        assert doc.id == "doc123"

    @pytest.mark.asyncio
    async def test_list_documents(self, repo: StorageRepository) -> None:
        await repo.register_document(
            doc_id="doc1",
            filename="test1.pdf",
            content_type="application/pdf",
            content_hash="hash1",
            chunk_count=5,
            total_chars=1000,
            collection_name="collection1",
        )
        await repo.register_document(
            doc_id="doc2",
            filename="test2.pdf",
            content_type="application/pdf",
            content_hash="hash2",
            chunk_count=3,
            total_chars=500,
            collection_name="collection2",
        )
        all_docs = await repo.list_documents()
        assert len(all_docs) == 2
        col1_docs = await repo.list_documents(collection_name="collection1")
        assert len(col1_docs) == 1

    @pytest.mark.asyncio
    async def test_deactivate_document(self, repo: StorageRepository) -> None:
        await repo.register_document(
            doc_id="doc123",
            filename="test.pdf",
            content_type="application/pdf",
            content_hash="abc123",
            chunk_count=5,
            total_chars=1000,
        )
        deactivated = await repo.deactivate_document("doc123")
        assert deactivated is True
        active_docs = await repo.list_documents(active_only=True)
        assert len(active_docs) == 0


class TestSettingsOperations:
    @pytest.mark.asyncio
    async def test_set_and_get_setting(self, repo: StorageRepository) -> None:
        await repo.set_setting("model", "gpt-4")
        value = await repo.get_setting("model")
        assert value == "gpt-4"

    @pytest.mark.asyncio
    async def test_update_setting(self, repo: StorageRepository) -> None:
        await repo.set_setting("key", "value1")
        await repo.set_setting("key", "value2")
        value = await repo.get_setting("key")
        assert value == "value2"

    @pytest.mark.asyncio
    async def test_get_all_settings(self, repo: StorageRepository) -> None:
        await repo.set_setting("key1", "value1")
        await repo.set_setting("key2", "value2")
        settings = await repo.get_all_settings()
        assert len(settings) == 2
        assert settings["key1"] == "value1"
        assert settings["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_delete_setting(self, repo: StorageRepository) -> None:
        await repo.set_setting("key", "value")
        deleted = await repo.delete_setting("key")
        assert deleted is True
        value = await repo.get_setting("key")
        assert value is None


class TestAgentConfigOperations:
    @pytest.mark.asyncio
    async def test_create_agent_config(self, repo: StorageRepository) -> None:
        agent = await repo.create_agent_config(
            agent_id="coder",
            name="Coder Agent",
            description="Writes code",
            system_prompt="You are a coding assistant",
            role="worker",
            capabilities=["python", "javascript"],
            enabled_tools=["execute_code"],
        )
        assert agent.id == "coder"
        assert agent.name == "Coder Agent"
        assert agent.capabilities == ["python", "javascript"]

    @pytest.mark.asyncio
    async def test_get_agent_config(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="coder",
            name="Coder Agent",
            description="Writes code",
            system_prompt="You are a coding assistant",
        )
        agent = await repo.get_agent_config("coder")
        assert agent is not None
        assert agent.name == "Coder Agent"

    @pytest.mark.asyncio
    async def test_list_agent_configs(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="agent1",
            name="Agent 1",
            description="First agent",
            system_prompt="Prompt 1",
        )
        await repo.create_agent_config(
            agent_id="agent2",
            name="Agent 2",
            description="Second agent",
            system_prompt="Prompt 2",
        )
        agents = await repo.list_agent_configs()
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_update_agent_prompt(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="coder",
            name="Coder Agent",
            description="Writes code",
            system_prompt="Original prompt",
        )
        updated = await repo.update_agent_prompt("coder", "New prompt")
        assert updated is not None
        assert updated.system_prompt == "New prompt"

    @pytest.mark.asyncio
    async def test_toggle_agent(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="coder",
            name="Coder Agent",
            description="Writes code",
            system_prompt="Prompt",
        )
        toggled = await repo.toggle_agent("coder", is_active=False)
        assert toggled is not None
        assert toggled.is_active is False
        active_agents = await repo.list_agent_configs(active_only=True)
        assert len(active_agents) == 0

    @pytest.mark.asyncio
    async def test_delete_agent_config(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="custom_agent",
            name="Custom Agent",
            description="A custom agent",
            system_prompt="Prompt",
            is_builtin=False,
        )
        deleted = await repo.delete_agent_config("custom_agent")
        assert deleted is True
        agent = await repo.get_agent_config("custom_agent")
        assert agent is None

    @pytest.mark.asyncio
    async def test_delete_builtin_agent_fails(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="builtin_agent",
            name="Builtin Agent",
            description="A builtin agent",
            system_prompt="Prompt",
            is_builtin=True,
        )
        deleted = await repo.delete_agent_config("builtin_agent")
        assert deleted is False
        agent = await repo.get_agent_config("builtin_agent")
        assert agent is not None


class TestExportAndClearAll:
    @pytest.mark.asyncio
    async def test_export_all(self, repo: StorageRepository) -> None:
        await repo.create_task("Test task")
        await repo.set_context("key", "value")
        await repo.store_fact("Test fact")
        await repo.register_document(
            doc_id="doc1",
            filename="test.pdf",
            content_type="application/pdf",
            content_hash="hash1",
            chunk_count=1,
            total_chars=100,
        )
        await repo.create_agent_config(
            agent_id="custom_agent",
            name="Custom Agent",
            description="Test agent",
            system_prompt="You are a test agent",
            is_builtin=False,
        )
        await repo.set_setting("test_key", "test_value")

        export = await repo.export_all()
        assert "exported_at" in export
        assert "version" in export
        assert len(export["tasks"]) == 1
        assert len(export["context"]) == 1
        assert len(export["facts"]) == 1
        assert len(export["documents"]) == 1
        assert len(export["agents"]) == 1
        assert len(export["settings"]) == 1
        assert "pipelines" in export
        assert "teams" in export
        assert "mcp_servers" in export

    @pytest.mark.asyncio
    async def test_export_all_excludes_builtin_agents(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="builtin_agent",
            name="Builtin Agent",
            description="Builtin agent",
            system_prompt="Prompt",
            is_builtin=True,
        )
        await repo.create_agent_config(
            agent_id="custom_agent",
            name="Custom Agent",
            description="Custom agent",
            system_prompt="Prompt",
            is_builtin=False,
        )
        export = await repo.export_all()
        assert len(export["agents"]) == 1
        agents = export["agents"]
        assert isinstance(agents, list)
        assert agents[0]["id"] == "custom_agent"

    @pytest.mark.asyncio
    async def test_clear_all(self, repo: StorageRepository) -> None:
        await repo.create_task("Test task")
        await repo.set_context("key", "value")
        await repo.store_fact("Test fact")
        result = await repo.clear_all(hard_delete=True)
        assert result["tasks"] == 1
        assert result["context"] == 1
        assert result["facts"] == 1


class TestImportAll:
    @pytest.mark.asyncio
    async def test_import_agents(self, repo: StorageRepository) -> None:
        import_data: dict[str, object] = {
            "agents": [
                {
                    "id": "imported_agent",
                    "name": "Imported Agent",
                    "description": "An imported agent",
                    "system_prompt": "You are imported",
                    "role": "worker",
                    "capabilities": ["python"],
                    "enabled_tools": ["execute_python"],
                    "is_active": True,
                }
            ]
        }
        counts = await repo.import_all(import_data, merge=True)
        assert counts["agents"] == 1

        agent = await repo.get_agent_config("imported_agent")
        assert agent is not None
        assert agent.name == "Imported Agent"
        assert agent.capabilities == ["python"]

    @pytest.mark.asyncio
    async def test_import_agents_merge_update(self, repo: StorageRepository) -> None:
        await repo.create_agent_config(
            agent_id="existing_agent",
            name="Original Name",
            description="Original",
            system_prompt="Original prompt",
        )

        import_data: dict[str, object] = {
            "agents": [
                {
                    "id": "existing_agent",
                    "name": "Updated Name",
                    "description": "Updated",
                    "system_prompt": "Updated prompt",
                }
            ]
        }
        counts = await repo.import_all(import_data, merge=True)
        assert counts["agents"] == 1

        agent = await repo.get_agent_config("existing_agent")
        assert agent is not None
        assert agent.name == "Updated Name"
        assert agent.description == "Updated"

    @pytest.mark.asyncio
    async def test_import_context(self, repo: StorageRepository) -> None:
        import_data: dict[str, object] = {
            "context": [{"key": "imported_key", "value": "imported_value", "category": "test"}]
        }
        counts = await repo.import_all(import_data, merge=True)
        assert counts["context"] == 1

        value = await repo.get_context("imported_key")
        assert value == "imported_value"

    @pytest.mark.asyncio
    async def test_import_facts(self, repo: StorageRepository) -> None:
        import_data: dict[str, object] = {
            "facts": [
                {
                    "id": "imported_fact_1",
                    "fact": "Imported fact",
                    "category": "imported",
                    "confidence": 0.9,
                    "source_agent": "importer",
                }
            ]
        }
        counts = await repo.import_all(import_data, merge=True)
        assert counts["facts"] == 1

        facts = await repo.get_facts(category="imported")
        assert len(facts) == 1
        assert facts[0].fact == "Imported fact"

    @pytest.mark.asyncio
    async def test_import_settings(self, repo: StorageRepository) -> None:
        import_data: dict[str, object] = {"settings": [{"key": "imported_setting", "value": "imported_value"}]}
        counts = await repo.import_all(import_data, merge=True)
        assert counts["settings"] == 1

        value = await repo.get_setting("imported_setting")
        assert value == "imported_value"

    @pytest.mark.asyncio
    async def test_import_empty_data(self, repo: StorageRepository) -> None:
        counts = await repo.import_all({}, merge=True)
        assert counts["agents"] == 0
        assert counts["context"] == 0
        assert counts["facts"] == 0
        assert counts["settings"] == 0

    @pytest.mark.asyncio
    async def test_import_invalid_agent_id_skipped(self, repo: StorageRepository) -> None:
        import_data: dict[str, object] = {
            "agents": [
                {"name": "No ID Agent", "description": "Missing ID"},
                {"id": 123, "name": "Invalid ID type"},
            ]
        }
        counts = await repo.import_all(import_data, merge=True)
        assert counts["agents"] == 0
