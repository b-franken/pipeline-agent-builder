from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.types import AgentState


class TestWorkflowCreation:
    def test_create_workflow_with_persistence(self) -> None:
        with (
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.SupervisorAgent") as mock_supervisor,
            patch("src.graph.workflow.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get_agent_names.return_value = ["coder", "researcher"]
            mock_agent = MagicMock()
            mock_agent.invoke = MagicMock()
            mock_registry.return_value.create_agent.return_value = mock_agent
            mock_supervisor.return_value.route = MagicMock()
            mock_checkpointer.return_value = MagicMock()

            from src.graph.workflow import clear_workflow_cache, create_workflow

            clear_workflow_cache()
            workflow = create_workflow(use_persistence=True)

            assert workflow is not None
            mock_checkpointer.assert_called()

    def test_create_workflow_without_persistence(self) -> None:
        with (
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.SupervisorAgent") as mock_supervisor,
            patch("src.graph.workflow.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get_agent_names.return_value = []
            mock_supervisor.return_value.route = MagicMock()
            mock_checkpointer.return_value = MagicMock()

            from src.graph.workflow import clear_workflow_cache, create_workflow

            clear_workflow_cache()
            workflow = create_workflow(use_persistence=False)

            assert workflow is not None
            mock_checkpointer.assert_called_with(backend="memory")

    def test_create_workflow_caching(self) -> None:
        with (
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.SupervisorAgent") as mock_supervisor,
            patch("src.graph.workflow.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get_agent_names.return_value = []
            mock_supervisor.return_value.route = MagicMock()
            mock_checkpointer.return_value = MagicMock()

            from src.graph.workflow import clear_workflow_cache, create_workflow

            clear_workflow_cache()

            workflow1 = create_workflow(use_persistence=False, supervisor_model="gpt-4")
            workflow2 = create_workflow(use_persistence=False, supervisor_model="gpt-4")

            assert workflow1 is workflow2

    def test_create_workflow_force_new(self) -> None:
        with (
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.SupervisorAgent") as mock_supervisor,
            patch("src.graph.workflow.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get_agent_names.return_value = []
            mock_supervisor.return_value.route = MagicMock()
            mock_checkpointer.return_value = MagicMock()

            from src.graph.workflow import clear_workflow_cache, create_workflow

            clear_workflow_cache()

            workflow1 = create_workflow(use_persistence=False)
            workflow2 = create_workflow(use_persistence=False, force_new=True)

            assert workflow1 is not workflow2

    def test_create_workflow_with_agent_configs(self) -> None:
        with (
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.SupervisorAgent") as mock_supervisor,
            patch("src.graph.workflow.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get_agent_names.return_value = ["coder"]
            mock_agent = MagicMock()
            mock_agent.invoke = MagicMock()
            mock_registry.return_value.create_agent.return_value = mock_agent
            mock_supervisor.return_value.route = MagicMock()
            mock_checkpointer.return_value = MagicMock()

            from src.graph.workflow import clear_workflow_cache, create_workflow

            clear_workflow_cache()
            agent_configs = {"coder": {"model": "gpt-4o"}}
            workflow = create_workflow(use_persistence=False, agent_configs=agent_configs)

            assert workflow is not None
            mock_registry.return_value.create_agent.assert_called_with("coder", model="gpt-4o")

    def test_create_simple_workflow(self) -> None:
        with (
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.SupervisorAgent") as mock_supervisor,
            patch("src.graph.workflow.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get_agent_names.return_value = []
            mock_supervisor.return_value.route = MagicMock()
            mock_checkpointer.return_value = MagicMock()

            from src.graph.workflow import clear_workflow_cache, create_simple_workflow

            clear_workflow_cache()
            workflow = create_simple_workflow()

            assert workflow is not None


class TestWorkflowCacheManagement:
    def test_clear_workflow_cache(self) -> None:
        from src.graph.workflow import (
            _workflow_cache,
            clear_workflow_cache,
            get_cached_workflow_count,
        )

        _workflow_cache["test"] = MagicMock()

        clear_workflow_cache()

        assert get_cached_workflow_count() == 0

    def test_invalidate_workflow(self) -> None:
        import src.graph.workflow as workflow_module

        workflow_module._default_workflow = MagicMock()
        workflow_module._workflow_cache = {"key": MagicMock()}

        workflow_module.invalidate_workflow()

        assert workflow_module._default_workflow is None
        assert len(workflow_module._workflow_cache) == 0

    def test_get_workflow_creates_if_none(self) -> None:
        with (
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.SupervisorAgent") as mock_supervisor,
            patch("src.graph.workflow.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get_agent_names.return_value = []
            mock_supervisor.return_value.route = MagicMock()
            mock_checkpointer.return_value = MagicMock()

            import src.graph.workflow as workflow_module

            workflow_module._default_workflow = None
            workflow_module._workflow_cache = {}

            workflow = workflow_module.get_workflow()

            assert workflow is not None
            assert workflow_module._default_workflow is workflow

    def test_get_workflow_returns_cached(self) -> None:
        import src.graph.workflow as workflow_module

        cached_workflow = MagicMock()
        workflow_module._default_workflow = cached_workflow

        workflow = workflow_module.get_workflow()

        assert workflow is cached_workflow

    def test_set_workflow(self) -> None:
        import src.graph.workflow as workflow_module

        new_workflow = MagicMock()

        workflow_module.set_workflow(new_workflow)

        assert workflow_module._default_workflow is new_workflow


class TestCacheKeyGeneration:
    def test_cache_key_consistency(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, "gpt-4", None, None)
        key2 = _make_cache_key(True, "gpt-4", None, None)

        assert key1 == key2

    def test_cache_key_different_persistence(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, "gpt-4", None, None)
        key2 = _make_cache_key(False, "gpt-4", None, None)

        assert key1 != key2

    def test_cache_key_different_model(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, "gpt-4", None, None)
        key2 = _make_cache_key(True, "gpt-4o", None, None)

        assert key1 != key2

    def test_cache_key_with_agent_configs(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, None, None, {"coder": {"model": "gpt-4"}})
        key2 = _make_cache_key(True, None, None, {"coder": {"model": "gpt-4o"}})

        assert key1 != key2

    def test_cache_key_with_provider(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, None, "openai", None)
        key2 = _make_cache_key(True, None, "anthropic", None)

        assert key1 != key2


class TestPipelineBuilderBuild:
    def test_build_empty_pipeline_raises_error(self) -> None:
        """Empty pipelines with no nodes/edges are invalid - validation catches this."""
        from src.graph.pipeline_builder import PipelineBuilder, PipelineValidationError
        from src.storage.models import Pipeline

        pipeline = MagicMock(spec=Pipeline)
        pipeline.id = "test-empty"
        pipeline.nodes = []
        pipeline.edges = []
        pipeline.settings = {}

        with (
            patch("src.graph.pipeline_builder.get_registry") as mock_registry,
            patch("src.graph.pipeline_builder.create_checkpointer") as mock_checkpointer,
        ):
            mock_registry.return_value.get.return_value = None
            mock_registry.return_value.get_agent_names.return_value = []
            mock_checkpointer.return_value = MagicMock()

            builder = PipelineBuilder(pipeline)

            with pytest.raises(PipelineValidationError, match="no agent or team nodes"):
                builder.build()

    def test_build_pipeline_with_agents(self) -> None:
        from src.graph.pipeline_builder import PipelineBuilder
        from src.storage.models import Pipeline

        pipeline = MagicMock(spec=Pipeline)
        pipeline.id = "test-agents"
        pipeline.nodes = [
            {"id": "start", "type": "start"},
            {"id": "agent1", "type": "agent", "data": {"agentName": "coder"}},
            {"id": "end", "type": "end"},
        ]
        pipeline.edges = [
            {"source": "start", "target": "agent1"},
            {"source": "agent1", "target": "end"},
        ]
        pipeline.settings = {}

        with (
            patch("src.graph.pipeline_builder.get_registry") as mock_registry,
            patch("src.graph.pipeline_builder.create_checkpointer") as mock_checkpointer,
        ):
            mock_agent_def = MagicMock()
            mock_agent_def.factory = MagicMock()
            mock_registry.return_value.get.return_value = mock_agent_def
            mock_registry.return_value.get_agent_names.return_value = ["coder"]

            mock_agent = MagicMock()
            mock_agent.invoke = AsyncMock()
            mock_registry.return_value.create_agent.return_value = mock_agent

            mock_checkpointer.return_value = MagicMock()

            builder = PipelineBuilder(pipeline)
            compiled = builder.build()

            assert compiled is not None

    def test_build_pipeline_with_team(self) -> None:
        from src.graph.pipeline_builder import PipelineBuilder
        from src.storage.models import Pipeline

        pipeline = MagicMock(spec=Pipeline)
        pipeline.id = "test-team"
        pipeline.nodes = [
            {"id": "start", "type": "start"},
            {
                "id": "team1",
                "type": "team",
                "data": {
                    "teamName": "Dev Team",
                    "agentIds": ["coder", "reviewer"],
                    "leadAgentId": "coder",
                },
            },
            {"id": "end", "type": "end"},
        ]
        pipeline.edges = [
            {"source": "start", "target": "team1"},
            {"source": "team1", "target": "end"},
        ]
        pipeline.settings = {}

        with (
            patch("src.graph.pipeline_builder.get_registry") as mock_registry,
            patch("src.graph.pipeline_builder.create_checkpointer") as mock_checkpointer,
        ):
            mock_agent_def = MagicMock()
            mock_agent_def.factory = MagicMock()
            mock_registry.return_value.get.return_value = mock_agent_def
            mock_registry.return_value.get_agent_names.return_value = ["coder", "reviewer"]

            mock_agent = MagicMock()
            mock_agent.invoke = AsyncMock(return_value={"messages": [AIMessage(content="Done")]})
            mock_registry.return_value.create_agent.return_value = mock_agent

            mock_checkpointer.return_value = MagicMock()

            builder = PipelineBuilder(pipeline)
            compiled = builder.build()

            assert compiled is not None


class TestPipelineConditionalEdges:
    def test_build_pipeline_with_conditional_edge(self) -> None:
        from src.graph.pipeline_builder import PipelineBuilder
        from src.storage.models import Pipeline

        pipeline = MagicMock(spec=Pipeline)
        pipeline.id = "test-conditional"
        pipeline.nodes = [
            {"id": "start", "type": "start"},
            {"id": "agent1", "type": "agent", "data": {"agentName": "coder"}},
            {"id": "agent2", "type": "agent", "data": {"agentName": "reviewer"}},
            {"id": "end", "type": "end"},
        ]
        pipeline.edges = [
            {"source": "start", "target": "agent1"},
            {
                "id": "edge1",
                "source": "agent1",
                "target": "agent2",
                "data": {
                    "edgeType": "conditional",
                    "condition": {"type": "contains", "value": "NEEDS_REVIEW"},
                },
            },
            {
                "id": "edge2",
                "source": "agent1",
                "target": "end",
                "data": {"edgeType": "default"},
            },
        ]
        pipeline.settings = {}

        with (
            patch("src.graph.pipeline_builder.get_registry") as mock_registry,
            patch("src.graph.pipeline_builder.create_checkpointer") as mock_checkpointer,
        ):
            mock_agent_def = MagicMock()
            mock_agent_def.factory = MagicMock()
            mock_registry.return_value.get.return_value = mock_agent_def
            mock_registry.return_value.get_agent_names.return_value = ["coder", "reviewer"]

            mock_agent = MagicMock()
            mock_agent.invoke = AsyncMock()
            mock_registry.return_value.create_agent.return_value = mock_agent

            mock_checkpointer.return_value = MagicMock()

            builder = PipelineBuilder(pipeline)
            compiled = builder.build()

            assert compiled is not None

    def test_build_pipeline_with_feedback_loop(self) -> None:
        from src.graph.pipeline_builder import PipelineBuilder
        from src.storage.models import Pipeline

        pipeline = MagicMock(spec=Pipeline)
        pipeline.id = "test-feedback"
        pipeline.nodes = [
            {"id": "start", "type": "start"},
            {"id": "writer", "type": "agent", "data": {"agentName": "writer"}},
            {"id": "reviewer", "type": "agent", "data": {"agentName": "reviewer"}},
            {"id": "end", "type": "end"},
        ]
        pipeline.edges = [
            {"source": "start", "target": "writer"},
            {"source": "writer", "target": "reviewer"},
            {
                "id": "feedback",
                "source": "reviewer",
                "target": "writer",
                "data": {
                    "edgeType": "feedback",
                    "condition": {"type": "contains", "value": "REVISION"},
                    "maxIterations": 3,
                },
            },
            {
                "id": "approved",
                "source": "reviewer",
                "target": "end",
                "data": {
                    "edgeType": "conditional",
                    "condition": {"type": "contains", "value": "APPROVED"},
                },
            },
        ]
        pipeline.settings = {}

        with (
            patch("src.graph.pipeline_builder.get_registry") as mock_registry,
            patch("src.graph.pipeline_builder.create_checkpointer") as mock_checkpointer,
        ):
            mock_agent_def = MagicMock()
            mock_agent_def.factory = MagicMock()
            mock_registry.return_value.get.return_value = mock_agent_def
            mock_registry.return_value.get_agent_names.return_value = ["writer", "reviewer"]

            mock_agent = MagicMock()
            mock_agent.invoke = AsyncMock()
            mock_registry.return_value.create_agent.return_value = mock_agent

            mock_checkpointer.return_value = MagicMock()

            builder = PipelineBuilder(pipeline)
            compiled = builder.build()

            assert compiled is not None


class TestTeamExecutor:
    @pytest.mark.asyncio
    async def test_team_executor_sequential_execution(self) -> None:
        from src.graph.pipeline_builder import PipelineBuilder
        from src.storage.models import Pipeline

        pipeline = MagicMock(spec=Pipeline)
        pipeline.nodes = [
            {
                "id": "team1",
                "type": "team",
                "data": {
                    "agentIds": ["agent1", "agent2"],
                    "leadAgentId": "agent1",
                },
            }
        ]
        pipeline.edges = []
        pipeline.settings = {}

        execution_order = []

        async def mock_invoke_agent1(state: AgentState) -> dict[str, Any]:
            execution_order.append("agent1")
            return {"messages": [AIMessage(content="Agent1 done")]}

        async def mock_invoke_agent2(state: AgentState) -> dict[str, Any]:
            execution_order.append("agent2")
            return {"messages": [AIMessage(content="Agent2 done")]}

        with patch("src.graph.pipeline_builder.get_registry") as mock_registry:
            mock_agent_def = MagicMock()
            mock_agent_def.factory = MagicMock()
            mock_registry.return_value.get.return_value = mock_agent_def

            mock_agent1 = MagicMock()
            mock_agent1.invoke = mock_invoke_agent1
            mock_agent2 = MagicMock()
            mock_agent2.invoke = mock_invoke_agent2

            call_count = [0]

            def create_agent(agent_id: str, can_handoff: bool = True) -> MagicMock:
                if call_count[0] == 0:
                    call_count[0] += 1
                    return mock_agent1
                return mock_agent2

            mock_registry.return_value.create_agent.side_effect = create_agent

            builder = PipelineBuilder(pipeline)
            team_node = pipeline.nodes[0]

            executor = builder._create_team_executor(team_node)

            state: AgentState = {
                "messages": [HumanMessage(content="Start")],
                "current_agent": "user",
                "context": {},
                "human_feedback": None,
                "iteration_count": 0,
                "execution_trace": [],
            }

            result = await executor(state)

            assert execution_order == ["agent1", "agent2"]
            assert len(result["messages"]) == 2

    def test_team_executor_no_agents(self) -> None:
        from src.graph.pipeline_builder import PipelineBuilder
        from src.storage.models import Pipeline

        pipeline = MagicMock(spec=Pipeline)
        pipeline.nodes = []
        pipeline.edges = []
        pipeline.settings = {}

        builder = PipelineBuilder(pipeline)
        team_node = {
            "id": "team1",
            "type": "team",
            "data": {"agentIds": [], "leadAgentId": None},
        }

        with patch("src.graph.pipeline_builder.get_registry"):
            executor = builder._create_team_executor(team_node)

        assert executor is None


class TestEdgeConditionAdvanced:
    def test_condition_with_none_content(self) -> None:
        from src.graph.pipeline_builder import EdgeCondition

        condition = EdgeCondition("contains", "test")
        mock_msg = MagicMock()
        mock_msg.content = None
        state: dict[str, Any] = {"messages": [mock_msg]}

        result = condition.evaluate(state)

        assert result is False

    def test_condition_context_field_missing(self) -> None:
        from src.graph.pipeline_builder import EdgeCondition

        condition = EdgeCondition("contains", "test", field="context")
        state: dict[str, Any] = {}  # No context

        result = condition.evaluate(state)

        assert result is False

    def test_condition_unknown_field(self) -> None:
        from src.graph.pipeline_builder import EdgeCondition

        condition = EdgeCondition("contains", "test", field="unknown_field")
        state: dict[str, Any] = {"messages": [MagicMock(content="test")]}

        result = condition.evaluate(state)

        assert result is False


class TestGetPipelineWorkflow:
    @pytest.mark.asyncio
    async def test_get_pipeline_workflow_by_id(self) -> None:
        from src.graph.pipeline_builder import get_pipeline_workflow
        from src.storage.models import Pipeline

        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.id = "pipeline-123"
        mock_pipeline.nodes = [
            {"id": "start", "type": "start"},
            {"id": "agent1", "type": "agent", "data": {"agentName": "coder"}},
            {"id": "end", "type": "end"},
        ]
        mock_pipeline.edges = [
            {"source": "start", "target": "agent1"},
            {"source": "agent1", "target": "end"},
        ]
        mock_pipeline.settings = {}

        mock_repo = MagicMock()
        mock_repo.get_pipeline = AsyncMock(return_value=mock_pipeline)

        with (
            patch("src.storage.get_repository", AsyncMock(return_value=mock_repo)),
            patch("src.graph.pipeline_builder.get_registry") as mock_registry,
            patch("src.graph.pipeline_builder.create_checkpointer") as mock_checkpointer,
        ):
            mock_agent_def = MagicMock()
            mock_agent_def.factory = MagicMock()
            mock_registry.return_value.get.return_value = mock_agent_def
            mock_registry.return_value.get_agent_names.return_value = ["coder"]

            mock_agent = MagicMock()
            mock_agent.invoke = AsyncMock()
            mock_registry.return_value.create_agent.return_value = mock_agent

            mock_checkpointer.return_value = MagicMock()

            workflow = await get_pipeline_workflow("pipeline-123")

            assert workflow is not None
            mock_repo.get_pipeline.assert_called_once_with("pipeline-123")

    @pytest.mark.asyncio
    async def test_get_pipeline_workflow_default(self) -> None:
        from src.graph.pipeline_builder import get_pipeline_workflow
        from src.storage.models import Pipeline

        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.id = "default-pipeline"
        mock_pipeline.nodes = [
            {"id": "start", "type": "start"},
            {"id": "agent1", "type": "agent", "data": {"agentName": "writer"}},
            {"id": "end", "type": "end"},
        ]
        mock_pipeline.edges = [
            {"source": "start", "target": "agent1"},
            {"source": "agent1", "target": "end"},
        ]
        mock_pipeline.settings = {}

        mock_repo = MagicMock()
        mock_repo.get_default_pipeline = AsyncMock(return_value=mock_pipeline)

        with (
            patch("src.storage.get_repository", AsyncMock(return_value=mock_repo)),
            patch("src.graph.pipeline_builder.get_registry") as mock_registry,
            patch("src.graph.pipeline_builder.create_checkpointer") as mock_checkpointer,
        ):
            mock_agent_def = MagicMock()
            mock_agent_def.factory = MagicMock()
            mock_registry.return_value.get.return_value = mock_agent_def
            mock_registry.return_value.get_agent_names.return_value = ["writer"]

            mock_agent = MagicMock()
            mock_agent.invoke = AsyncMock()
            mock_registry.return_value.create_agent.return_value = mock_agent

            mock_checkpointer.return_value = MagicMock()

            workflow = await get_pipeline_workflow()

            assert workflow is not None
            mock_repo.get_default_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pipeline_workflow_not_found(self) -> None:
        from src.graph.pipeline_builder import get_pipeline_workflow

        mock_repo = MagicMock()
        mock_repo.get_pipeline = AsyncMock(return_value=None)

        with patch("src.storage.get_repository", AsyncMock(return_value=mock_repo)):
            workflow = await get_pipeline_workflow("nonexistent")

            assert workflow is None
