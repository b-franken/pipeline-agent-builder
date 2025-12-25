from typing import Any
from unittest.mock import MagicMock, patch

from src.graph.pipeline_builder import EdgeCondition, PipelineBuilder
from src.storage.models import Pipeline


class TestEdgeCondition:
    def test_contains_match(self) -> None:
        condition = EdgeCondition("contains", "APPROVED")
        state: dict[str, Any] = {"messages": [MagicMock(content="This is APPROVED by reviewer")]}

        assert condition.evaluate(state) is True

    def test_contains_no_match(self) -> None:
        condition = EdgeCondition("contains", "APPROVED")
        state: dict[str, Any] = {"messages": [MagicMock(content="This needs revision")]}

        assert condition.evaluate(state) is False

    def test_contains_case_insensitive(self) -> None:
        condition = EdgeCondition("contains", "approved")
        state: dict[str, Any] = {"messages": [MagicMock(content="APPROVED")]}

        assert condition.evaluate(state) is True

    def test_not_contains_match(self) -> None:
        condition = EdgeCondition("not_contains", "ERROR")
        state: dict[str, Any] = {"messages": [MagicMock(content="All good")]}

        assert condition.evaluate(state) is True

    def test_not_contains_no_match(self) -> None:
        condition = EdgeCondition("not_contains", "ERROR")
        state: dict[str, Any] = {"messages": [MagicMock(content="ERROR occurred")]}

        assert condition.evaluate(state) is False

    def test_equals_match(self) -> None:
        condition = EdgeCondition("equals", "DONE")
        state: dict[str, Any] = {"messages": [MagicMock(content="  DONE  ")]}

        assert condition.evaluate(state) is True

    def test_equals_no_match(self) -> None:
        condition = EdgeCondition("equals", "DONE")
        state: dict[str, Any] = {"messages": [MagicMock(content="DONE and more")]}

        assert condition.evaluate(state) is False

    def test_regex_match(self) -> None:
        condition = EdgeCondition("regex", r"(fail|error|issue)")
        state: dict[str, Any] = {"messages": [MagicMock(content="There was an error")]}

        assert condition.evaluate(state) is True

    def test_regex_no_match(self) -> None:
        condition = EdgeCondition("regex", r"(fail|error|issue)")
        state: dict[str, Any] = {"messages": [MagicMock(content="All good")]}

        assert condition.evaluate(state) is False

    def test_regex_invalid_pattern(self) -> None:
        condition = EdgeCondition("regex", r"[invalid(")
        state: dict[str, Any] = {"messages": [MagicMock(content="test")]}

        assert condition.evaluate(state) is False

    def test_empty_messages(self) -> None:
        condition = EdgeCondition("contains", "test")
        state: dict[str, Any] = {"messages": []}

        assert condition.evaluate(state) is False

    def test_context_field(self) -> None:
        condition = EdgeCondition("contains", "important", field="context")
        state: dict[str, Any] = {"context": {"key": "important value"}}

        assert condition.evaluate(state) is True

    def test_unknown_condition_type(self) -> None:
        condition = EdgeCondition("unknown_type", "test")
        state: dict[str, Any] = {"messages": [MagicMock(content="test")]}

        assert condition.evaluate(state) is False

    def test_message_without_content_attribute(self) -> None:
        condition = EdgeCondition("contains", "test")
        state: dict[str, Any] = {"messages": ["plain string message with test"]}

        assert condition.evaluate(state) is True


class TestPipelineBuilder:
    def _create_pipeline(
        self,
        nodes: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> Pipeline:
        pipeline = MagicMock(spec=Pipeline)
        pipeline.nodes = nodes or []
        pipeline.edges = edges or []
        pipeline.settings = {}
        return pipeline

    def test_build_node_map(self) -> None:
        nodes = [
            {"id": "node1", "type": "agent"},
            {"id": "node2", "type": "agent"},
        ]
        pipeline = self._create_pipeline(nodes=nodes)
        builder = PipelineBuilder(pipeline)

        node_map = builder._build_node_map()

        assert "node1" in node_map
        assert "node2" in node_map
        assert node_map["node1"]["type"] == "agent"

    def test_get_agent_nodes(self) -> None:
        nodes = [
            {"id": "node1", "type": "agent"},
            {"id": "node2", "type": "team"},
            {"id": "node3", "type": "agent"},
        ]
        pipeline = self._create_pipeline(nodes=nodes)
        builder = PipelineBuilder(pipeline)

        agent_nodes = builder._get_agent_nodes()

        assert len(agent_nodes) == 2
        assert all(n["type"] == "agent" for n in agent_nodes)

    def test_get_team_nodes(self) -> None:
        nodes = [
            {"id": "node1", "type": "agent"},
            {"id": "node2", "type": "team"},
            {"id": "node3", "type": "team"},
        ]
        pipeline = self._create_pipeline(nodes=nodes)
        builder = PipelineBuilder(pipeline)

        team_nodes = builder._get_team_nodes()

        assert len(team_nodes) == 2
        assert all(n["type"] == "team" for n in team_nodes)

    def test_find_start_node(self) -> None:
        nodes = [
            {"id": "start_node", "type": "start"},
            {"id": "agent1", "type": "agent"},
        ]
        pipeline = self._create_pipeline(nodes=nodes)
        builder = PipelineBuilder(pipeline)

        start_id = builder._find_start_node()

        assert start_id == "start_node"

    def test_find_start_node_not_found(self) -> None:
        nodes = [{"id": "agent1", "type": "agent"}]
        pipeline = self._create_pipeline(nodes=nodes)
        builder = PipelineBuilder(pipeline)

        start_id = builder._find_start_node()

        assert start_id is None

    def test_find_end_node(self) -> None:
        nodes = [
            {"id": "agent1", "type": "agent"},
            {"id": "end_node", "type": "end"},
        ]
        pipeline = self._create_pipeline(nodes=nodes)
        builder = PipelineBuilder(pipeline)

        end_id = builder._find_end_node()

        assert end_id == "end_node"

    def test_find_first_node_after_start(self) -> None:
        nodes = [
            {"id": "start", "type": "start"},
            {"id": "agent1", "type": "agent"},
        ]
        edges = [{"source": "start", "target": "agent1"}]
        pipeline = self._create_pipeline(nodes=nodes, edges=edges)
        builder = PipelineBuilder(pipeline)
        builder._node_map = builder._build_node_map()

        first_node = builder._find_first_node_after_start()

        assert first_node == "agent1"

    def test_extract_agent_name_from_data(self) -> None:
        node = {"id": "node1", "type": "agent", "data": {"agentName": "coder"}}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        name = builder._extract_agent_name(node)

        assert name == "coder"

    def test_extract_agent_name_fallback_to_name(self) -> None:
        node = {"id": "node1", "type": "agent", "data": {"name": "researcher"}}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        name = builder._extract_agent_name(node)

        assert name == "researcher"

    def test_extract_agent_name_fallback_to_label(self) -> None:
        node = {"id": "node1", "type": "agent", "data": {"label": "writer"}}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        name = builder._extract_agent_name(node)

        assert name == "writer"

    def test_get_edge_config(self) -> None:
        edge = {"id": "edge1", "data": {"edgeType": "conditional", "priority": 1}}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        config = builder._get_edge_config(edge)

        assert config["edgeType"] == "conditional"
        assert config["priority"] == 1

    def test_get_edge_config_empty(self) -> None:
        edge = {"id": "edge1"}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        config = builder._get_edge_config(edge)

        assert config == {}

    def test_group_edges_by_source(self) -> None:
        edges = [
            {"source": "node1", "target": "node2"},
            {"source": "node1", "target": "node3"},
            {"source": "node2", "target": "node3"},
        ]
        pipeline = self._create_pipeline(edges=edges)
        builder = PipelineBuilder(pipeline)

        grouped = builder._group_edges_by_source()

        assert len(grouped["node1"]) == 2
        assert len(grouped["node2"]) == 1

    def test_parse_handle_id_agent_out(self) -> None:
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        agent_id, handle_type = builder._parse_handle_id("agent-coder-out")

        assert agent_id == "coder"
        assert handle_type == "out"

    def test_parse_handle_id_agent_in(self) -> None:
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        agent_id, handle_type = builder._parse_handle_id("agent-researcher-in")

        assert agent_id == "researcher"
        assert handle_type == "in"

    def test_parse_handle_id_not_agent(self) -> None:
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        agent_id, handle_type = builder._parse_handle_id("default-handle")

        assert agent_id is None
        assert handle_type is None

    def test_parse_handle_id_none(self) -> None:
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        agent_id, handle_type = builder._parse_handle_id(None)

        assert agent_id is None
        assert handle_type is None

    def test_resolve_edge_target_individual_agent(self) -> None:
        edge = {"target": "team1", "targetHandle": "agent-coder-in"}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        resolved = builder._resolve_edge_target(edge)

        assert resolved == "team1:coder"

    def test_resolve_edge_target_regular(self) -> None:
        edge = {"target": "node2", "targetHandle": "default"}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        resolved = builder._resolve_edge_target(edge)

        assert resolved == "node2"

    def test_resolve_edge_source_individual_agent(self) -> None:
        edge = {"source": "team1", "sourceHandle": "agent-writer-out"}
        pipeline = self._create_pipeline()
        builder = PipelineBuilder(pipeline)

        resolved = builder._resolve_edge_source(edge)

        assert resolved == "team1:writer"

    def test_find_individually_connected_agents(self) -> None:
        edges = [
            {"source": "team1", "target": "node2", "sourceHandle": "agent-coder-out", "targetHandle": "default"},
            {"source": "node1", "target": "team2", "sourceHandle": "default", "targetHandle": "agent-writer-in"},
        ]
        pipeline = self._create_pipeline(edges=edges)
        builder = PipelineBuilder(pipeline)

        individual = builder._find_individually_connected_agents()

        assert "team1:coder" in individual
        assert "team2:writer" in individual


class TestBuildPipelineWorkflow:
    def test_build_pipeline_workflow_function(self) -> None:
        from src.graph.pipeline_builder import build_pipeline_workflow

        pipeline = MagicMock(spec=Pipeline)
        pipeline.id = "test-pipeline"
        pipeline.nodes = []
        pipeline.edges = []
        pipeline.settings = {}

        with patch.object(PipelineBuilder, "build") as mock_build:
            mock_build.return_value = MagicMock()

            build_pipeline_workflow(pipeline, strict=False)

            mock_build.assert_called_once_with(
                use_persistence=False,
                strict=False,
                auto_connect_dangling=True,
            )
