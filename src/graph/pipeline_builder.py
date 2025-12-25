import logging
import re
from collections.abc import Hashable
from typing import Any, Final

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.memory.checkpointer import create_checkpointer
from src.registry import get_registry
from src.storage.models import Pipeline
from src.types import AgentState

logger: Final = logging.getLogger(__name__)


class ToolNodeRegistry:
    """Registry for tool nodes that can inject tools into agents at runtime."""

    def __init__(self) -> None:
        self._tool_nodes: dict[str, dict[str, Any]] = {}
        self._agent_tools: dict[str, list[str]] = {}  # agent_node_id -> [tool_node_ids]

    def register_tool_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        """Register a tool node with its configuration."""
        self._tool_nodes[node_id] = node_data

    def connect_tool_to_agent(self, tool_node_id: str, agent_node_id: str) -> None:
        """Connect a tool node to an agent node."""
        if agent_node_id not in self._agent_tools:
            self._agent_tools[agent_node_id] = []
        if tool_node_id not in self._agent_tools[agent_node_id]:
            self._agent_tools[agent_node_id].append(tool_node_id)

    def get_tools_for_agent(self, agent_node_id: str) -> list[str]:
        """Get all tool node IDs connected to an agent."""
        return self._agent_tools.get(agent_node_id, [])

    def get_tool_node(self, node_id: str) -> dict[str, Any] | None:
        """Get tool node configuration."""
        return self._tool_nodes.get(node_id)

    async def load_mcp_tools_for_agent(self, agent_node_id: str) -> list[BaseTool]:
        """Load MCP tools for an agent based on connected tool nodes."""
        from src.mcp.client import create_mcp_client_for_servers, is_mcp_available
        from src.storage import get_repository

        tool_node_ids = self.get_tools_for_agent(agent_node_id)
        if not tool_node_ids:
            return []

        if not is_mcp_available():
            logger.warning("MCP not available, skipping tool loading for agent %s", agent_node_id)
            return []

        mcp_server_ids: list[str] = []
        for tool_node_id in tool_node_ids:
            tool_node = self.get_tool_node(tool_node_id)
            if tool_node:
                server_id = tool_node.get("mcpServerId")
                server_ids = tool_node.get("mcpServerIds", [])
                if server_id:
                    mcp_server_ids.append(server_id)
                mcp_server_ids.extend(server_ids)

        if not mcp_server_ids:
            return []

        try:
            repo = await get_repository()
            servers_to_load = []
            for server_id in set(mcp_server_ids):  # Deduplicate
                server = await repo.get_mcp_server(server_id)
                if server and server.is_active:
                    servers_to_load.append(server)

            if servers_to_load:
                logger.info(
                    "Loading %d MCP servers for agent node %s: %s",
                    len(servers_to_load),
                    agent_node_id,
                    [s.name for s in servers_to_load],
                )
                return await create_mcp_client_for_servers(servers_to_load)
        except Exception:
            logger.exception("Failed to load MCP tools for agent node %s", agent_node_id)

        return []


class EdgeCondition:
    """Evaluates conditions on agent state for conditional routing."""

    def __init__(
        self,
        condition_type: str,
        value: str,
        field: str = "last_message",
    ) -> None:
        self.condition_type = condition_type
        self.value = value
        self.field = field

    def evaluate(self, state: AgentState) -> bool:
        """Evaluate the condition against the current state."""
        field_value = self._get_field_value(state)

        if self.condition_type == "exists":
            return field_value is not None
        elif self.condition_type == "not_exists":
            return field_value is None
        elif self.condition_type == "equals_bool":
            return field_value is True if self.value.lower() == "true" else field_value is False

        text = str(field_value) if field_value is not None else ""

        if self.condition_type == "contains":
            return self.value.lower() in text.lower()
        elif self.condition_type == "not_contains":
            return self.value.lower() not in text.lower()
        elif self.condition_type == "equals":
            return text.strip().lower() == self.value.strip().lower()
        elif self.condition_type == "regex":
            try:
                return bool(re.search(self.value, text, re.IGNORECASE))
            except re.error:
                return False
        return False

    def _get_field_value(self, state: AgentState) -> Any:
        """Extract the field value from state using dot notation for nested access."""
        if self.field == "last_message":
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                return str(content) if content else ""
            return ""

        if "." in self.field:
            parts = self.field.split(".")
            current: Any = state
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
                if current is None:
                    return None
            return current

        return state.get(self.field)


class PipelineValidationError(Exception):
    """Raised when pipeline validation fails during build."""

    def __init__(self, pipeline_id: str, errors: list[str]) -> None:
        self.pipeline_id = pipeline_id
        self.errors = errors
        super().__init__(f"Pipeline '{pipeline_id}' validation failed: {'; '.join(errors)}")


class PipelineBuilder:
    """Builds executable LangGraph workflows from pipeline configurations."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.nodes = pipeline.nodes or []
        self.edges = pipeline.edges or []
        self._node_map: dict[str, dict[str, Any]] = {}
        self._tool_registry = ToolNodeRegistry()
        self._preloaded_tools: dict[str, list[BaseTool]] = {}  # agent_node_id -> tools

    def validate(self) -> list[str]:
        """Validate pipeline configuration before building.

        Returns list of error messages. Empty list means valid.
        """
        errors: list[str] = []
        registry = get_registry()
        registered_agents = registry.get_agent_names()

        agent_nodes = self._get_agent_nodes()
        team_nodes = self._get_team_nodes()

        if not agent_nodes and not team_nodes:
            errors.append("Pipeline has no agent or team nodes")

        for node in agent_nodes:
            node_id = node.get("id", "unknown")
            agent_name = self._extract_agent_name(node)
            if not agent_name:
                errors.append(f"Agent node '{node_id}' has no agent name configured")
            elif agent_name not in registered_agents:
                errors.append(f"Agent node '{node_id}' references unknown agent: {agent_name}")

        for node in team_nodes:
            node_id = node.get("id", "unknown")
            data = node.get("data", {})
            agent_ids = data.get("agentIds", [])
            if not agent_ids:
                errors.append(f"Team node '{node_id}' has no agents configured")
            for agent_id in agent_ids:
                if agent_id not in registered_agents:
                    errors.append(f"Team node '{node_id}' references unknown agent: {agent_id}")

        node_ids = {n.get("id") for n in self.nodes if n.get("id")}
        for edge in self.edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and source not in node_ids:
                errors.append(f"Edge references unknown source node: {source}")
            if target and target not in node_ids:
                errors.append(f"Edge references unknown target node: {target}")

        return errors

    async def _process_tool_nodes(self) -> None:
        """Process tool nodes and their connections to agents.

        Tool nodes are connected to agent nodes via edges. This method:
        1. Registers all tool nodes
        2. Finds edges from tool nodes to agent nodes
        3. Preloads MCP tools for each agent based on connected tool nodes
        """
        tool_nodes = self._get_tool_nodes()
        if not tool_nodes:
            return

        for node in tool_nodes:
            node_id = node.get("id", "")
            node_data = node.get("data", {})
            self._tool_registry.register_tool_node(node_id, node_data)

        for edge in self.edges:
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")

            source_node = self._node_map.get(source_id, {})
            target_node = self._node_map.get(target_id, {})

            if source_node.get("type") == "tool" and target_node.get("type") in ("agent", "team"):
                self._tool_registry.connect_tool_to_agent(source_id, target_id)
                logger.debug(
                    "Connected tool node '%s' to agent node '%s'",
                    source_id,
                    target_id,
                )

        for agent_node_id in self._tool_registry._agent_tools:
            tools = await self._tool_registry.load_mcp_tools_for_agent(agent_node_id)
            if tools:
                self._preloaded_tools[agent_node_id] = tools
                logger.info(
                    "Preloaded %d MCP tools for agent node '%s'",
                    len(tools),
                    agent_node_id,
                )

    def _get_tool_nodes(self) -> list[dict[str, Any]]:
        """Get all tool nodes from the pipeline."""
        return [n for n in self.nodes if n.get("type") == "tool"]

    async def build_async(
        self,
        use_persistence: bool = False,
        strict: bool = True,
        auto_connect_dangling: bool = True,
    ) -> CompiledStateGraph[Any]:
        """Build the pipeline workflow asynchronously (supports MCP tool loading).

        Args:
            use_persistence: Whether to use persistent checkpointing.
            strict: If True, raises PipelineValidationError on validation failures.
                   If False, logs warnings and attempts to build anyway.
            auto_connect_dangling: If True, nodes without outgoing edges are auto-connected to END.
                   If False and strict=True, dangling nodes cause validation failure.
        """
        if strict:
            errors = self.validate()
            if errors:
                raise PipelineValidationError(self.pipeline.id, errors)

        self._node_map = self._build_node_map()

        await self._process_tool_nodes()

        return self._build_graph(use_persistence, auto_connect_dangling, strict)

    def build(
        self,
        use_persistence: bool = False,
        strict: bool = True,
        auto_connect_dangling: bool = True,
    ) -> CompiledStateGraph[Any]:
        """Build the pipeline workflow (sync version, no MCP tool loading from tool nodes).

        For MCP tool support via tool nodes, use build_async() instead.

        Args:
            use_persistence: Whether to use persistent checkpointing.
            strict: If True, raises PipelineValidationError on validation failures.
                   If False, logs warnings and attempts to build anyway.
            auto_connect_dangling: If True, nodes without outgoing edges are auto-connected to END.
                   If False and strict=True, dangling nodes cause validation failure.
        """
        if strict:
            errors = self.validate()
            if errors:
                raise PipelineValidationError(self.pipeline.id, errors)

        self._node_map = self._build_node_map()
        return self._build_graph(use_persistence, auto_connect_dangling, strict)

    def _build_graph(
        self,
        use_persistence: bool = False,
        auto_connect_dangling: bool = True,
        strict: bool = True,
    ) -> CompiledStateGraph[Any]:
        """Internal method to build the graph after node processing."""
        registry = get_registry()
        graph: StateGraph[AgentState] = StateGraph(AgentState)

        agent_nodes = self._get_agent_nodes()
        team_nodes = self._get_team_nodes()
        individual_agents_used = self._find_individually_connected_agents()

        for node in agent_nodes:
            node_id = node.get("id", "")
            agent_name = self._extract_agent_name(node)
            if not agent_name:
                logger.warning("Pipeline '%s': agent node '%s' has no agent name, skipping", self.pipeline.id, node_id)
                continue

            agent_def = registry.get(agent_name)
            if not agent_def or not agent_def.factory:
                logger.warning(
                    "Pipeline '%s': agent '%s' not in registry",
                    self.pipeline.id,
                    agent_name,
                )
                continue

            extra_tools = self._preloaded_tools.get(node_id, [])
            if extra_tools:
                logger.info(
                    "Pipeline '%s': injecting %d tools from tool nodes into agent '%s'",
                    self.pipeline.id,
                    len(extra_tools),
                    agent_name,
                )
                agent = registry.create_agent(agent_name, can_handoff=False, tools=extra_tools)
            else:
                agent = registry.create_agent(agent_name, can_handoff=False)
            graph.add_node(node_id, agent.invoke)

        for node in team_nodes:
            node_id = node.get("id", "")
            data = node.get("data", {})
            team_agent_ids = data.get("agentIds", [])

            team_has_individual_connections = any(
                f"{node_id}:{agent_id}" in individual_agents_used for agent_id in team_agent_ids
            )

            if team_has_individual_connections:
                for agent_id in team_agent_ids:
                    individual_node_id = f"{node_id}:{agent_id}"
                    if individual_node_id in individual_agents_used:
                        agent_executor = self._create_individual_agent_executor(agent_id)
                        if agent_executor:
                            graph.add_node(individual_node_id, agent_executor)
            else:
                team_executor = self._create_team_executor(node)
                if team_executor:
                    graph.add_node(node_id, team_executor)

        first_node_id = self._find_first_node_after_start()
        if first_node_id:
            graph.add_edge(START, first_node_id)
        elif agent_nodes:
            first_node_id = agent_nodes[0].get("id", "")
            if first_node_id:
                graph.add_edge(START, first_node_id)

        edges_by_source = self._group_edges_by_resolved_source()

        for resolved_source_id, source_edges in edges_by_source.items():
            base_source_id = resolved_source_id.split(":")[0]
            source_node = self._node_map.get(base_source_id, {})
            source_type = source_node.get("type", "")

            if source_type in ("start", "tool"):
                continue

            has_conditional = any(
                self._get_edge_config(e).get("edgeType") in ("conditional", "feedback") for e in source_edges
            )

            if has_conditional:
                self._add_conditional_edges(graph, resolved_source_id, source_edges, individual_agents_used)
            else:
                for edge in source_edges:
                    resolved_target = self._resolve_edge_target(edge)
                    base_target_id = resolved_target.split(":")[0]
                    target_node = self._node_map.get(base_target_id, {})
                    target_type = target_node.get("type", "")

                    if target_type == "end":
                        graph.add_edge(resolved_source_id, END)
                    elif target_type in ("agent", "team"):
                        graph.add_edge(resolved_source_id, resolved_target)

        all_added_nodes = set()
        for node in agent_nodes:
            all_added_nodes.add(node.get("id", ""))
        for node in team_nodes:
            node_id = node.get("id", "")
            data = node.get("data", {})
            team_agent_ids = data.get("agentIds", [])
            has_individual = any(f"{node_id}:{aid}" in individual_agents_used for aid in team_agent_ids)
            if has_individual:
                for aid in team_agent_ids:
                    if f"{node_id}:{aid}" in individual_agents_used:
                        all_added_nodes.add(f"{node_id}:{aid}")
            else:
                all_added_nodes.add(node_id)

        dangling_nodes: list[str] = []
        for node_id in all_added_nodes:
            has_outgoing = any(self._resolve_edge_source(e) == node_id for e in self.edges)
            if not has_outgoing:
                dangling_nodes.append(node_id)

        if dangling_nodes:
            if auto_connect_dangling:
                for node_id in dangling_nodes:
                    logger.warning(
                        "Pipeline '%s': node '%s' has no outgoing edges, auto-connecting to END",
                        self.pipeline.id,
                        node_id,
                    )
                    graph.add_edge(node_id, END)
            elif strict:
                raise PipelineValidationError(
                    self.pipeline.id,
                    [f"Node '{node_id}' has no outgoing edges" for node_id in dangling_nodes],
                )

        checkpointer = create_checkpointer() if use_persistence else create_checkpointer(backend="memory")
        return graph.compile(checkpointer=checkpointer)

    def _create_team_executor(self, team_node: dict[str, Any]) -> Any:
        """Create an executor function for a team node."""
        registry = get_registry()
        data = team_node.get("data", {})

        agent_ids = data.get("agentIds", [])
        lead_agent_id = data.get("leadAgentId")

        if not agent_ids:
            return None

        ordered_agent_ids = []
        if lead_agent_id and lead_agent_id in agent_ids:
            ordered_agent_ids.append(lead_agent_id)
            ordered_agent_ids.extend([a for a in agent_ids if a != lead_agent_id])
        else:
            ordered_agent_ids = list(agent_ids)

        agents = []
        for agent_id in ordered_agent_ids:
            agent_def = registry.get(agent_id)
            if agent_def and agent_def.factory:
                agents.append(registry.create_agent(agent_id, can_handoff=False))

        if not agents:
            return None

        async def team_executor(state: AgentState) -> dict[str, Any]:
            """Execute team agents sequentially, preserving full state."""
            current_state: dict[str, Any] = dict(state)
            all_messages: list[Any] = []
            merged_context: dict[str, Any] = dict(current_state.get("context", {}))
            execution_trace: list[str] = list(current_state.get("execution_trace", []))

            for agent in agents:
                result = await agent.invoke(current_state)

                if isinstance(result, dict):
                    new_messages: list[Any] = result.get("messages", [])
                    all_messages.extend(new_messages)
                    existing_messages: list[Any] = current_state.get("messages", [])
                    current_state["messages"] = existing_messages + new_messages

                    if "context" in result and isinstance(result["context"], dict):
                        merged_context.update(result["context"])
                        current_state["context"] = merged_context

                    if "execution_trace" in result:
                        new_trace = result["execution_trace"]
                        if isinstance(new_trace, list):
                            for item in new_trace:
                                if item not in execution_trace:
                                    execution_trace.append(item)
                            current_state["execution_trace"] = execution_trace

            return {
                "messages": all_messages,
                "context": merged_context,
                "execution_trace": execution_trace,
            }

        return team_executor

    def _create_individual_agent_executor(self, agent_id: str) -> Any:
        """Create an executor function for an individual agent within a team."""
        registry = get_registry()
        agent_def = registry.get(agent_id)
        if not agent_def or not agent_def.factory:
            return None

        agent = registry.create_agent(agent_id, can_handoff=False)
        return agent.invoke

    def _parse_handle_id(self, handle_id: str | None) -> tuple[str | None, str | None]:
        """Parse a handle ID to extract agent ID if it's an individual agent handle.

        Returns (agent_id, handle_type) or (None, None) if not an agent handle.
        Format: agent-{agent_id}-in or agent-{agent_id}-out or agent-{agent_id}-bottom
        """
        if not handle_id or not handle_id.startswith("agent-"):
            return None, None

        parts = handle_id.split("-")
        if len(parts) >= 3:
            handle_type = parts[-1]
            if handle_type in ("in", "out", "bottom"):
                agent_id = "-".join(parts[1:-1])
                return agent_id, handle_type
        return None, None

    def _find_individually_connected_agents(self) -> set[str]:
        """Find all individual agents within teams that have direct connections.

        Returns a set of node IDs in format 'team_node_id:agent_id'.
        """
        individual_agents: set[str] = set()

        for edge in self.edges:
            source_handle = edge.get("sourceHandle", "")
            target_handle = edge.get("targetHandle", "")
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")

            source_agent_id, _ = self._parse_handle_id(source_handle)
            if source_agent_id:
                individual_agents.add(f"{source_id}:{source_agent_id}")

            target_agent_id, _ = self._parse_handle_id(target_handle)
            if target_agent_id:
                individual_agents.add(f"{target_id}:{target_agent_id}")

        return individual_agents

    def _resolve_edge_target(self, edge: dict[str, Any]) -> str:
        """Resolve the actual target node ID for an edge.

        If the edge targets an individual agent within a team, returns 'team_id:agent_id'.
        Otherwise returns the regular target node ID.
        """
        target_id: str = edge.get("target", "") or ""
        target_handle: str = edge.get("targetHandle", "") or ""

        agent_id, _ = self._parse_handle_id(target_handle)
        if agent_id:
            return f"{target_id}:{agent_id}"

        return target_id

    def _resolve_edge_source(self, edge: dict[str, Any]) -> str:
        """Resolve the actual source node ID for an edge.

        If the edge originates from an individual agent within a team, returns 'team_id:agent_id'.
        Otherwise returns the regular source node ID.
        """
        source_id: str = edge.get("source", "") or ""
        source_handle: str = edge.get("sourceHandle", "") or ""

        agent_id, _ = self._parse_handle_id(source_handle)
        if agent_id:
            return f"{source_id}:{agent_id}"

        return source_id

    def _add_conditional_edges(
        self,
        graph: StateGraph[AgentState],
        source_id: str,
        edges: list[dict[str, Any]],
        individual_agents_used: set[str] | None = None,
    ) -> None:
        sorted_edges = sorted(
            edges,
            key=lambda e: self._get_edge_config(e).get("priority", 0),
            reverse=True,
        )

        routing_info: list[tuple[str, str, dict[str, Any]]] = []
        for edge in sorted_edges:
            resolved_target = self._resolve_edge_target(edge)
            base_target_id = resolved_target.split(":")[0]
            target_node = self._node_map.get(base_target_id, {})
            target_type = target_node.get("type", "")

            actual_target = END if target_type == "end" else resolved_target
            config = self._get_edge_config(edge)
            routing_info.append((edge.get("id", ""), actual_target, config))

        def routing_function(state: AgentState) -> str:
            loop_counts: dict[str, int] = dict(state.get("loop_counts", {}))

            for edge_id, target, config in routing_info:
                edge_type = config.get("edgeType", "default")

                if edge_type == "feedback":
                    max_iterations = config.get("maxIterations", 3)
                    current_count = loop_counts.get(edge_id, 0)
                    if current_count >= max_iterations:
                        continue

                    condition_data = config.get("condition", {})
                    condition_matches = True
                    if condition_data:
                        condition = EdgeCondition(
                            condition_type=condition_data.get("type", "contains"),
                            value=condition_data.get("value", ""),
                            field=condition_data.get("field", "last_message"),
                        )
                        condition_matches = condition.evaluate(state)

                    if condition_matches:
                        loop_counts[edge_id] = current_count + 1
                        state["loop_counts"] = loop_counts
                        return target

                elif edge_type == "conditional":
                    condition_data = config.get("condition", {})
                    if condition_data:
                        condition = EdgeCondition(
                            condition_type=condition_data.get("type", "contains"),
                            value=condition_data.get("value", ""),
                            field=condition_data.get("field", "last_message"),
                        )
                        if condition.evaluate(state):
                            return target
                    else:
                        return target
                else:
                    return target

            return END

        path_map: dict[Hashable, str] = {END: END}
        for _, target, _ in routing_info:
            if target != END:
                path_map[target] = target

        graph.add_conditional_edges(source_id, routing_function, path_map)

    def _get_edge_config(self, edge: dict[str, Any]) -> dict[str, Any]:
        """Extract edge configuration from edge data."""
        data = edge.get("data", {})
        if isinstance(data, dict):
            return data
        return {}

    def _build_node_map(self) -> dict[str, dict[str, Any]]:
        """Build a map of node ID to node data."""
        return {node.get("id", ""): node for node in self.nodes if node.get("id")}

    def _get_agent_nodes(self) -> list[dict[str, Any]]:
        """Get all agent nodes from the pipeline."""
        return [n for n in self.nodes if n.get("type") == "agent"]

    def _get_team_nodes(self) -> list[dict[str, Any]]:
        """Get all team nodes from the pipeline."""
        return [n for n in self.nodes if n.get("type") == "team"]

    def _find_start_node(self) -> str | None:
        """Find the start node ID."""
        for node in self.nodes:
            if node.get("type") == "start":
                node_id = node.get("id")
                return str(node_id) if node_id else None
        return None

    def _find_end_node(self) -> str | None:
        """Find the end node ID."""
        for node in self.nodes:
            if node.get("type") == "end":
                node_id = node.get("id")
                return str(node_id) if node_id else None
        return None

    def _find_first_node_after_start(self) -> str | None:
        """Find the first executable node connected to start."""
        start_id = self._find_start_node()
        if not start_id:
            return None

        for edge in self.edges:
            if edge.get("source") == start_id:
                target_id: str = edge.get("target", "") or ""
                target_node = self._node_map.get(target_id, {})
                if target_node.get("type") in ("agent", "team"):
                    return target_id
        return None

    def _group_edges_by_source(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for edge in self.edges:
            source_id = edge.get("source", "")
            if source_id:
                if source_id not in result:
                    result[source_id] = []
                result[source_id].append(edge)
        return result

    def _group_edges_by_resolved_source(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for edge in self.edges:
            resolved_source = self._resolve_edge_source(edge)
            if resolved_source:
                if resolved_source not in result:
                    result[resolved_source] = []
                result[resolved_source].append(edge)
        return result

    def _extract_agent_name(self, node: dict[str, Any]) -> str | None:
        """Extract the agent name from a node."""
        data = node.get("data", {})
        if isinstance(data, dict):
            return data.get("agentName") or data.get("name") or data.get("label")
        return None


def build_pipeline_workflow(
    pipeline: Pipeline,
    use_persistence: bool = False,
    strict: bool = True,
    auto_connect_dangling: bool = True,
) -> CompiledStateGraph[Any]:
    """Build a workflow from a pipeline configuration.

    Args:
        pipeline: The pipeline configuration to build.
        use_persistence: Whether to use persistent checkpointing.
        strict: If True, raises PipelineValidationError on validation failures.
        auto_connect_dangling: If True, nodes without outgoing edges are auto-connected to END.

    Raises:
        PipelineValidationError: If strict=True and pipeline is invalid.
    """
    builder = PipelineBuilder(pipeline)
    return builder.build(
        use_persistence=use_persistence,
        strict=strict,
        auto_connect_dangling=auto_connect_dangling,
    )


async def build_pipeline_workflow_async(
    pipeline: Pipeline,
    use_persistence: bool = False,
    strict: bool = True,
    auto_connect_dangling: bool = True,
) -> CompiledStateGraph[Any]:
    """Build a workflow from a pipeline configuration (async, supports tool nodes).

    Args:
        pipeline: The pipeline configuration to build.
        use_persistence: Whether to use persistent checkpointing.
        strict: If True, raises PipelineValidationError on validation failures.
        auto_connect_dangling: If True, nodes without outgoing edges are auto-connected to END.

    Raises:
        PipelineValidationError: If strict=True and pipeline is invalid.
    """
    builder = PipelineBuilder(pipeline)
    return await builder.build_async(
        use_persistence=use_persistence,
        strict=strict,
        auto_connect_dangling=auto_connect_dangling,
    )


async def get_pipeline_workflow(pipeline_id: str | None = None) -> CompiledStateGraph[Any] | None:
    """Get a compiled workflow for a pipeline.

    If pipeline_id is None, returns the default pipeline workflow.
    Returns None if no pipeline found.

    Uses async build to support tool nodes with MCP servers.
    """
    from src.storage import get_repository

    repo = await get_repository()

    if pipeline_id:
        pipeline = await repo.get_pipeline(pipeline_id)
    else:
        pipeline = await repo.get_default_pipeline()

    if not pipeline:
        return None

    return await build_pipeline_workflow_async(pipeline)
