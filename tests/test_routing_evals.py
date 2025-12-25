from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.types import AgentState


class MockRouteDecision(BaseModel):
    next_agent: str
    reasoning: str


def create_mock_llm(next_agent: str, reasoning: str = "Test reasoning") -> tuple[AsyncMock, str]:
    mock = AsyncMock()
    mock.ainvoke.return_value = MockRouteDecision(next_agent=next_agent, reasoning=reasoning)
    return (mock, "mock-model")


def create_test_state(user_message: str, execution_trace: list[str] | None = None) -> AgentState:
    """Create a test state with a user message."""
    return {
        "messages": [HumanMessage(content=user_message)],
        "current_agent": "user",
        "context": {},
        "human_feedback": None,
        "iteration_count": 0,
        "execution_trace": execution_trace or [],
        "loop_counts": {},
    }


class TestRoutingDecisions:
    """Eval tests for routing decisions."""

    @pytest.mark.asyncio
    async def test_routes_code_task_to_coder(self) -> None:
        """Code-related tasks should route to coder agent."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder", "researcher", "writer"]
            mock_create.return_value = create_mock_llm("coder")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Write a Python function to sort a list")

            result = await supervisor.route(state)

            assert result.goto == "coder"

    @pytest.mark.asyncio
    async def test_routes_research_task_to_researcher(self) -> None:
        """Research tasks should route to researcher agent."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder", "researcher", "writer"]
            mock_create.return_value = create_mock_llm("researcher")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Find information about quantum computing")

            result = await supervisor.route(state)

            assert result.goto == "researcher"

    @pytest.mark.asyncio
    async def test_routes_writing_task_to_writer(self) -> None:
        """Writing tasks should route to writer agent."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder", "researcher", "writer"]
            mock_create.return_value = create_mock_llm("writer")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Write a blog post about AI trends")

            result = await supervisor.route(state)

            assert result.goto == "writer"

    @pytest.mark.asyncio
    async def test_finish_routes_to_end(self) -> None:
        """FINISH decision should route to END."""
        from langgraph.graph import END

        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder", "researcher", "writer"]
            mock_create.return_value = create_mock_llm("FINISH")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Thank you, that's all I need")

            result = await supervisor.route(state)

            assert result.goto == END


class TestInvalidRoutingHandling:
    """Test handling of invalid routing decisions."""

    @pytest.mark.asyncio
    async def test_invalid_agent_defaults_to_finish(self) -> None:
        """Invalid agent names should default to FINISH/END."""
        from langgraph.graph import END

        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder", "researcher"]
            mock_create.return_value = create_mock_llm("nonexistent_agent")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Do something")

            result = await supervisor.route(state)

            assert result.goto == END


class TestEscalationHandling:
    """Test escalation when agents are in a loop."""

    @pytest.mark.asyncio
    async def test_escalation_includes_loop_context(self) -> None:
        """Escalation should include context about the agent loop."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder", "researcher"]
            mock_llm, _ = create_mock_llm("coder")
            mock_create.return_value = (mock_llm, "mock-model")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state(
                "Complex task that caused a loop",
                execution_trace=["coder", "researcher", "coder", "researcher"],
            )

            await supervisor.route(state)

            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][0]
            system_message = messages[0].content

            assert "escalation" in system_message.lower()
            assert "loop" in system_message.lower()

    @pytest.mark.asyncio
    async def test_escalation_routes_to_different_agent(self) -> None:
        """Escalation should attempt to route to a different agent."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder", "researcher", "writer"]
            mock_create.return_value = create_mock_llm("writer")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state(
                "Task that was stuck between coder and researcher",
                execution_trace=["coder", "researcher", "coder", "researcher"],
            )

            result = await supervisor.route(state)

            assert result.goto == "writer"


class TestStateUpdates:
    """Test that routing updates state correctly."""

    @pytest.mark.asyncio
    async def test_routing_increments_iteration_count(self) -> None:
        """Routing should increment the iteration count."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder"]
            mock_create.return_value = create_mock_llm("coder")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Write code")
            state["iteration_count"] = 5

            result = await supervisor.route(state)

            assert result.update is not None
            assert result.update["iteration_count"] == 6

    @pytest.mark.asyncio
    async def test_routing_appends_to_execution_trace(self) -> None:
        """Routing should append supervisor to execution trace for loop detection."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder"]
            mock_create.return_value = create_mock_llm("coder")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Write code", execution_trace=["old", "trace"])

            result = await supervisor.route(state)

            assert result.update is not None
            assert result.update["execution_trace"] == ["old", "trace", "supervisor"]

    @pytest.mark.asyncio
    async def test_routing_adds_supervisor_message(self) -> None:
        """Routing should add a supervisor message explaining the decision."""
        with (
            patch("src.agents.supervisor.create_llm_with_structured_output") as mock_create,
            patch("src.agents.supervisor.get_registry") as mock_registry,
        ):
            mock_registry.return_value.get_routable_agents.return_value = ["coder"]
            mock_create.return_value = create_mock_llm("coder", "Task requires coding skills")

            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            state = create_test_state("Write a function")

            result = await supervisor.route(state)

            assert result.update is not None
            messages = result.update.get("messages", [])
            assert len(messages) == 1
            assert "[Supervisor]" in messages[0].content
            assert "coder" in messages[0].content
