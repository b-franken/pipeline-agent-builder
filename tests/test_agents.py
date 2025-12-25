"""Tests for agent functionality."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.types import AgentState
from tests.fixtures import (
    MockLLM,
    MockToolCallLLM,
    assert_message_contains,
    create_conversation,
)


class TestMockLLM:
    """Tests for the MockLLM testing utility."""

    def test_returns_responses_in_order(self):
        llm = MockLLM(responses=["First", "Second", "Third"])
        result1 = llm.invoke([HumanMessage(content="Hi")])
        result2 = llm.invoke([HumanMessage(content="Hello")])
        assert result1.content == "First"
        assert result2.content == "Second"

    def test_cycles_through_responses(self):
        llm = MockLLM(responses=["A", "B"])
        llm.invoke([HumanMessage(content="1")])
        llm.invoke([HumanMessage(content="2")])
        result = llm.invoke([HumanMessage(content="3")])
        assert result.content == "A"  # Cycles back

    def test_captures_prompts(self):
        llm = MockLLM(responses=["Response"])
        llm.invoke([HumanMessage(content="Test prompt")])
        assert llm.call_count == 1
        assert "Test prompt" in llm.get_last_prompt_text()

    def test_reset_clears_history(self):
        llm = MockLLM(responses=["A", "B"])
        llm.invoke([HumanMessage(content="1")])
        llm.reset()
        assert llm.call_count == 0
        assert llm.prompts == []


class TestMockToolCallLLM:
    """Tests for the MockToolCallLLM testing utility."""

    def test_returns_tool_calls_first(self):
        llm = MockToolCallLLM(
            tool_calls=[{"name": "search", "args": {"query": "test"}}],
            responses=["Done"],
        )
        result = llm.invoke([HumanMessage(content="Search")])
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_returns_text_after_tool_calls(self):
        llm = MockToolCallLLM(
            tool_calls=[{"name": "search", "args": {}}],
            responses=["Search complete"],
        )
        llm.invoke([HumanMessage(content="1")])  # First: tool calls
        result = llm.invoke([HumanMessage(content="2")])  # Second: text
        assert result.content == "Search complete"


class TestHelperFunctions:
    """Tests for test helper functions."""

    def test_create_conversation(self):
        messages = create_conversation(
            ("human", "Hello"),
            ("ai", "Hi there!"),
            ("human", "How are you?"),
        )
        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Hello"

    def test_assert_message_contains(self):
        messages = [
            HumanMessage(content="Hello world"),
            AIMessage(content="Hi there"),
        ]
        assert_message_contains(messages, "Hello")
        assert_message_contains(messages, "there")

        with pytest.raises(AssertionError):
            assert_message_contains(messages, "xyz")


class TestBaseAgent:
    def test_create_response_message(self):
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def process(self, state: AgentState) -> dict:
                return {"messages": [self.create_response_message("Test")]}

        with patch.object(BaseAgent, "__init__", lambda x, *args, **kwargs: None):
            agent = TestAgent.__new__(TestAgent)
            agent.name = "test_agent"
            agent.system_prompt = "Test"
            agent.tools = []

            msg = agent.create_response_message("Hello")
            assert msg.name == "test_agent"
            assert msg.content == "Hello"


class TestAgentCycleDetection:
    def test_detect_two_agent_cycle(self):
        from src.agents.base import detect_agent_cycle

        history = ["coder", "writer", "coder", "writer"]
        assert detect_agent_cycle(history) is True

    def test_detect_three_agent_cycle(self):
        from src.agents.base import detect_agent_cycle

        history = ["a", "b", "c", "a", "b", "c"]
        assert detect_agent_cycle(history) is True

    def test_detect_four_agent_cycle(self):
        from src.agents.base import detect_agent_cycle

        history = ["a", "b", "c", "d", "a", "b", "c", "d"]
        assert detect_agent_cycle(history) is True

    def test_no_cycle_different_agents(self):
        from src.agents.base import detect_agent_cycle

        history = ["coder", "writer", "researcher"]
        assert detect_agent_cycle(history) is False

    def test_no_cycle_short_history(self):
        from src.agents.base import detect_agent_cycle

        history = ["coder", "writer"]
        assert detect_agent_cycle(history) is False

    def test_no_cycle_incomplete_pattern(self):
        from src.agents.base import detect_agent_cycle

        history = ["a", "b", "c", "a", "b"]
        assert detect_agent_cycle(history) is False

    def test_same_agent_repeated_is_cycle(self):
        from src.agents.base import detect_agent_cycle

        history = ["coder", "coder", "coder", "coder"]
        assert detect_agent_cycle(history) is True

    def test_no_cycle_three_same_agents(self):
        from src.agents.base import detect_agent_cycle

        history = ["coder", "coder", "coder"]
        assert detect_agent_cycle(history) is False


class TestSupervisorAgent:
    def test_supervisor_initialization(self):
        with patch("src.llm.create_llm_with_structured_output") as mock_llm:
            mock_llm.return_value = AsyncMock()
            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent()
            assert supervisor.name == "supervisor"


class TestAgentState:
    def test_state_structure(self, sample_state: AgentState):
        required = [
            "messages",
            "current_agent",
            "context",
            "human_feedback",
            "iteration_count",
            "execution_trace",
        ]
        for field in required:
            assert field in sample_state


class TestWorkflow:
    def test_create_simple_workflow(self):
        with (
            patch("src.graph.workflow.SupervisorAgent"),
            patch("src.graph.workflow.get_registry") as mock_registry,
            patch("src.graph.workflow.create_checkpointer"),
        ):
            mock_registry.return_value.get_agent_names.return_value = []
            from src.graph.workflow import create_simple_workflow

            workflow = create_simple_workflow()
            assert workflow is not None
