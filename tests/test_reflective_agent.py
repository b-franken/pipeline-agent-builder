from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.reflective import (
    DEFAULT_CRITIQUE_PROMPT,
    DEFAULT_MAX_CONTEXT_MESSAGES,
)


@pytest.fixture
def mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.get_handoff_tools.return_value = []
    return registry


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def reflective_agent(mock_llm: MagicMock, mock_registry: MagicMock) -> Any:
    with (
        patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
        patch("src.agents.base.get_registry", return_value=mock_registry),
    ):
        from src.agents.reflective import ReflectiveAgent

        agent = ReflectiveAgent(
            name="test_agent",
            system_prompt="You are a test agent",
        )
        return agent


@pytest.fixture
def custom_reflective_agent(mock_llm: MagicMock, mock_registry: MagicMock) -> Any:
    with (
        patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
        patch("src.agents.base.get_registry", return_value=mock_registry),
    ):
        from src.agents.reflective import ReflectiveAgent

        agent = ReflectiveAgent(
            name="custom_agent",
            system_prompt="Custom prompt",
            max_refinements=3,
            critique_prompt="Custom critique",
            max_context_messages=10,
        )
        return agent


class TestReflectiveAgentInit:
    def test_default_values(self, reflective_agent: Any) -> None:
        assert reflective_agent.name == "test_agent"
        assert reflective_agent.system_prompt == "You are a test agent"
        assert reflective_agent.max_refinements == 1
        assert reflective_agent.critique_prompt == DEFAULT_CRITIQUE_PROMPT
        assert reflective_agent.max_context_messages == DEFAULT_MAX_CONTEXT_MESSAGES

    def test_custom_values(self, custom_reflective_agent: Any) -> None:
        assert custom_reflective_agent.max_refinements == 3
        assert custom_reflective_agent.critique_prompt == "Custom critique"
        assert custom_reflective_agent.max_context_messages == 10


class TestEnsureValidMessageSequence:
    def test_empty_messages(self, reflective_agent: Any) -> None:
        result = reflective_agent._ensure_valid_message_sequence([])
        assert result == []

    def test_simple_messages_no_tools(self, reflective_agent: Any) -> None:
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        result = reflective_agent._ensure_valid_message_sequence(messages)
        assert len(result) == 2

    def test_filters_incomplete_tool_pairs(self, reflective_agent: Any) -> None:
        ai_with_tools = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool", "args": {}}])
        orphan_tool_message = ToolMessage(content="result", tool_call_id="call_2")

        messages = [
            HumanMessage(content="Hello"),
            ai_with_tools,
            orphan_tool_message,
        ]

        result = reflective_agent._ensure_valid_message_sequence(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_keeps_complete_tool_pairs(self, reflective_agent: Any) -> None:
        ai_with_tools = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool", "args": {}}])
        tool_response = ToolMessage(content="result", tool_call_id="call_1")

        messages = [
            HumanMessage(content="Hello"),
            ai_with_tools,
            tool_response,
        ]

        result = reflective_agent._ensure_valid_message_sequence(messages)

        assert len(result) == 3


class TestGetTruncatedContext:
    def test_short_context_unchanged(self, mock_llm: MagicMock, mock_registry: MagicMock) -> None:
        with (
            patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
            patch("src.agents.base.get_registry", return_value=mock_registry),
        ):
            from src.agents.reflective import ReflectiveAgent

            agent = ReflectiveAgent(name="test", system_prompt="test", max_context_messages=6)
            messages = [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
            ]
            state: dict[str, Any] = {"messages": messages}

            result = agent._get_truncated_context(state)

            assert len(result) == 2

    def test_long_context_truncated(self, mock_llm: MagicMock, mock_registry: MagicMock) -> None:
        with (
            patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
            patch("src.agents.base.get_registry", return_value=mock_registry),
        ):
            from src.agents.reflective import ReflectiveAgent

            agent = ReflectiveAgent(name="test", system_prompt="test", max_context_messages=3)
            messages = [
                HumanMessage(content="First human message"),
                AIMessage(content="Response 1"),
                HumanMessage(content="Second"),
                AIMessage(content="Response 2"),
                HumanMessage(content="Third"),
                AIMessage(content="Response 3"),
            ]
            state: dict[str, Any] = {"messages": messages}

            result = agent._get_truncated_context(state)

            assert len(result) <= 3

    def test_preserves_first_human_message(self, mock_llm: MagicMock, mock_registry: MagicMock) -> None:
        with (
            patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
            patch("src.agents.base.get_registry", return_value=mock_registry),
        ):
            from src.agents.reflective import ReflectiveAgent

            agent = ReflectiveAgent(name="test", system_prompt="test", max_context_messages=2)
            messages = [
                HumanMessage(content="Original question"),
                AIMessage(content="Response 1"),
                AIMessage(content="Response 2"),
                AIMessage(content="Response 3"),
            ]
            state: dict[str, Any] = {"messages": messages}

            result = agent._get_truncated_context(state)

            has_original = any(isinstance(m, HumanMessage) and m.content == "Original question" for m in result)
            assert has_original


class TestCritique:
    @pytest.mark.asyncio
    async def test_critique_returns_string(self, reflective_agent: Any) -> None:
        response = AIMessage(content="Test response")
        state: dict[str, Any] = {"messages": [HumanMessage(content="Question")]}

        with patch.object(reflective_agent, "_invoke_llm_with_retry", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = AIMessage(content="This looks good. APPROVED")

            result = await reflective_agent._critique(response, state)

            assert "APPROVED" in result
            mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_with_empty_messages(self, reflective_agent: Any) -> None:
        response = AIMessage(content="Test response")
        state: dict[str, Any] = {"messages": []}

        with patch.object(reflective_agent, "_invoke_llm_with_retry", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = AIMessage(content="Needs work")

            result = await reflective_agent._critique(response, state)

            assert isinstance(result, str)


class TestRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_ai_message(self, reflective_agent: Any) -> None:
        response = AIMessage(content="Original response")
        critique = "Add more details"
        state: dict[str, Any] = {"messages": [HumanMessage(content="Question")]}

        refined_response = AIMessage(content="Improved response with details")

        with patch.object(reflective_agent, "_invoke_llm_with_retry", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = refined_response

            result = await reflective_agent._refine(response, critique, state)

            assert isinstance(result, AIMessage)
            assert result.content == "Improved response with details"


class TestProcess:
    @pytest.mark.asyncio
    async def test_process_approved_immediately(self, mock_llm: MagicMock, mock_registry: MagicMock) -> None:
        with (
            patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
            patch("src.agents.base.get_registry", return_value=mock_registry),
        ):
            from src.agents.reflective import ReflectiveAgent

            agent = ReflectiveAgent(name="test", system_prompt="test", max_refinements=2)
            state: dict[str, Any] = {"messages": [HumanMessage(content="Hello")]}

            initial_response = AIMessage(content="Hello there!")

            with (
                patch.object(agent, "get_messages_with_system", return_value=[]),
                patch.object(agent, "_invoke_llm_with_retry", new_callable=AsyncMock) as mock_invoke,
                patch.object(agent, "handle_tool_calls", new_callable=AsyncMock) as mock_handle_tools,
                patch.object(agent, "_critique", new_callable=AsyncMock) as mock_critique,
            ):
                mock_invoke.return_value = initial_response
                mock_handle_tools.return_value = (initial_response, [])
                mock_critique.return_value = "APPROVED - looks great!"

                result = await agent.process(state)

                assert isinstance(result, dict)
                assert "messages" in result
                mock_critique.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_refinement(self, mock_llm: MagicMock, mock_registry: MagicMock) -> None:
        with (
            patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
            patch("src.agents.base.get_registry", return_value=mock_registry),
        ):
            from src.agents.reflective import ReflectiveAgent

            agent = ReflectiveAgent(name="test", system_prompt="test", max_refinements=2)
            state: dict[str, Any] = {"messages": [HumanMessage(content="Hello")]}

            initial_response = AIMessage(content="Initial")
            refined_response = AIMessage(content="Refined")

            with (
                patch.object(agent, "get_messages_with_system", return_value=[]),
                patch.object(agent, "_invoke_llm_with_retry", new_callable=AsyncMock) as mock_invoke,
                patch.object(agent, "handle_tool_calls", new_callable=AsyncMock) as mock_handle_tools,
                patch.object(agent, "_critique", new_callable=AsyncMock) as mock_critique,
                patch.object(agent, "_refine", new_callable=AsyncMock) as mock_refine,
            ):
                mock_invoke.return_value = initial_response
                mock_handle_tools.return_value = (initial_response, [])
                mock_critique.side_effect = ["Needs improvement", "APPROVED"]
                mock_refine.return_value = refined_response

                result = await agent.process(state)

                assert isinstance(result, dict)
                mock_refine.assert_called_once()
                assert mock_critique.call_count == 2

    @pytest.mark.asyncio
    async def test_process_with_tool_calls_returns_command(self, mock_llm: MagicMock, mock_registry: MagicMock) -> None:
        from langgraph.types import Command

        with (
            patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
            patch("src.agents.base.get_registry", return_value=mock_registry),
        ):
            from src.agents.reflective import ReflectiveAgent

            agent = ReflectiveAgent(name="test", system_prompt="test")
            state: dict[str, Any] = {"messages": [HumanMessage(content="Hello")]}

            initial_response = AIMessage(content="", tool_calls=[{"id": "1", "name": "tool", "args": {}}])
            command_result = Command(goto="next_agent", update={})

            with (
                patch.object(agent, "get_messages_with_system", return_value=[]),
                patch.object(agent, "_invoke_llm_with_retry", new_callable=AsyncMock) as mock_invoke,
                patch.object(agent, "handle_tool_calls", new_callable=AsyncMock) as mock_handle_tools,
            ):
                mock_invoke.return_value = initial_response
                mock_handle_tools.return_value = (command_result, [ToolMessage(content="result", tool_call_id="1")])

                result = await agent.process(state)

                assert isinstance(result, Command)

    @pytest.mark.asyncio
    async def test_process_non_ai_message_raises(self, mock_llm: MagicMock, mock_registry: MagicMock) -> None:
        with (
            patch("src.agents.base.create_llm", return_value=(mock_llm, "test-model")),
            patch("src.agents.base.get_registry", return_value=mock_registry),
        ):
            from src.agents.reflective import ReflectiveAgent

            agent = ReflectiveAgent(name="test", system_prompt="test")
            state: dict[str, Any] = {"messages": [HumanMessage(content="Hello")]}

            with (
                patch.object(agent, "get_messages_with_system", return_value=[]),
                patch.object(agent, "_invoke_llm_with_retry", new_callable=AsyncMock) as mock_invoke,
            ):
                mock_invoke.return_value = "not an AIMessage"

                with pytest.raises(TypeError, match="Expected AIMessage"):
                    await agent.process(state)
