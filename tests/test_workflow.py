from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent, ToolExecutionError, detect_agent_cycle
from src.types import AgentState


class TestWorkflowCaching:
    def test_clear_workflow_cache(self) -> None:
        from src.graph.workflow import clear_workflow_cache, get_cached_workflow_count

        clear_workflow_cache()
        assert get_cached_workflow_count() == 0

    def test_make_cache_key_deterministic(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, "gpt-4", None, None)
        key2 = _make_cache_key(True, "gpt-4", None, None)
        assert key1 == key2

    def test_make_cache_key_different_for_different_configs(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, "gpt-4", None, None)
        key2 = _make_cache_key(True, "gpt-4o", None, None)
        assert key1 != key2

    def test_make_cache_key_different_for_persistence(self) -> None:
        from src.graph.workflow import _make_cache_key

        key1 = _make_cache_key(True, "gpt-4", None, None)
        key2 = _make_cache_key(False, "gpt-4", None, None)
        assert key1 != key2


class TestAgentCycleDetectionEdgeCases:
    def test_empty_history(self) -> None:
        assert detect_agent_cycle([]) is False

    def test_single_agent(self) -> None:
        assert detect_agent_cycle(["coder"]) is False

    def test_two_agents_no_cycle(self) -> None:
        assert detect_agent_cycle(["coder", "writer"]) is False

    def test_three_agents_no_cycle(self) -> None:
        assert detect_agent_cycle(["coder", "writer", "researcher"]) is False

    def test_exact_two_cycle(self) -> None:
        assert detect_agent_cycle(["a", "b", "a", "b"]) is True

    def test_exact_three_cycle(self) -> None:
        assert detect_agent_cycle(["a", "b", "c", "a", "b", "c"]) is True

    def test_exact_four_cycle(self) -> None:
        assert detect_agent_cycle(["a", "b", "c", "d", "a", "b", "c", "d"]) is True

    def test_partial_cycle_not_detected(self) -> None:
        assert detect_agent_cycle(["a", "b", "a"]) is False

    def test_long_history_with_cycle_at_end(self) -> None:
        history = ["x", "y", "z", "a", "b", "a", "b"]
        assert detect_agent_cycle(history) is True

    def test_long_history_without_cycle(self) -> None:
        history = ["a", "b", "c", "d", "e", "f", "g"]
        assert detect_agent_cycle(history) is False

    def test_same_agent_twice_not_cycle(self) -> None:
        assert detect_agent_cycle(["coder", "coder"]) is False

    def test_same_agent_four_times_is_cycle(self) -> None:
        assert detect_agent_cycle(["coder", "coder", "coder", "coder"]) is True


class TestToolExecutionError:
    def test_error_message(self) -> None:
        original = ValueError("Something went wrong")
        error = ToolExecutionError("search_tool", original)
        assert "search_tool" in str(error)
        assert "Something went wrong" in str(error)

    def test_error_attributes(self) -> None:
        original = RuntimeError("Timeout")
        error = ToolExecutionError("execute_code", original)
        assert error.tool_name == "execute_code"
        assert error.original_error is original


class TestAgentStateStructure:
    def test_minimal_state(self) -> None:
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "current_agent": "user",
            "context": {},
            "human_feedback": None,
            "iteration_count": 0,
            "execution_trace": [],
        }
        assert len(state["messages"]) == 1
        assert state["iteration_count"] == 0

    def test_state_with_context(self) -> None:
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "current_agent": "coder",
            "context": {"task_id": "abc123", "priority": "high"},
            "human_feedback": "looks good",
            "iteration_count": 5,
            "execution_trace": ["supervisor", "coder", "writer"],
        }
        assert state["context"]["task_id"] == "abc123"
        assert len(state["execution_trace"]) == 3


class TestBaseAgentBehavior:
    def test_create_response_message_with_name(self) -> None:
        with patch.object(BaseAgent, "__init__", lambda x, *args, **kwargs: None):
            agent = BaseAgent.__new__(BaseAgent)
            agent.name = "test_agent"
            msg = agent.create_response_message("Hello world")
            assert msg.content == "Hello world"
            assert msg.name == "test_agent"

    def test_get_messages_with_system_filters_supervisor(self) -> None:
        with patch.object(BaseAgent, "__init__", lambda x, *args, **kwargs: None):
            agent = BaseAgent.__new__(BaseAgent)
            agent.name = "coder"
            agent.system_prompt = "You are a coder"

            state: AgentState = {
                "messages": [
                    HumanMessage(content="Write code"),
                    AIMessage(content="[Supervisor] Routing to coder"),
                    AIMessage(content="Here is the code", name="coder"),
                ],
                "current_agent": "coder",
                "context": {},
                "human_feedback": None,
                "iteration_count": 0,
                "execution_trace": [],
            }

            messages = agent.get_messages_with_system(state)
            contents = [m.content for m in messages]
            assert "[Supervisor]" not in str(contents)
            assert "Write code" in str(contents)


class TestMaxIterationsHandling:
    @pytest.mark.asyncio
    async def test_max_iterations_returns_end_command(self) -> None:
        from langgraph.graph import END

        from src.config import settings

        with patch.object(BaseAgent, "__init__", lambda x, *args, **kwargs: None):
            agent = BaseAgent.__new__(BaseAgent)
            agent.name = "test_agent"

            state: AgentState = {
                "messages": [HumanMessage(content="Test")],
                "current_agent": "test_agent",
                "context": {},
                "human_feedback": None,
                "iteration_count": settings.max_iterations,
                "execution_trace": [],
            }

            with patch.object(agent, "process", new_callable=AsyncMock) as mock_process:
                result = await agent.invoke(state)

                mock_process.assert_not_called()
                assert result.goto == END


class TestHandoffHistoryTracking:
    @pytest.mark.asyncio
    async def test_execution_trace_appended(self) -> None:
        with patch.object(BaseAgent, "__init__", lambda x, *args, **kwargs: None):
            agent = BaseAgent.__new__(BaseAgent)
            agent.name = "coder"
            agent.system_prompt = "You are a coder"
            agent.tools = []

            state: AgentState = {
                "messages": [HumanMessage(content="Test")],
                "current_agent": "user",
                "context": {},
                "human_feedback": None,
                "iteration_count": 0,
                "execution_trace": ["supervisor"],
            }

            with patch.object(agent, "process", new_callable=AsyncMock) as mock_process:
                mock_process.return_value = {"messages": [AIMessage(content="Done")]}

                result = await agent.invoke(state)

                assert "coder" in result["execution_trace"]
                assert result["iteration_count"] == 1


class TestChunkingStrategies:
    def test_auto_detection_python_file(self) -> None:
        from src.knowledge.chunking import ChunkingStrategy, _detect_content_type

        result = _detect_content_type("def foo():\n    pass", "test.py")
        assert result == ChunkingStrategy.CODE

    def test_auto_detection_markdown_file(self) -> None:
        from src.knowledge.chunking import ChunkingStrategy, _detect_content_type

        result = _detect_content_type("# Title\n\nContent", "readme.md")
        assert result == ChunkingStrategy.MARKDOWN

    def test_auto_detection_by_content_code(self) -> None:
        from src.knowledge.chunking import ChunkingStrategy, _detect_content_type

        code_content = "\n".join(
            [
                "import os",
                "import sys",
                "import json",
                "def main():",
                "    pass",
                "def helper():",
                "    pass",
                "class Foo:",
                "    def bar(self):",
                "        return 42",
            ]
        )
        result = _detect_content_type(code_content, "unknown.txt")
        assert result == ChunkingStrategy.CODE

    def test_auto_detection_by_content_markdown(self) -> None:
        from src.knowledge.chunking import ChunkingStrategy, _detect_content_type

        md_content = "\n".join(
            [
                "# Heading 1",
                "## Heading 2",
                "- Item 1",
                "- Item 2",
                "* Bullet",
                "> Quote",
            ]
        )
        result = _detect_content_type(md_content, "unknown.txt")
        assert result == ChunkingStrategy.MARKDOWN

    def test_auto_detection_fallback_recursive(self) -> None:
        from src.knowledge.chunking import ChunkingStrategy, _detect_content_type

        plain_text = "This is just plain text without any special formatting."
        result = _detect_content_type(plain_text, "plain.txt")
        assert result == ChunkingStrategy.RECURSIVE

    def test_estimate_chunks(self) -> None:
        from src.knowledge.chunking import ChunkConfig, estimate_chunks

        config = ChunkConfig(chunk_size=100, chunk_overlap=20)
        content = "x" * 500
        estimate = estimate_chunks(content, config)
        assert estimate > 0
        assert estimate == 500 // (100 - 20)
