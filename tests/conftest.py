"""Pytest configuration and fixtures.

Fixtures defined here are automatically available to all tests.
"""

from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from src.types import AgentState
from tests.fixtures import MockLLM, MockStreamingLLM, MockToolCallLLM


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a basic mock LLM."""
    return MockLLM(responses=["Mock response"])


@pytest.fixture
def mock_tool_llm() -> MockToolCallLLM:
    """Create a mock LLM that returns tool calls."""
    return MockToolCallLLM()


@pytest.fixture
def mock_streaming_llm() -> MockStreamingLLM:
    """Create a mock LLM that supports streaming."""
    return MockStreamingLLM(responses=["Streamed response"])


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state for testing."""
    return {
        "messages": [HumanMessage(content="Test message")],
        "current_agent": "user",
        "context": {},
        "human_feedback": None,
        "iteration_count": 0,
        "execution_trace": [],
    }


@pytest.fixture
def state_factory():
    """Factory for creating custom agent states."""

    def _create(
        messages: list[BaseMessage] | None = None,
        current_agent: str = "user",
        context: dict[str, Any] | None = None,
        human_feedback: str | None = None,
        iteration_count: int = 0,
        execution_trace: list[str] | None = None,
    ) -> AgentState:
        return {
            "messages": messages or [HumanMessage(content="Test")],
            "current_agent": current_agent,
            "context": context or {},
            "human_feedback": human_feedback,
            "iteration_count": iteration_count,
            "execution_trace": execution_trace or [],
        }

    return _create
