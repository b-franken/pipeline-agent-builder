from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.types import AgentState


class MockLLM(BaseChatModel):
    responses: list[str] = Field(default_factory=lambda: ["Mock response"])
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    call_count: int = 0
    _response_index: int = 0

    def __init__(self, responses: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if responses is not None:
            self.responses = responses
        self.prompts = []
        self.call_count = 0
        self._response_index = 0

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(messages)
        self.call_count += 1

        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def reset(self) -> None:
        self.prompts = []
        self.call_count = 0
        self._response_index = 0

    def get_last_prompt(self) -> list[BaseMessage] | None:
        return self.prompts[-1] if self.prompts else None

    def get_last_prompt_text(self) -> str:
        if not self.prompts:
            return ""
        return "\n".join(str(m.content) for m in self.prompts[-1])


class MockToolCallLLM(MockLLM):
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    def __init__(
        self,
        tool_calls: list[dict[str, Any]] | None = None,
        responses: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if responses is None:
            responses = ["Tool result processed"]
        super().__init__(responses=responses, **kwargs)
        if tool_calls is not None:
            self.tool_calls = tool_calls
        else:
            self.tool_calls = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(messages)
        self.call_count += 1

        if self.call_count == 1 and self.tool_calls:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{i}",
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                    }
                    for i, tc in enumerate(self.tool_calls)
                ],
            )
        else:
            response = self.responses[(self._response_index - 1) % len(self.responses)]
            self._response_index += 1
            message = AIMessage(content=response)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


class MockStreamingLLM(MockLLM):
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGeneration]:
        self.prompts.append(messages)
        self.call_count += 1

        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1

        for char in response:
            yield ChatGeneration(message=AIMessage(content=char))


def assert_message_contains(messages: list[BaseMessage], content: str) -> None:
    for msg in messages:
        if content in str(msg.content):
            return
    raise AssertionError(f"No message contains '{content}'")


def assert_agent_responded(state: AgentState, agent_name: str) -> None:
    for msg in state["messages"]:
        if hasattr(msg, "name") and msg.name == agent_name:
            return
    raise AssertionError(f"Agent '{agent_name}' did not respond")


def create_conversation(
    *messages: tuple[str, str],
) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for role, content in messages:
        if role in ("human", "user"):
            result.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            result.append(AIMessage(content=content))
    return result
