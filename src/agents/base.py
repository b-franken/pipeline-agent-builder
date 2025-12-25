import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any, Final

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.types import Command
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.config import settings
from src.llm import Provider, create_llm
from src.registry import get_registry
from src.trace import get_tracer
from src.types import AgentState

logger: Final = logging.getLogger(__name__)

_RETRYABLE_ERRORS: Final = frozenset(
    {
        "RateLimitError",
        "RateLimit",
        "APIStatusError",
        "APIConnectionError",
        "APITimeoutError",
        "ServiceUnavailableError",
        "InternalServerError",
        "OverloadedError",
        "Timeout",
        "ConnectionError",
        "OSError",
    }
)

_FATAL_ERRORS: Final = frozenset(
    {
        "AuthenticationError",
        "InvalidRequestError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
    }
)


def _is_retryable_error(exception: BaseException) -> bool:
    error_name = type(exception).__name__

    if any(fatal in error_name for fatal in _FATAL_ERRORS):
        return False

    if any(retryable in error_name for retryable in _RETRYABLE_ERRORS):
        return True

    status_code = getattr(exception, "status_code", None)
    return status_code is not None and status_code in (429, 500, 502, 503, 504)


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    if retry_state.attempt_number > 1:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            f"LLM call retry attempt {retry_state.attempt_number}/6 "
            f"after {type(exception).__name__ if exception else 'unknown error'}"
        )


MAX_PING_PONG_HANDOFFS: Final = 3
MIN_CYCLE_LENGTH: Final = 2
MAX_CYCLE_LENGTH: Final = 4


def detect_agent_cycle(
    execution_trace: list[str],
    min_cycle_len: int = MIN_CYCLE_LENGTH,
    max_cycle_len: int = MAX_CYCLE_LENGTH,
) -> bool:
    history_len = len(execution_trace)

    for cycle_len in range(min_cycle_len, max_cycle_len + 1):
        required_len = cycle_len * 2
        if history_len < required_len:
            continue

        recent = execution_trace[-required_len:]
        first_half = recent[:cycle_len]
        second_half = recent[cycle_len:]

        if first_half == second_half:
            return True

    return False


class ToolExecutionError(Exception):
    def __init__(self, tool_name: str, original_error: Exception) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {original_error}")


class BaseAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Sequence[BaseTool] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        provider: Provider | None = None,
        can_handoff: bool = True,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.can_handoff = can_handoff
        self.custom_tools: Sequence[BaseTool] = tools or []
        registry = get_registry()
        handoff_tools = registry.get_handoff_tools(name) if can_handoff else []
        self.tools: list[BaseTool] = list(self.custom_tools) + list(handoff_tools)

        base_llm, self.model_name = create_llm(model=model, temperature=temperature, provider=provider)
        self.llm: Runnable[Sequence[BaseMessage], AIMessage] = base_llm

        if self.tools:
            self.llm = base_llm.bind_tools(self.tools)

    async def process(self, state: AgentState) -> dict[str, Any] | Command[Any]:
        messages = self.get_messages_with_system(state)
        initial_response = await self._invoke_llm_with_retry(messages)

        if not isinstance(initial_response, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(initial_response)}")

        final_response, tool_messages = await self.handle_tool_calls(initial_response, messages)

        if isinstance(final_response, Command):
            if initial_response.tool_calls:
                return Command(
                    goto=final_response.goto,
                    update={"messages": [initial_response, *tool_messages]},
                )
            return Command(goto=final_response.goto, update={"messages": list(tool_messages)})

        content = final_response.content
        if not isinstance(content, str):
            content = str(content)

        if tool_messages and initial_response.tool_calls:
            return {
                "messages": [initial_response, *tool_messages, self.create_response_message(content)],
            }

        return {
            "messages": [self.create_response_message(content)],
        }

    async def invoke(self, state: AgentState) -> dict[str, Any] | Command[Any]:
        tracer = get_tracer()

        if state.get("iteration_count", 0) >= settings.max_iterations:
            tracer.error(self.name, "Max iterations reached")
            return Command(
                goto=END, update={"messages": [AIMessage(content=f"[{self.name}] Max iterations.", name=self.name)]}
            )

        last_msg = state["messages"][-1].content if state["messages"] else ""
        tracer.agent_start(self.name, str(last_msg))

        result = await self.process(state)
        execution_trace = list(state.get("execution_trace", []))
        execution_trace.append(self.name)

        if isinstance(result, Command):
            target = str(result.goto)

            if target != "supervisor" and target != END and detect_agent_cycle(execution_trace):
                tracer.error(self.name, "Ping-pong detected between agents. Escalating to supervisor.")
                return Command(
                    goto="supervisor",
                    update={
                        "messages": [
                            AIMessage(
                                content=f"[{self.name}] Conflict detected: agents are in a loop. "
                                f"Supervisor intervention required.",
                                name=self.name,
                            )
                        ],
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "current_agent": self.name,
                        "execution_trace": execution_trace,
                    },
                )

            tracer.handoff(self.name, target)
            update: dict[str, Any] = dict(result.update) if result.update else {}
            update["iteration_count"] = state.get("iteration_count", 0) + 1
            update["current_agent"] = self.name
            update["execution_trace"] = execution_trace
            update["loop_counts"] = state.get("loop_counts", {})
            return Command(goto=result.goto, update=update)

        result["iteration_count"] = state.get("iteration_count", 0) + 1
        result["current_agent"] = self.name
        result["execution_trace"] = execution_trace
        result["loop_counts"] = state.get("loop_counts", {})
        return result

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(min=1, max=60),
        retry=retry_if_exception(_is_retryable_error),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    async def _invoke_llm_with_retry(self, messages: list[BaseMessage]) -> AIMessage:
        tracer = get_tracer()
        start_time = time.perf_counter()

        response = await self.llm.ainvoke(messages)

        latency_ms = (time.perf_counter() - start_time) * 1000

        if not isinstance(response, AIMessage):
            return AIMessage(content=str(response.content), name=self.name)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", 0) or 0
            completion_tokens = getattr(usage, "output_tokens", 0) or 0
            tracer.llm_call(self.name, self.model_name, prompt_tokens, completion_tokens, latency_ms)

        return response

    def _is_handoff_tool(self, tool_name: str) -> bool:
        return tool_name.startswith("handoff_to_") or tool_name == "finish"

    async def _execute_single_tool(self, tool_call: Any, tracer: Any) -> tuple[str, ToolMessage, Command[Any] | None]:
        tool_call_id = str(tool_call["id"])
        tool_name = tool_call["name"]
        tool_args: dict[str, Any] = tool_call["args"]

        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            tracer.error(self.name, f"Tool not found: {tool_name}")
            return (
                tool_call_id,
                ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call_id),
                None,
            )

        tracer.tool_call(self.name, tool_name, tool_args)
        timeout_seconds = settings.tool_timeout_seconds

        try:
            async with asyncio.timeout(timeout_seconds):
                result = await tool.ainvoke(tool_args)

            tracer.tool_result(self.name, tool_name, result)

            if isinstance(result, Command):
                return (
                    tool_call_id,
                    ToolMessage(content=f"Handing off to {result.goto}", tool_call_id=tool_call_id),
                    result,
                )
            return (
                tool_call_id,
                ToolMessage(content=str(result), tool_call_id=tool_call_id),
                None,
            )

        except TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after {timeout_seconds}s"
            tracer.error(self.name, error_msg)
            return (
                tool_call_id,
                ToolMessage(content=f"Error: {error_msg}", tool_call_id=tool_call_id),
                None,
            )

        except Exception as e:
            error_msg = f"Tool '{tool_name}' failed: {type(e).__name__}: {e}"
            tracer.error(self.name, error_msg)
            return (
                tool_call_id,
                ToolMessage(content=f"Error: {error_msg}", tool_call_id=tool_call_id),
                None,
            )

    async def handle_tool_calls(
        self, response: AIMessage, messages: list[BaseMessage]
    ) -> tuple[AIMessage | Command[Any], list[ToolMessage]]:
        if not response.tool_calls:
            return response, []

        tracer = get_tracer()

        handoff_calls = [tc for tc in response.tool_calls if self._is_handoff_tool(tc["name"])]
        regular_calls = [tc for tc in response.tool_calls if not self._is_handoff_tool(tc["name"])]

        if handoff_calls:
            if len(handoff_calls) > 1:
                targets = [tc["name"] for tc in handoff_calls]
                logger.warning(f"Multiple handoff tools in single turn: {targets}. Using first: {targets[0]}")
            handoff_call = handoff_calls[0]
            _, tool_message, command = await self._execute_single_tool(handoff_call, tracer)

            other_messages: list[ToolMessage] = []
            for tc in response.tool_calls:
                if tc["id"] != handoff_call["id"]:
                    other_messages.append(
                        ToolMessage(
                            content=f"Skipped: handoff to {command.goto if command else 'unknown'} takes priority",
                            tool_call_id=str(tc["id"]),
                        )
                    )

            if command:
                return command, [tool_message, *other_messages]
            return response, [tool_message, *other_messages]

        tool_messages: list[ToolMessage] = []
        for tool_call in regular_calls:
            _, tool_message, command = await self._execute_single_tool(tool_call, tracer)
            tool_messages.append(tool_message)

            if command:
                for remaining_tc in regular_calls[regular_calls.index(tool_call) + 1 :]:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Skipped: handoff to {command.goto} takes priority",
                            tool_call_id=str(remaining_tc["id"]),
                        )
                    )
                return command, tool_messages

        all_messages: list[BaseMessage] = [*messages, response, *tool_messages]
        final_response = await self._invoke_llm_with_retry(all_messages)
        return final_response, tool_messages

    def get_messages_with_system(self, state: AgentState) -> list[BaseMessage]:
        tool_call_ids_with_responses: set[str] = set()
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                tool_call_ids_with_responses.add(msg.tool_call_id)

        complete_tool_call_ids: set[str] = set()
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_ids: list[str] = [str(tc["id"]) for tc in msg.tool_calls if tc.get("id") is not None]
                all_answered = all(tid in tool_call_ids_with_responses for tid in tool_ids)
                if all_answered:
                    complete_tool_call_ids.update(tool_ids)

        messages: list[BaseMessage] = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str) and content.startswith("[Supervisor]"):
                    continue
                if msg.tool_calls:
                    all_complete = all(tc["id"] in complete_tool_call_ids for tc in msg.tool_calls)
                    if not all_complete:
                        continue
            elif isinstance(msg, ToolMessage):
                if msg.tool_call_id not in complete_tool_call_ids:
                    continue
            messages.append(msg)

        return [SystemMessage(content=self.system_prompt), *messages]

    def create_response_message(self, content: str) -> AIMessage:
        return AIMessage(content=content, name=self.name)
