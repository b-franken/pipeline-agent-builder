from typing import Any, Final

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.llm import Provider
from src.trace import get_tracer
from src.types import AgentState

DEFAULT_CRITIQUE_PROMPT: Final = """You are a critical reviewer. Analyze the response below and identify:
1. Factual errors or unsupported claims
2. Missing important information
3. Clarity and structure issues
4. Whether it fully addresses the original request

If the response is good, say "APPROVED".
If it needs improvement, explain what's wrong and how to fix it.

Be concise and specific."""

DEFAULT_MAX_CONTEXT_MESSAGES: Final = 6
DEFAULT_MAX_TOKENS_PER_RUN: Final = 50000


class ReflectiveAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Any = None,
        model: str | None = None,
        temperature: float | None = None,
        provider: Provider | None = None,
        can_handoff: bool = True,
        max_refinements: int = 1,
        critique_prompt: str = DEFAULT_CRITIQUE_PROMPT,
        max_context_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
        max_tokens_per_run: int = DEFAULT_MAX_TOKENS_PER_RUN,
    ) -> None:
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
            temperature=temperature,
            provider=provider,
            can_handoff=can_handoff,
        )
        self.max_refinements = max_refinements
        self.critique_prompt = critique_prompt
        self.max_context_messages = max_context_messages
        self.max_tokens_per_run = max_tokens_per_run

    async def process(self, state: AgentState) -> dict[str, Any] | Command[Any]:
        tracer = get_tracer()
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

        current_response = final_response
        tokens_at_start = tracer.token_usage.total_tokens

        for i in range(self.max_refinements):
            tokens_used = tracer.token_usage.total_tokens - tokens_at_start
            if tokens_used >= self.max_tokens_per_run:
                tracer.tool_result(
                    self.name,
                    "budget_limit",
                    f"Token budget exceeded ({tokens_used}/{self.max_tokens_per_run}), stopping refinements",
                )
                break

            critique = await self._critique(current_response, state)

            if "APPROVED" in critique.upper():
                tracer.tool_result(self.name, "self_critique", f"Approved after {i} refinements")
                break

            tracer.tool_call(self.name, "self_critique", {"refinement": i + 1})
            current_response = await self._refine(current_response, critique, state)
            tracer.tool_result(self.name, "refine", f"Refined response #{i + 1}")

        content = current_response.content
        if not isinstance(content, str):
            content = str(content)

        if tool_messages and initial_response.tool_calls:
            return {
                "messages": [initial_response, *tool_messages, self.create_response_message(content)],
            }

        return {
            "messages": [self.create_response_message(content)],
        }

    async def _critique(self, response: AIMessage, state: AgentState) -> str:
        original_request = state["messages"][-1].content if state["messages"] else ""

        critique_messages: list[BaseMessage] = [
            SystemMessage(content=self.critique_prompt),
            SystemMessage(content=f"Original request: {original_request}"),
            response,
        ]

        critique_response = await self._invoke_llm_with_retry(critique_messages)
        return str(critique_response.content)

    def _get_truncated_context(self, state: AgentState) -> list[BaseMessage]:
        messages = list(state["messages"])

        if len(messages) <= self.max_context_messages:
            return self._ensure_valid_message_sequence(messages)

        first_human = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                first_human = msg
                break

        recent = messages[-self.max_context_messages :]
        if first_human and first_human not in recent:
            recent = [first_human, *recent[1:]]

        return self._ensure_valid_message_sequence(recent)

    def _ensure_valid_message_sequence(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        tool_call_ids_with_responses: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_call_ids_with_responses.add(msg.tool_call_id)

        complete_tool_call_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_ids: list[str] = [str(tc["id"]) for tc in msg.tool_calls if tc.get("id") is not None]
                all_answered = all(tid in tool_call_ids_with_responses for tid in tool_ids)
                if all_answered:
                    complete_tool_call_ids.update(tool_ids)

        result: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                all_complete = all(tc["id"] in complete_tool_call_ids for tc in msg.tool_calls)
                if not all_complete:
                    continue
            elif isinstance(msg, ToolMessage):
                if msg.tool_call_id not in complete_tool_call_ids:
                    continue
            result.append(msg)

        return result

    async def _refine(self, response: AIMessage, critique: str, state: AgentState) -> AIMessage:
        context = self._get_truncated_context(state)

        refine_messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=f"Feedback on your response:\n{critique}\n\nProvide an improved response."),
            *context,
            response,
        ]

        refined = await self._invoke_llm_with_retry(refine_messages)
        return refined
