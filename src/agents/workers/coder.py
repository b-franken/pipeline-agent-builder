"""Coder agent - writes and debugs code."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.llm import Provider, create_llm
from src.prompts import CODER_PROMPT
from src.schemas import CodeOutput
from src.tools.code import create_code_tools
from src.tools.knowledge import create_knowledge_tools
from src.types import AgentState


class CoderAgent(BaseAgent):
    def __init__(
        self,
        model: str | None = None,
        provider: Provider | None = None,
        use_structured_output: bool = False,
        use_knowledge: bool = True,
        can_handoff: bool = True,
    ) -> None:
        tools: list[BaseTool] = list(create_code_tools())
        if use_knowledge:
            tools.extend(create_knowledge_tools())

        super().__init__(
            name="coder",
            system_prompt=CODER_PROMPT,
            tools=tools,
            model=model,
            provider=provider,
            can_handoff=can_handoff,
        )
        self.use_structured_output = use_structured_output
        self._structured_llm: Any = None

        if use_structured_output:
            base_llm, _ = create_llm(model=model, provider=provider)
            self._structured_llm = base_llm.with_structured_output(CodeOutput)

    async def generate_code(self, state: AgentState) -> CodeOutput:
        if self._structured_llm is None:
            raise RuntimeError("Structured output not enabled. Set use_structured_output=True")

        messages = self.get_messages_with_system(state)
        result = await self._structured_llm.ainvoke(messages)

        if not isinstance(result, CodeOutput):
            raise TypeError(f"Expected CodeOutput, got {type(result).__name__}")

        return result

    async def process(self, state: AgentState) -> dict[str, Any] | Command[Any]:
        if not self.use_structured_output:
            return await super().process(state)

        code_output = await self.generate_code(state)

        content_parts = [
            f"**Language:** {code_output.language}",
            f"\n```{code_output.language}\n{code_output.code}\n```",
        ]

        if code_output.explanation:
            content_parts.append(f"\n**Explanation:** {code_output.explanation}")

        if code_output.dependencies:
            deps = ", ".join(code_output.dependencies)
            content_parts.append(f"\n**Dependencies:** {deps}")

        content = "\n".join(content_parts)

        return {
            "messages": [AIMessage(content=content, name=self.name)],
        }
