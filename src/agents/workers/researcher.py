"""Research agent - finds and synthesizes information."""

from langchain_core.tools import BaseTool

from src.agents.base import BaseAgent
from src.llm import Provider
from src.prompts import RESEARCHER_PROMPT
from src.tools.knowledge import create_knowledge_tools
from src.tools.search import create_search_tool


class ResearcherAgent(BaseAgent):
    def __init__(
        self,
        model: str | None = None,
        provider: Provider | None = None,
        use_knowledge: bool = True,
        can_handoff: bool = True,
    ) -> None:
        tools: list[BaseTool] = [create_search_tool()]
        if use_knowledge:
            tools.extend(create_knowledge_tools())

        super().__init__(
            name="researcher",
            system_prompt=RESEARCHER_PROMPT,
            tools=tools,
            model=model,
            provider=provider,
            can_handoff=can_handoff,
        )
