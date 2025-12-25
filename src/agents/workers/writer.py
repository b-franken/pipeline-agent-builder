"""Writer agent - creates and edits content."""

from src.agents.base import BaseAgent
from src.llm import Provider
from src.prompts import WRITER_PROMPT
from src.tools.knowledge import create_knowledge_tools


class WriterAgent(BaseAgent):
    def __init__(
        self,
        model: str | None = None,
        provider: Provider | None = None,
        use_knowledge: bool = True,
        can_handoff: bool = True,
    ) -> None:
        tools = create_knowledge_tools() if use_knowledge else []

        super().__init__(
            name="writer",
            system_prompt=WRITER_PROMPT,
            tools=tools,
            model=model,
            provider=provider,
            can_handoff=can_handoff,
        )
