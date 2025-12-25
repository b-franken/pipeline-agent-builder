import logging
from typing import Final

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.llm import create_llm

logger: Final = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES: Final = 20
DEFAULT_SUMMARY_THRESHOLD: Final = 15
DEFAULT_KEEP_RECENT: Final = 6


class SummaryConfig(BaseModel):
    max_messages: int = DEFAULT_MAX_MESSAGES
    summary_threshold: int = DEFAULT_SUMMARY_THRESHOLD
    keep_recent: int = DEFAULT_KEEP_RECENT
    include_system_in_count: bool = False


class ConversationSummary(BaseModel):
    summary: str
    key_points: list[str]
    task_context: str


SUMMARIZATION_PROMPT: Final = """Summarize the following conversation concisely.
Focus on:
1. The main task or goal
2. Key decisions made
3. Important context for continuing the conversation
4. Any unresolved items

Keep the summary brief (2-3 sentences) but preserve essential information.

Conversation to summarize:
{messages}

Respond with a concise summary."""


def estimate_tokens(messages: list[BaseMessage]) -> int:
    total_chars = sum(len(str(msg.content)) for msg in messages)
    return total_chars // 4


def should_summarize(
    messages: list[BaseMessage],
    config: SummaryConfig | None = None,
) -> bool:
    config = config or SummaryConfig()

    countable = messages
    if not config.include_system_in_count:
        countable = [m for m in messages if not isinstance(m, SystemMessage)]

    return len(countable) >= config.summary_threshold


async def summarize_messages(
    messages: list[BaseMessage],
    config: SummaryConfig | None = None,
) -> tuple[list[BaseMessage], str | None]:
    config = config or SummaryConfig()

    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    conversation = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(conversation) < config.summary_threshold:
        return messages, None

    to_summarize = conversation[: -config.keep_recent]
    to_keep = conversation[-config.keep_recent :]

    if not to_summarize:
        return messages, None

    formatted = _format_messages_for_summary(to_summarize)

    try:
        llm, _ = create_llm(temperature=0.3)
        prompt = SUMMARIZATION_PROMPT.format(messages=formatted)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary_text = str(response.content)
    except Exception as e:
        logger.warning(f"Summarization failed, keeping all messages: {e}")
        return messages, None

    summary_message = AIMessage(
        content=f"[Previous conversation summary]\n{summary_text}",
        name="system_summary",
    )

    result = [*system_messages, summary_message, *to_keep]

    logger.debug(f"Summarized {len(to_summarize)} messages into summary, keeping {len(to_keep)} recent messages")

    return result, summary_text


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    lines: list[str] = []

    for msg in messages:
        role = _get_role_name(msg)
        content = str(msg.content)[:500]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _get_role_name(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        name = getattr(msg, "name", None)
        return name.capitalize() if name else "Assistant"
    if isinstance(msg, SystemMessage):
        return "System"
    return "Unknown"


def trim_messages(
    messages: list[BaseMessage],
    max_messages: int | None = None,
    keep_system: bool = True,
) -> list[BaseMessage]:
    max_messages = max_messages or DEFAULT_MAX_MESSAGES

    if len(messages) <= max_messages:
        return messages

    if keep_system:
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        conversation = [m for m in messages if not isinstance(m, SystemMessage)]

        available_slots = max_messages - len(system_messages)
        recent = conversation[-available_slots:] if available_slots > 0 else []

        return [*system_messages, *recent]

    return messages[-max_messages:]


class ContextManager:
    def __init__(self, config: SummaryConfig | None = None) -> None:
        self.config = config or SummaryConfig()
        self._last_summary: str | None = None

    @property
    def last_summary(self) -> str | None:
        return self._last_summary

    async def prepare_context(
        self,
        messages: list[BaseMessage],
        force_summarize: bool = False,
    ) -> list[BaseMessage]:
        if force_summarize or should_summarize(messages, self.config):
            result, summary = await summarize_messages(messages, self.config)
            self._last_summary = summary
            return result

        return messages

    def quick_trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return trim_messages(messages, self.config.max_messages)
