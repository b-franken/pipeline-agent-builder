"""
Local tracing/observability - see exactly what agents are doing.

No external services needed. Just logs to console with colors and structure.
Optionally broadcasts events to WebSocket clients for real-time frontend updates.
Supports correlation IDs for request tracing and structured JSON logging.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Final

logger: Final = logging.getLogger(__name__)

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    cid = correlation_id or str(uuid.uuid4())[:8]
    _correlation_id.set(cid)
    return cid


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    SUPERVISOR = "\033[95m"
    RESEARCHER = "\033[94m"
    CODER = "\033[92m"
    WRITER = "\033[93m"

    HANDOFF = "\033[96m"
    TOOL = "\033[33m"
    ERROR = "\033[91m"
    SUCCESS = "\033[92m"


AGENT_COLORS: dict[str, str] = {
    "supervisor": Colors.SUPERVISOR,
    "researcher": Colors.RESEARCHER,
    "coder": Colors.CODER,
    "writer": Colors.WRITER,
}


class TraceLevel(Enum):
    MINIMAL = 1  # Only agent switches and final output
    NORMAL = 2  # + tool calls and handoffs
    VERBOSE = 3  # + message content
    DEBUG = 4  # + all internal state


BroadcastFn = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class TokenUsage:
    """Tracks token usage across a trace."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.llm_calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        """Add token counts from an LLM call."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.llm_calls += 1

    def to_dict(self) -> dict[str, int]:
        """Export as dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


class Tracer:
    """Local tracer for agent communication visibility."""

    def __init__(
        self,
        level: TraceLevel = TraceLevel.NORMAL,
        correlation_id: str | None = None,
        structured_logging: bool = False,
        task_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self.level = level
        self.events: list[dict[str, Any]] = []
        self.start_time = datetime.now(UTC)
        self._broadcast_fn: BroadcastFn | None = None
        self.correlation_id = correlation_id or set_correlation_id()
        self.structured_logging = structured_logging
        self._span_counter = 0
        self.token_usage = TokenUsage()
        self.task_id = task_id
        self.thread_id = thread_id

    def set_broadcast(self, fn: BroadcastFn | None) -> None:
        """Set a broadcast function for real-time WebSocket updates."""
        self._broadcast_fn = fn

    def _broadcast(self, event: dict[str, Any]) -> None:
        if self._broadcast_fn is not None:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self._broadcast_fn(event))
                task.add_done_callback(self._handle_broadcast_error)
            except RuntimeError:
                pass

    def _handle_broadcast_error(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("WebSocket broadcast failed: %s: %s", type(exc).__name__, exc)

    def _next_span_id(self) -> str:
        self._span_counter += 1
        return f"{self.correlation_id}-{self._span_counter:04d}"

    def _timestamp(self) -> str:
        elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
        return f"{elapsed:6.2f}s"

    def _iso_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()

    def _color(self, agent: str) -> str:
        return AGENT_COLORS.get(agent, Colors.RESET)

    def _log_structured(self, event: dict[str, Any]) -> None:
        if self.structured_logging:
            logger.info(json.dumps(event, ensure_ascii=True, default=str))

    def _create_base_event(self, event_type: str, agent: str) -> dict[str, Any]:
        event: dict[str, Any] = {
            "correlation_id": self.correlation_id,
            "span_id": self._next_span_id(),
            "time": self._timestamp(),
            "timestamp_iso": self._iso_timestamp(),
            "type": event_type,
            "agent": agent,
            "timestamp": datetime.now(UTC).timestamp(),
        }
        if self.task_id:
            event["task_id"] = self.task_id
        if self.thread_id:
            event["thread_id"] = self.thread_id
        return event

    def agent_start(self, agent: str, message: str = "") -> None:
        """Log when an agent starts processing."""
        color = self._color(agent)
        ts = self._timestamp()

        cid_display = f"[{self.correlation_id}] " if self.correlation_id else ""
        print(
            f"\n{Colors.DIM}{ts}{Colors.RESET} {Colors.DIM}{cid_display}{Colors.RESET}"
            f"{color}{Colors.BOLD}[{agent.upper()}]{Colors.RESET}"
        )

        if message and self.level.value >= TraceLevel.VERBOSE.value:
            display = message[:200] + "..." if len(message) > 200 else message
            print(f"       {Colors.DIM}Input: {display}{Colors.RESET}")

        event = self._create_base_event("agent_start", agent)
        event["message"] = message[:200] if message else ""
        self.events.append(event)
        self._log_structured(event)
        self._broadcast(event)

    def agent_response(self, agent: str, content: str) -> None:
        """Log agent's response."""
        if self.level.value >= TraceLevel.VERBOSE.value:
            color = self._color(agent)
            display = content[:500] + "..." if len(content) > 500 else content
            print(f"       {color}{display}{Colors.RESET}")

        event = self._create_base_event("agent_response", agent)
        event["message"] = content[:300] if content else ""
        self.events.append(event)
        self._log_structured(event)
        self._broadcast(event)

    def tool_call(self, agent: str, tool_name: str, args: dict[str, Any]) -> None:
        """Log a tool call."""
        ts = self._timestamp()
        if self.level.value >= TraceLevel.NORMAL.value:
            args_str = json.dumps(args, ensure_ascii=True)[:100]
            print(f"{Colors.DIM}{ts}{Colors.RESET} {Colors.TOOL}  -> tool: {tool_name}({args_str}){Colors.RESET}")

        event = self._create_base_event("tool_call", agent)
        event["message"] = f"Calling {tool_name}"
        event["tool"] = tool_name
        event["args"] = {k: str(v)[:100] for k, v in args.items()}
        self.events.append(event)
        self._log_structured(event)
        self._broadcast(event)

    def tool_result(self, agent: str, tool_name: str, result: Any) -> None:
        """Log a tool result."""
        result_str = str(result)[:200]
        if self.level.value >= TraceLevel.VERBOSE.value:
            print(f"       {Colors.DIM}  → {result_str}{Colors.RESET}")

        event = self._create_base_event("tool_result", agent)
        event["message"] = f"{tool_name} completed"
        event["tool"] = tool_name
        event["result"] = result_str
        self.events.append(event)
        self._log_structured(event)
        self._broadcast(event)

    def llm_call(
        self,
        agent: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float | None = None,
    ) -> None:
        """Log an LLM call with token usage."""
        self.token_usage.add(prompt_tokens, completion_tokens)
        total = prompt_tokens + completion_tokens

        if self.level.value >= TraceLevel.DEBUG.value:
            ts = self._timestamp()
            latency_str = f" ({latency_ms:.0f}ms)" if latency_ms else ""
            print(
                f"{Colors.DIM}{ts}   llm: {model} "
                f"[{prompt_tokens}+{completion_tokens}={total} tokens]{latency_str}{Colors.RESET}"
            )

        event = self._create_base_event("llm_call", agent)
        event["model"] = model
        event["prompt_tokens"] = prompt_tokens
        event["completion_tokens"] = completion_tokens
        event["total_tokens"] = total
        event["latency_ms"] = latency_ms
        self.events.append(event)
        self._log_structured(event)

    def handoff(self, from_agent: str, to_agent: str, reason: str = "") -> None:
        """Log a handoff between agents."""
        ts = self._timestamp()
        from_color = self._color(from_agent)
        to_color = self._color(to_agent)

        handoff_msg = (
            f"{Colors.DIM}{ts}{Colors.RESET} {Colors.HANDOFF}  >> handoff: "
            f"{from_color}{from_agent}{Colors.RESET} -> {to_color}{to_agent}{Colors.RESET}"
        )
        print(handoff_msg)

        if reason and self.level.value >= TraceLevel.VERBOSE.value:
            print(f"       {Colors.DIM}  reason: {reason}{Colors.RESET}")

        event = self._create_base_event("handoff", from_agent)
        event["from"] = from_agent
        event["to"] = to_agent
        event["message"] = f"Handoff: {from_agent} -> {to_agent}"
        event["reason"] = reason
        self.events.append(event)
        self._log_structured(event)
        self._broadcast(event)

    def finish(self, agent: str) -> None:
        """Log workflow completion."""
        ts = self._timestamp()
        cid_display = f"[{self.correlation_id}] " if self.correlation_id else ""
        print(
            f"\n{Colors.DIM}{ts}{Colors.RESET} {Colors.DIM}{cid_display}{Colors.RESET}"
            f"{Colors.SUCCESS}[DONE] Finished{Colors.RESET} (by {agent})"
        )

        event = self._create_base_event("finish", agent)
        event["message"] = f"Task completed by {agent}"
        event["duration_seconds"] = (datetime.now(UTC) - self.start_time).total_seconds()
        self.events.append(event)
        self._log_structured(event)
        self._broadcast(event)

    def error(self, agent: str, error: str) -> None:
        """Log an error."""
        ts = self._timestamp()
        cid_display = f"[{self.correlation_id}] " if self.correlation_id else ""
        print(
            f"{Colors.DIM}{ts}{Colors.RESET} {Colors.DIM}{cid_display}{Colors.RESET}"
            f"{Colors.ERROR}[ERR] Error in {agent}: {error}{Colors.RESET}"
        )

        event = self._create_base_event("error", agent)
        event["message"] = f"Error: {error[:200]}"
        event["error"] = error
        self.events.append(event)
        self._log_structured(event)
        self._broadcast(event)

    def state_update(self, agent: str, updates: dict[str, Any]) -> None:
        """Log state changes (debug level only)."""
        if self.level.value >= TraceLevel.DEBUG.value:
            ts = self._timestamp()
            updates_str = json.dumps(updates, ensure_ascii=True, default=str)[:300]
            print(f"{Colors.DIM}{ts}   state: {updates_str}{Colors.RESET}")

        event = self._create_base_event("state_update", agent)
        event["updates"] = updates
        self.events.append(event)
        self._log_structured(event)

    def summary(self) -> None:
        """Print a summary of the trace."""
        cid_display = f" [{self.correlation_id}]" if self.correlation_id else ""
        print(f"\n{Colors.BOLD}--- Trace Summary{cid_display} ---{Colors.RESET}")

        agents_used: set[str] = set()
        handoffs = 0
        tools_called = 0
        errors = 0

        for e in self.events:
            if e["type"] == "agent_start":
                agents_used.add(str(e["agent"]))
            elif e["type"] == "handoff":
                handoffs += 1
            elif e["type"] == "tool_call":
                tools_called += 1
            elif e["type"] == "error":
                errors += 1

        elapsed = (datetime.now(UTC) - self.start_time).total_seconds()

        print(f"Correlation ID: {self.correlation_id}")
        print(f"Duration: {elapsed:.2f}s")
        print(f"Agents:   {', '.join(agents_used)}")
        print(f"Handoffs: {handoffs}")
        print(f"Tools:    {tools_called}")
        if self.token_usage.llm_calls > 0:
            usage = self.token_usage
            print(f"LLM Calls: {usage.llm_calls}")
            print(f"Tokens:   {usage.total_tokens} (in: {usage.prompt_tokens}, out: {usage.completion_tokens})")
        if errors > 0:
            print(f"{Colors.ERROR}Errors:   {errors}{Colors.RESET}")
        print(f"{Colors.DIM}---------------------{Colors.RESET}\n")

    def export_json(self) -> str:
        """Export trace as JSON for analysis."""
        export_data = {
            "correlation_id": self.correlation_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": (datetime.now(UTC) - self.start_time).total_seconds(),
            "token_usage": self.token_usage.to_dict(),
            "events": self.events,
        }
        return json.dumps(export_data, indent=2, ensure_ascii=True, default=str)


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def set_trace_level(level: TraceLevel) -> None:
    """Set the trace level."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(level)
    else:
        _tracer.level = level


def reset_tracer(
    level: TraceLevel = TraceLevel.NORMAL,
    correlation_id: str | None = None,
    structured_logging: bool = False,
    task_id: str | None = None,
    thread_id: str | None = None,
) -> Tracer:
    """Reset the tracer for a new run with optional correlation ID and task context."""
    global _tracer
    _tracer = Tracer(
        level=level,
        correlation_id=correlation_id,
        structured_logging=structured_logging,
        task_id=task_id,
        thread_id=thread_id,
    )
    return _tracer
