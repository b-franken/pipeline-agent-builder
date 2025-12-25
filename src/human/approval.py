from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import interrupt

from src.types import AgentState


class InterruptType(str, Enum):
    APPROVAL = "approval"
    EDIT = "edit"
    CHOICE = "choice"
    FEEDBACK = "feedback"


@dataclass
class HumanResponse:
    interrupt_type: InterruptType
    approved: bool = False
    edited_content: str | None = None
    selected_option: int | None = None
    feedback_message: str | None = None
    raw_response: dict[str, Any] | None = None

    @classmethod
    def from_interrupt(cls, response: dict[str, Any], interrupt_type: InterruptType) -> HumanResponse:
        if isinstance(response, bool):
            return cls(interrupt_type=interrupt_type, approved=response)

        if isinstance(response, str):
            if interrupt_type == InterruptType.EDIT:
                return cls(interrupt_type=interrupt_type, approved=True, edited_content=response)
            if interrupt_type == InterruptType.FEEDBACK:
                return cls(interrupt_type=interrupt_type, approved=True, feedback_message=response)
            approved = response.lower().strip() in ("yes", "y", "approve", "ok", "true", "1")
            return cls(interrupt_type=interrupt_type, approved=approved)

        if isinstance(response, int):
            return cls(interrupt_type=interrupt_type, approved=True, selected_option=response)

        if isinstance(response, dict):
            return cls(
                interrupt_type=interrupt_type,
                approved=bool(response.get("approved", response.get("accept", False))),
                edited_content=response.get("edited_content", response.get("content")),
                selected_option=response.get("selected_option", response.get("choice")),
                feedback_message=response.get("feedback", response.get("message")),
                raw_response=response,
            )

        return cls(interrupt_type=interrupt_type, approved=False)


def request_approval(action: str, details: str) -> HumanResponse:
    response: dict[str, Any] = interrupt(
        {
            "type": InterruptType.APPROVAL.value,
            "action": action,
            "details": details,
            "message": f"Approval needed: {action}\n\n{details}",
            "options": ["approve", "reject"],
        }
    )
    return HumanResponse.from_interrupt(response, InterruptType.APPROVAL)


def request_edit(content: str, instruction: str = "Please review and edit if needed") -> HumanResponse:
    response: dict[str, Any] = interrupt(
        {
            "type": InterruptType.EDIT.value,
            "content": content,
            "instruction": instruction,
            "message": f"{instruction}\n\n---\n{content}\n---",
        }
    )
    result = HumanResponse.from_interrupt(response, InterruptType.EDIT)
    if result.approved and not result.edited_content:
        result.edited_content = content
    return result


def request_choice(options: list[str], prompt: str = "Please select an option") -> HumanResponse:
    formatted_options = "\n".join(f"  [{i}] {opt}" for i, opt in enumerate(options))
    response: dict[str, Any] = interrupt(
        {
            "type": InterruptType.CHOICE.value,
            "options": options,
            "prompt": prompt,
            "message": f"{prompt}\n\n{formatted_options}",
        }
    )
    return HumanResponse.from_interrupt(response, InterruptType.CHOICE)


def request_feedback(context: str, question: str = "Please provide feedback") -> HumanResponse:
    response: dict[str, Any] = interrupt(
        {
            "type": InterruptType.FEEDBACK.value,
            "context": context,
            "question": question,
            "message": f"{question}\n\nContext:\n{context}",
        }
    )
    return HumanResponse.from_interrupt(response, InterruptType.FEEDBACK)


class HumanApprovalNode:
    def __init__(self) -> None:
        self.name = "human_approval"

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        context = state.get("context", {})
        reason = str(context.get("approval_reason", "Action requires approval"))
        last_msg = state["messages"][-1] if state["messages"] else None
        details = str(last_msg.content) if last_msg else "No details"

        response = request_approval(action=reason, details=details)

        new_context = dict(context)
        new_context["pending_human_approval"] = False
        new_context["human_approved"] = response.approved

        status = "Approved" if response.approved else "Rejected"
        return {
            "messages": [HumanMessage(content=f"[{status}: {reason}]")],
            "context": new_context,
            "human_feedback": response.feedback_message or str(response.raw_response),
        }


class HumanEditNode:
    def __init__(self, instruction: str = "Review and edit the output if needed") -> None:
        self.name = "human_edit"
        self.instruction = instruction

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        last_msg = state["messages"][-1] if state["messages"] else None
        content = str(last_msg.content) if last_msg else ""

        response = request_edit(content=content, instruction=self.instruction)

        new_context = dict(state.get("context", {}))
        new_context["human_edited"] = response.edited_content != content
        new_context["original_content"] = content

        messages: list[BaseMessage] = []
        if response.edited_content and response.edited_content != content:
            messages.append(HumanMessage(content=f"[Edited output]\n{response.edited_content}"))
        else:
            messages.append(HumanMessage(content="[Output approved without changes]"))

        return {
            "messages": messages,
            "context": new_context,
            "human_feedback": response.edited_content,
        }


class HumanChoiceNode:
    def __init__(self, options_key: str = "options", prompt: str = "Choose an option") -> None:
        self.name = "human_choice"
        self.options_key = options_key
        self.prompt = prompt

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        context = state.get("context", {})
        options = context.get(self.options_key, ["Option A", "Option B"])
        if not isinstance(options, list):
            options = [str(options)]

        response = request_choice(options=options, prompt=self.prompt)

        new_context = dict(context)
        new_context["selected_option_index"] = response.selected_option
        if response.selected_option is not None and 0 <= response.selected_option < len(options):
            new_context["selected_option"] = options[response.selected_option]
        else:
            new_context["selected_option"] = None

        selected = new_context.get("selected_option", "None")
        return {
            "messages": [HumanMessage(content=f"[Selected: {selected}]")],
            "context": new_context,
            "human_feedback": str(response.selected_option),
        }


class HumanFeedbackNode:
    def __init__(self, question: str = "Please provide your feedback") -> None:
        self.name = "human_feedback"
        self.question = question

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        recent_messages = state["messages"][-3:] if state["messages"] else []
        context = "\n".join(str(m.content) for m in recent_messages)

        response = request_feedback(context=context, question=self.question)

        new_context = dict(state.get("context", {}))
        feedback_history = list(new_context.get("feedback_history", []))
        if response.feedback_message:
            feedback_history.append(response.feedback_message)
        new_context["feedback_history"] = feedback_history
        new_context["last_feedback"] = response.feedback_message

        messages: list[BaseMessage] = []
        if response.feedback_message:
            messages.append(HumanMessage(content=f"[Feedback received]\n{response.feedback_message}"))
        else:
            messages.append(HumanMessage(content="[No feedback provided]"))

        return {
            "messages": messages,
            "context": new_context,
            "human_feedback": response.feedback_message,
        }


def request_human_approval(action: str, details: str) -> dict[str, str]:
    response = request_approval(action, details)
    return {
        "approved": str(response.approved),
        "feedback": response.feedback_message or "",
    }
