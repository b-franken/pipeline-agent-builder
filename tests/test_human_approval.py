from src.human.approval import (
    HumanResponse,
    InterruptType,
)


class TestInterruptType:
    def test_approval_value(self) -> None:
        assert InterruptType.APPROVAL.value == "approval"

    def test_edit_value(self) -> None:
        assert InterruptType.EDIT.value == "edit"

    def test_choice_value(self) -> None:
        assert InterruptType.CHOICE.value == "choice"

    def test_feedback_value(self) -> None:
        assert InterruptType.FEEDBACK.value == "feedback"


class TestHumanResponseFromInterrupt:
    def test_from_bool_true(self) -> None:
        response = HumanResponse.from_interrupt(True, InterruptType.APPROVAL)
        assert response.approved is True
        assert response.interrupt_type == InterruptType.APPROVAL

    def test_from_bool_false(self) -> None:
        response = HumanResponse.from_interrupt(False, InterruptType.APPROVAL)
        assert response.approved is False

    def test_from_string_yes(self) -> None:
        response = HumanResponse.from_interrupt("yes", InterruptType.APPROVAL)
        assert response.approved is True

    def test_from_string_y(self) -> None:
        response = HumanResponse.from_interrupt("Y", InterruptType.APPROVAL)
        assert response.approved is True

    def test_from_string_approve(self) -> None:
        response = HumanResponse.from_interrupt("approve", InterruptType.APPROVAL)
        assert response.approved is True

    def test_from_string_ok(self) -> None:
        response = HumanResponse.from_interrupt("ok", InterruptType.APPROVAL)
        assert response.approved is True

    def test_from_string_no(self) -> None:
        response = HumanResponse.from_interrupt("no", InterruptType.APPROVAL)
        assert response.approved is False

    def test_from_string_reject(self) -> None:
        response = HumanResponse.from_interrupt("reject", InterruptType.APPROVAL)
        assert response.approved is False

    def test_from_string_edit_type(self) -> None:
        response = HumanResponse.from_interrupt("edited content", InterruptType.EDIT)
        assert response.approved is True
        assert response.edited_content == "edited content"

    def test_from_string_feedback_type(self) -> None:
        response = HumanResponse.from_interrupt("my feedback", InterruptType.FEEDBACK)
        assert response.approved is True
        assert response.feedback_message == "my feedback"

    def test_from_int(self) -> None:
        response = HumanResponse.from_interrupt(2, InterruptType.CHOICE)
        assert response.approved is True
        assert response.selected_option == 2

    def test_from_dict_approved(self) -> None:
        response = HumanResponse.from_interrupt({"approved": True}, InterruptType.APPROVAL)
        assert response.approved is True

    def test_from_dict_accept(self) -> None:
        response = HumanResponse.from_interrupt({"accept": True}, InterruptType.APPROVAL)
        assert response.approved is True

    def test_from_dict_edited_content(self) -> None:
        response = HumanResponse.from_interrupt({"approved": True, "edited_content": "new content"}, InterruptType.EDIT)
        assert response.approved is True
        assert response.edited_content == "new content"

    def test_from_dict_content_alias(self) -> None:
        response = HumanResponse.from_interrupt({"approved": True, "content": "new content"}, InterruptType.EDIT)
        assert response.edited_content == "new content"

    def test_from_dict_selected_option(self) -> None:
        response = HumanResponse.from_interrupt({"approved": True, "selected_option": 1}, InterruptType.CHOICE)
        assert response.selected_option == 1

    def test_from_dict_choice_alias(self) -> None:
        response = HumanResponse.from_interrupt({"approved": True, "choice": 2}, InterruptType.CHOICE)
        assert response.selected_option == 2

    def test_from_dict_feedback(self) -> None:
        response = HumanResponse.from_interrupt({"approved": True, "feedback": "good work"}, InterruptType.FEEDBACK)
        assert response.feedback_message == "good work"

    def test_from_dict_message_alias(self) -> None:
        response = HumanResponse.from_interrupt({"approved": True, "message": "feedback msg"}, InterruptType.FEEDBACK)
        assert response.feedback_message == "feedback msg"

    def test_from_dict_raw_response_stored(self) -> None:
        raw = {"approved": True, "custom_field": "value"}
        response = HumanResponse.from_interrupt(raw, InterruptType.APPROVAL)
        assert response.raw_response == raw

    def test_from_unknown_type_returns_not_approved(self) -> None:
        response = HumanResponse.from_interrupt(
            object(),
            InterruptType.APPROVAL,  # type: ignore[arg-type]
        )
        assert response.approved is False


class TestHumanResponseDefaults:
    def test_default_values(self) -> None:
        response = HumanResponse(interrupt_type=InterruptType.APPROVAL)
        assert response.approved is False
        assert response.edited_content is None
        assert response.selected_option is None
        assert response.feedback_message is None
        assert response.raw_response is None

    def test_with_values(self) -> None:
        response = HumanResponse(
            interrupt_type=InterruptType.EDIT,
            approved=True,
            edited_content="edited",
            selected_option=1,
            feedback_message="feedback",
            raw_response={"key": "value"},
        )
        assert response.interrupt_type == InterruptType.EDIT
        assert response.approved is True
        assert response.edited_content == "edited"
        assert response.selected_option == 1
        assert response.feedback_message == "feedback"
        assert response.raw_response == {"key": "value"}
