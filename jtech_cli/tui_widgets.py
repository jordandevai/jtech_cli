"""Reusable Textual widgets and messages used by the chat TUI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Protocol, runtime_checkable

from rich.console import Console
from rich.errors import MarkupError
from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Input, TextArea


@runtime_checkable
class SuggestionHost(Protocol):
    """The small app surface needed by ``_ChatInput`` completion actions."""

    @property
    def suggestion_navigated(self) -> bool: ...

    def accepted_suggestion(self) -> str | None: ...

    def apply_completion(self, completed: str) -> None: ...


class MessagePusher(Protocol):
    """The app surface needed by ``OutputSink``."""

    def push_message(self, role: str, text: str) -> object: ...


def render_menu_rows(items: Sequence[tuple[str, str]], index: int) -> Text:
    """Render menu rows, marking the selected row with a chevron."""
    parts: list[object] = []
    for i, (name, detail) in enumerate(items):
        selected = i == index
        line = Text()
        line.append("▸ " if selected else "  ")
        line.append(name, "bold" if selected else "")
        if detail:
            line.append(f"  {detail}", "dim")
        parts.append(line)
        if i < len(items) - 1:
            parts.append("\n")
    return Text.assemble(*parts)


class InputToMultiline(Message):
    """Request that the app replace the single-line editor with a text area."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


class MultilineSubmit(Message):
    """The multi-line editor asks the app to submit its current value."""


class MultilineCancel(Message):
    """The multi-line editor asks the app to cancel editing."""


class FieldCommit(Message):
    """The in-place settings editor asks its screen to commit the value."""


class FieldCancel(Message):
    """The in-place settings editor asks its screen to discard the value."""


class _ChatInput(Input):
    """Single-line input with command completion and multi-line shortcuts."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+enter", "submit_input", "Submit", show=False),
        Binding("shift+enter", "to_multiline", "Multi-line", show=False),
        Binding("enter", "submit_or_complete", "Submit", show=False),
        Binding("tab", "tab_complete", "Complete", show=False),
    ]

    def action_submit_input(self) -> None:
        self.post_message(Input.Submitted(self, self.value))

    def action_submit_or_complete(self) -> None:
        app = self.app
        if isinstance(app, SuggestionHost):
            completed = app.accepted_suggestion()
            if completed is not None:
                if app.suggestion_navigated or self.value.strip() == completed.strip():
                    self.post_message(Input.Submitted(self, completed))
                    return
                app.apply_completion(completed)
                return
        self.post_message(Input.Submitted(self, self.value))

    def action_tab_complete(self) -> None:
        app = self.app
        if isinstance(app, SuggestionHost):
            completed = app.accepted_suggestion()
            if completed is not None:
                app.apply_completion(completed)
                return
        self.screen.focus_next()

    def action_to_multiline(self) -> None:
        self.post_message(InputToMultiline(self.value))


class _MultilineInput(TextArea):
    """Multi-line input; Ctrl+Enter submits and Esc cancels."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+enter", "multiline_submit", "Submit", show=False),
        Binding("escape", "multiline_cancel", "Cancel", show=False),
    ]

    def action_multiline_submit(self) -> None:
        self.post_message(MultilineSubmit())

    def action_multiline_cancel(self) -> None:
        self.post_message(MultilineCancel())


class _FieldInput(Input):
    """Single-line in-place settings editor; Enter commits and Esc cancels."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "commit_field", "Commit", show=False, priority=True),
        Binding("escape", "cancel_field", "Cancel", show=False, priority=True),
    ]

    def action_commit_field(self) -> None:
        self.post_message(FieldCommit())

    def action_cancel_field(self) -> None:
        self.post_message(FieldCancel())


class _PromptEditor(TextArea):
    """Multi-line in-place editor for the system prompt row."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "commit_field", "Commit", show=False, priority=True),
        Binding("escape", "cancel_field", "Cancel", show=False, priority=True),
    ]

    def action_commit_field(self) -> None:
        self.post_message(FieldCommit())

    def action_cancel_field(self) -> None:
        self.post_message(FieldCancel())


class OutputSink:
    """Render command output into the chat instead of writing to stdout."""

    def __init__(self, app: MessagePusher) -> None:
        self._app = app
        self._renderer = Console(width=100, no_color=True, highlight=False)

    def _to_text(self, obj: object) -> str:
        """Convert one printable object to plain text."""
        if isinstance(obj, str):
            return obj
        if not hasattr(obj, "__rich_console__") and not hasattr(obj, "__rich__"):
            return str(obj)
        with self._renderer.capture() as capture:
            self._renderer.print(obj)
        return capture.get().rstrip("\n")

    def print(self, *objects: object, sep: str = " ", end: str = "\n") -> None:
        text = sep.join(self._to_text(obj) for obj in objects) + end
        if not text.strip():
            return
        try:
            plain = Text.from_markup(text).plain
        except MarkupError:
            # Command handlers deliberately use Rich markup; malformed markup
            # should still be visible as literal output in the chat.
            plain = text
        self._app.push_message("system", plain)
