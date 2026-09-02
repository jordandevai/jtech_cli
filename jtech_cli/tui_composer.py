"""The Primary composer: suggestions, the pending queue, and the editor.

One object owns everything the user is in the middle of saying but has not
sent — a half-typed slash command and its menu, messages parked behind a
running turn, and a promoted multi-line draft. It is deliberately not a
widget: the queue's notices live in the chat transcript and the promotion
handshake starts in an app-level message handler, so the state outlives any
single widget that displays it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from textual.widget import Widget
from textual.widgets import Static

from jtech_cli.commands import CommandRegistry
from jtech_cli.tui_status import StatusView
from jtech_cli.tui_widgets import (
    PlainTail,
    Transcript,
    _ChatInput,
    _MultilineInput,
    render_menu_rows,
)


class ComposerHost(Protocol):
    """The app surface the composer draws and queues through."""

    commands: CommandRegistry
    status: StatusView

    @property
    def focused(self) -> Widget | None:
        """The widget with keyboard focus, or ``None``."""
        ...

    def chat_input(self) -> _ChatInput:
        """The single-line composer."""
        ...

    def suggestion_box(self) -> Static:
        """The completion menu."""
        ...

    def readonly_notice(self) -> Static:
        """The notice shown in place of the composer for a subagent view."""
        ...

    def primary_transcript(self) -> Transcript:
        """Primary's own activity stream, where queue notices live."""
        ...

    def mount(self, widget: Widget) -> Any:
        """Mount the multi-line editor."""
        ...


class Composer:
    """Unsent Primary input: completions, the queue, and the editor."""

    def __init__(self, host: ComposerHost) -> None:
        self._host = host
        self._suggestions: list[tuple[str, str]] = []
        self._suggestion_index = 0
        self._suggestion_navigated = False
        self._queue: list[str] = []
        self._queue_lines: list[PlainTail] = []
        self._multiline_textarea: _MultilineInput | None = None
        # ``None`` is the resolved value for a cancel, so it is distinct from a
        # submitted empty editor: only the latter is content the user chose.
        self._multiline_future: asyncio.Future[str | None] | None = None

    # ---------------------------------------------------------- suggestions

    def _typing_here(self) -> bool:
        """Whether the keyboard is in the composer rather than elsewhere."""
        return isinstance(self._host.focused, _ChatInput)

    def update_suggestions(self) -> None:
        value = self._host.chat_input().value
        if not value.startswith("/") or " " in value:
            self.hide_suggestions()
            return
        matches = self._host.commands.completions(value)
        if not matches:
            self.hide_suggestions()
            return
        self._suggestions = matches
        self._suggestion_index = 0
        self._suggestion_navigated = False
        self._render_suggestions()

    def hide_suggestions(self) -> None:
        if not self._suggestions:
            return
        self._suggestions = []
        self._suggestion_index = 0
        self._host.suggestion_box().display = False

    def _render_suggestions(self) -> None:
        items = [(f"/{name}", help_text) for name, help_text in self._suggestions]
        box = self._host.suggestion_box()
        box.display = True
        box.update(render_menu_rows(items, self._suggestion_index))

    def navigate_up(self) -> None:
        """Up moves through an open menu, or recalls the next queued message."""
        if self._suggestions and self._typing_here():
            self.cycle(-1)
        else:
            self.recall_queued()

    def cycle(self, direction: int) -> None:
        if not self._suggestions or not self._typing_here():
            return
        self._suggestion_index = (self._suggestion_index + direction) % len(
            self._suggestions
        )
        self._suggestion_navigated = True
        self._render_suggestions()

    @property
    def suggestion_navigated(self) -> bool:
        return self._suggestion_navigated

    def accepted_suggestion(self) -> str | None:
        if not self._suggestions or not self._typing_here():
            return None
        name, _ = self._suggestions[self._suggestion_index]
        return f"/{name} "

    def apply_completion(self, completed: str) -> None:
        inp = self._host.chat_input()
        inp.value = completed
        inp.cursor_position = len(completed)
        self.hide_suggestions()
        inp.focus()

    # ---------------------------------------------------------------- queue

    @property
    def queue(self) -> Sequence[str]:
        """Messages waiting behind the accepted turn, in arrival order."""
        return self._queue

    def enqueue(self, text: str) -> None:
        """Queue a message while a reply or tool round is in flight.

        The notice is literal app text rather than model Markdown, so it is a
        plain live entry: it can be withdrawn again by recall or by draining.
        """
        chat = self._host.primary_transcript()
        self._queue.append(text)
        self._queue_lines.append(chat.begin_plain("system", f"Queued: {text}"))
        self._host.status.render()

    def pop_next(self) -> str:
        """Take the oldest queued message, withdrawing its chat line."""
        text = self._queue[0]
        self._remove_entry(0)
        return text

    def _remove_entry(self, index: int) -> None:
        """Drop a queued message and its transient chat line."""
        self._queue.pop(index)
        self._host.primary_transcript().remove(self._queue_lines.pop(index))
        self._host.status.render()

    def recall_queued(self) -> None:
        """Recall the next queued message without submitting it."""
        if not self._queue or not self._typing_here():
            return
        inp = self._host.chat_input()
        if inp.value.strip():
            return
        inp.value = self.pop_next()
        inp.cursor_position = len(inp.value)

    # ------------------------------------------------------------ multiline

    @property
    def multiline(self) -> _MultilineInput | None:
        """The mounted multi-line editor, or ``None`` when it is not open."""
        return self._multiline_textarea

    @property
    def multiline_future(self) -> asyncio.Future[str | None] | None:
        """The unresolved result of an open editor, for tests and diagnostics."""
        return self._multiline_future

    async def enter_multiline(
        self, prefill: str = "", cursor_offset: int | None = None
    ) -> str | None:
        """Mount the multi-line editor and wait for a submit or a cancel.

        Returns the submitted text — ``""`` included, which is a real choice a
        caller such as ``/write`` must be able to honour — or ``None`` when the
        user cancelled with ``Esc``.

        ``cursor_offset`` is a codepoint index into ``prefill``. It crosses the
        widget boundary as an index rather than a row/column pair because the
        producer is a single-line ``Input`` that has no rows; converting it is
        left to the document API so line-ending arithmetic is not reimplemented
        here. ``None`` means the caret belongs at the end, which is what every
        caller that does not come from a keyboard or paste edit wants.
        """
        return await self.open_editor(prefill, cursor_offset)

    def open_editor(
        self, prefill: str, cursor_offset: int | None
    ) -> asyncio.Future[str | None]:
        """Mount the editor and claim ownership in one synchronous step.

        Returning the future rather than reading ``self._multiline_future``
        later matters: ``resolve()`` clears that attribute, and a submit can
        land before the awaiting worker has even started.
        """
        if self._multiline_textarea is not None:
            raise RuntimeError("A multi-line editor is already open.")
        future: asyncio.Future[str | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._multiline_future = future
        inp = self._host.chat_input()
        textarea = _MultilineInput(classes="multiline", id="multiline-input")
        inp.display = False
        self._host.mount(textarea)
        self._multiline_textarea = textarea
        if prefill:
            textarea.text = prefill
        offset = len(prefill) if cursor_offset is None else cursor_offset
        offset = max(0, min(offset, len(prefill)))
        textarea.move_cursor(textarea.document.get_location_from_index(offset))
        textarea.focus()
        return future

    def resolve(self, content: str | None) -> None:
        """Close the editor and answer whoever is awaiting its result."""
        textarea = self._multiline_textarea
        if textarea is None:
            return
        textarea.remove()
        self._multiline_textarea = None
        inp = self._host.chat_input()
        inp.display = True
        inp.focus()
        future = self._multiline_future
        self._multiline_future = None
        if future is not None and not future.done():
            future.set_result(content)

    # ----------------------------------------------------------- visibility

    def show(self, show: bool) -> None:
        """Show the Primary composer, or the read-only subagent notice.

        Hiding is display-only: the input value and selection, the suggestion
        data, the multi-line text and its unresolved future, the queue, and the
        Primary session are all left exactly as they were, so returning to
        Primary restores the draft the user left behind. A disabled ``Input`` is
        deliberately not used — it still looks like a destination.
        """
        readonly = self._host.readonly_notice()
        suggestions = self._host.suggestion_box()
        chat_input = self._host.chat_input()
        textarea = self._multiline_textarea
        if not show:
            suggestions.display = False
            chat_input.display = False
            if textarea is not None:
                textarea.display = False
            readonly.display = True
            return
        readonly.display = False
        if textarea is not None:
            # The same mounted editor and the same unresolved future: multi-line
            # mode was never left, only hidden.
            textarea.display = True
            return
        chat_input.display = True
        # Rebuilt from the preserved value rather than kept as a stale painted
        # menu, so the suggestions can never disagree with the input.
        self.update_suggestions()
