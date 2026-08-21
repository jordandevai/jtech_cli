"""Full-screen Textual TUI.

The app owns the interactive loop and its UI-facing command context. Everything
it needs (settings, session, server info) is injected via the constructor by the
composition root (cli.py). Input is a pinned single-line ``Input`` that swaps to
a multi-line ``TextArea`` for ``'''`` heredocs, ``/write``, or Shift+Enter;
replies stream live into Markdown "bubbles" in the chat pane and Esc aborts a
stream in progress. While a slash-command prefix is typed, matching commands list
above the input and Up/Down plus Tab/Enter pick and run one. The settings dialog
is the same menu shape: rows for each setting, Up/Down to move, Enter to edit a
row in place (Enter saves, Esc cancels).
"""

from __future__ import annotations

import asyncio
import enum
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Markdown, Static, TextArea

from jtech_cli.cmd_tools import (
    CmdPolicy,
    ExecResult,
    allow_rule_for,
    decide,
    extract_cmd_blocks,
    format_result,
    truncate_output,
)
from jtech_cli.commands import CommandContext, build_registry
from jtech_cli.config import (
    CONFIG_PATH,
    SETTING_DESCRIPTIONS,
    SETTINGS,
    Settings,
    save_settings,
)
from jtech_cli import server_info
from jtech_cli.llm_client import stream_reply
from jtech_cli.server_info import ServerInfo
from jtech_cli.session import Session
from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT, textual_theme_name

CONNECTION_ERROR = "Connection failed — check base_url in /settings"
SPINNER_FRAMES = "-\\|/"

# Caps for the AI command loop: blocks per reply and re-stream rounds per turn.
MAX_BLOCKS_PER_REPLY = 5
MAX_TOOL_ROUNDS = 5


class CmdChoice(enum.Enum):
    ALLOW = "allow"
    ALWAYS = "always"
    DECLINE = "decline"


class CommandPrompt(ModalScreen[CmdChoice]):
    """Approval prompt for one AI-requested shell command.

    y = allow once, a = always allow (persists a prefix rule to config),
    n/Esc = decline. The decision is fed back to the model as the command's
    result, so a decline does not cancel the rest of the turn.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "allow_once", "Allow", show=False),
        Binding("a", "always_allow", "Always allow", show=False),
        Binding("n", "decline_cmd", "Decline", show=False),
        Binding("escape", "decline_cmd", "Decline", show=False),
    ]

    def __init__(self, command: str, reason: str) -> None:
        super().__init__()
        self._command = command
        self._reason = reason

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="cmd-dialog"):
            yield Static("Run command?", classes="dialog-title")
            yield Static(self._command, id="cmd-command")
            if self._reason:
                yield Static(self._reason, id="cmd-reason")
            yield Static("y allow · a always allow · n decline", id="cmd-hint")

    def action_allow_once(self) -> None:
        self.dismiss(CmdChoice.ALLOW)

    def action_always_allow(self) -> None:
        self.dismiss(CmdChoice.ALWAYS)

    def action_decline_cmd(self) -> None:
        self.dismiss(CmdChoice.DECLINE)


def render_menu_rows(items: Sequence[tuple[str, str]], index: int) -> Text:
    """Render (name, detail) rows as a ``Text`` block, marking the row at ``index``.

    Shared by the command menu and the settings menu so both look and behave alike.
    """
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


class _ChatInput(Input):
    """Single-line input; submits on Ctrl+Enter, opens multi-line on Shift+Enter.

    While the value is a slash-command prefix the app lists matching commands
    above the input: Up/Down move the highlight, Tab completes it, and Enter
    runs the highlighted command once you've scrolled (or when it's an exact
    match) — otherwise Enter just completes.
    """

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
        if isinstance(app, ChatApp):
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
        if isinstance(app, ChatApp):
            completed = app.accepted_suggestion()
            if completed is not None:
                app.apply_completion(completed)
                return
        self.screen.focus_next()

    def action_to_multiline(self) -> None:
        self.post_message(InputToMultiline(self.value))


class MultilineSubmit(Message):
    pass


class MultilineCancel(Message):
    pass


class InputToMultiline(Message):
    """Swap the single-line input for a multi-line editor pre-filled with ``value``."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


class _MultilineInput(TextArea):
    """Multi-line input; Ctrl+Enter submits, Esc cancels."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+enter", "multiline_submit", "Submit", show=False),
        Binding("escape", "multiline_cancel", "Cancel", show=False),
    ]

    def action_multiline_submit(self) -> None:
        self.post_message(MultilineSubmit())

    def action_multiline_cancel(self) -> None:
        self.post_message(MultilineCancel())


class FieldCommit(Message):
    """The in-place settings editor asks its screen to commit the edited value."""


class FieldCancel(Message):
    """The in-place settings editor asks its screen to discard the edit."""


class _FieldInput(Input):
    """Single-line in-place editor for a settings row; Enter commits, Esc cancels."""

    BINDINGS: ClassVar[list[Binding]] = [
        # priority so Enter wins before the base widget swallows the key
        Binding("enter", "commit_field", "Commit", show=False, priority=True),
        Binding("escape", "cancel_field", "Cancel", show=False, priority=True),
    ]

    def action_commit_field(self) -> None:
        self.post_message(FieldCommit())

    def action_cancel_field(self) -> None:
        self.post_message(FieldCancel())


class _PromptEditor(TextArea):
    """Multi-line in-place editor for the system prompt row; Enter commits, Esc cancels.

    Enter must be a priority binding: TextArea's own key handler consumes plain
    Enter (newline) before app-level bindings are consulted.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "commit_field", "Commit", show=False, priority=True),
        Binding("escape", "cancel_field", "Cancel", show=False, priority=True),
    ]

    def action_commit_field(self) -> None:
        self.post_message(FieldCommit())

    def action_cancel_field(self) -> None:
        self.post_message(FieldCancel())


class OutputSink:
    """Duck-typed rich ``Console`` used as ``CommandContext.console`` in the TUI.

    ``.print`` parses rich markup and routes the result into the chat as a dim
    "system" message instead of writing to stdout.
    """

    def __init__(self, app: ChatApp) -> None:
        self._app = app

    def print(self, *objects: object, sep: str = " ", end: str = "\n") -> None:
        text = sep.join(str(o) for o in objects) + end
        if not text.strip():
            return
        try:
            plain = Text.from_markup(text).plain
        except Exception:  # noqa: BLE001 - never crash on markup
            plain = text
        self._app.push_message("system", plain)


class SettingsScreen(ModalScreen[None]):
    """Menu-style settings, the same list shape as the command menu.

    Up/Down move between setting rows, Enter edits a row in place (Enter
    commits, Esc cancels the edit), Esc on the list closes. Edits validate and
    persist immediately, like /set — so there are no Save/Cancel buttons.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "row_up", "Prev", show=False),
        Binding("down", "row_down", "Next", show=False),
        Binding("enter", "edit_row", "Edit", show=False),
        Binding("escape", "close_modal", "Close", show=False),
    ]

    # Derived from the SETTINGS spec (config.py) — the single source of truth
    # for which settings exist, their order, and their labels.
    _ROWS: ClassVar[tuple[tuple[str, str], ...]] = tuple((s.name, s.label) for s in SETTINGS)
    _HINT_IDLE = "↑/↓ move · Enter edit · Esc close"
    _HINT_EDITING = "Enter save · Esc cancel"

    def __init__(
        self,
        settings: Settings,
        config_path: Path,
        on_save: Callable[[], None],
        cmd: CmdPolicy | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._config_path = config_path
        self._on_save = on_save
        self._cmd = cmd
        self._cursor = 0
        self._row: str | None = None  # key of the row currently being edited

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="settings-dialog"):
            yield Static("Settings", classes="dialog-title")
            yield Static(id="settings-rows")
            yield Vertical(id="settings-editor")
            yield Static(id="settings-help")
            yield Static(self._HINT_IDLE, id="settings-hint")

    def on_mount(self) -> None:
        self._render_rows()

    def action_close_modal(self) -> None:
        self.app.pop_screen()

    def action_row_up(self) -> None:
        self._move_row(-1)

    def action_row_down(self) -> None:
        self._move_row(1)

    def _move_row(self, direction: int) -> None:
        if self._row is not None:
            return
        self._cursor = (self._cursor + direction) % len(self._ROWS)
        self._render_rows()

    async def action_edit_row(self) -> None:
        if self._row is not None:
            return
        key = self._ROWS[self._cursor][0]
        editor: _FieldInput | _PromptEditor
        if key == "system_prompt":
            editor = _PromptEditor(self._row_value(key), id="settings-field")
        else:
            editor = _FieldInput(self._row_value(key), id="settings-field")
        await self.query_one("#settings-editor", Vertical).mount(editor)
        self._row = key
        self._set_hint(self._HINT_EDITING)
        editor.focus()

    def on_field_commit(self, _event: FieldCommit) -> None:
        self._commit_row()

    def on_field_cancel(self, _event: FieldCancel) -> None:
        self._close_editor()

    def _commit_row(self) -> None:
        key = self._row
        if key is None:
            return
        field = self.query_one("#settings-field")
        value = field.text if isinstance(field, TextArea) else field.value
        if key == "system_prompt":
            self._settings.system_prompt = value
        else:
            try:
                self._settings.set(key, value.strip())
            except ValueError as e:
                self.notify(str(e), severity="error")
                return
        try:
            if self._cmd is not None:
                self._cmd.mode = self._settings.cmd_mode
                save_settings(self._settings, self._config_path, cmd=self._cmd)
            else:
                save_settings(self._settings, self._config_path)
        except OSError as e:
            self.notify(f"Could not save settings: {e}", severity="warning")
        self._on_save()
        self._close_editor()

    def _close_editor(self) -> None:
        self.query_one("#settings-editor", Vertical).remove_children()
        self._row = None
        self._set_hint(self._HINT_IDLE)
        self._render_rows()

    def _set_hint(self, hint: str) -> None:
        self.query_one("#settings-hint", Static).update(hint)

    def _row_value(self, key: str) -> str:
        return str(getattr(self._settings, key))

    def _render_rows(self) -> None:
        items = [
            (label, self._preview(self._row_value(key))) for key, label in self._ROWS
        ]
        self.query_one("#settings-rows", Static).update(render_menu_rows(items, self._cursor))
        key, _ = self._ROWS[self._cursor]
        self.query_one("#settings-help", Static).update(SETTING_DESCRIPTIONS.get(key, ""))

    @staticmethod
    def _preview(value: str) -> str:
        return value if len(value) <= 60 else value[:59] + "…"


class ChatApp(App):
    CSS = """
    #chat {
        height: 1fr;
        padding: 0 1;
    }
    #status {
        height: 1;
        color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    #input {
        border: round $panel;
        padding: 0 1;
        margin-bottom: 1;
    }
    #suggestions {
        height: auto;
        max-height: 11;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    #multiline-input {
        height: 6;
        dock: bottom;
    }
    .bubble-label {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }
    .bubble {
        margin-bottom: 1;
    }
    /* Wrap long code lines instead of letting them run off the right edge:
       MarkdownFence's Label defaults to content width (expand=True) and the
       fence's horizontal scrollbar is hidden, so unbroken words were cut off.
       Constrained to the bubble width, the content folds like normal text. */
    #chat MarkdownFence > Label {
        width: 1fr;
    }
    .bubble.user {
        background: $surface;
        color: $text;
        padding: 1 2;
    }
    .bubble.ai {
        color: $text;
    }
    .bubble.reasoning {
        color: $text-muted;
    }
    .bubble.error {
        color: $error;
    }
    .bubble.system {
        color: $text-muted;
    }
    #settings-dialog {
        width: 72;
        height: auto;
        max-height: 100%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
        margin: 1 4;
    }
    #settings-dialog .dialog-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #settings-rows {
        padding: 0 1;
    }
    #settings-editor {
        height: auto;
        padding: 0 1;
    }
    #settings-editor TextArea {
        height: 5;
    }
    #settings-help {
        color: $text-muted;
        padding: 0 1;
        margin-top: 1;
    }
    #settings-hint {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
    }
    #cmd-dialog {
        width: 72;
        height: auto;
        max-height: 100%;
        border: round $warning;
        background: $surface;
        padding: 1 2;
        margin: 1 4;
    }
    #cmd-dialog .dialog-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #cmd-command {
        padding: 0 1;
        margin-bottom: 1;
    }
    #cmd-reason {
        color: $warning;
        margin-bottom: 1;
    }
    #cmd-hint {
        color: $text-muted;
        text-align: center;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+s", "settings", "Settings"),
        Binding("escape", "stop_stream", "Stop", show=False),
        Binding("up", "suggestion_up", "Prev", show=False),
        Binding("down", "suggestion_down", "Next", show=False),
    ]

    def __init__(
        self,
        *,
        settings: Settings,
        session: Session,
        server: ServerInfo,
        config_path: Path = CONFIG_PATH,
        cmd: CmdPolicy | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.session = session
        self.server = server
        self.config_path = config_path
        self.cmd = cmd if cmd is not None else CmdPolicy()
        self.settings.cmd_mode = self.cmd.mode
        self._project_root = Path.cwd().resolve()
        self._running_proc: subprocess.Popen | None = None
        self._cmd_interrupted = False
        self._tool_rounds_active = False
        self._multiline_textarea: _MultilineInput | None = None
        self._multiline_future: asyncio.Future[str] | None = None
        self._multiline_terminator = "'''"
        self._generating = False
        self._stop_event = threading.Event()
        self._prompt_tokens: int = 0
        self._queue: list[str] = []
        # The "Queued" chat line for each entry (same order as _queue), so the
        # line can be removed when its message leaves the queue.
        self._queue_lines: list[tuple[Static, Markdown]] = []
        self.ctx = CommandContext(
            session=session,
            settings=settings,
            console=OutputSink(self),
            server=server,
            cmd=self.cmd,
            config_path=config_path,
            enter_multiline=self._enter_multiline,
            refresh_footer=self._render_status,
            open_settings=self._open_settings,
            clear_chat=self._clear_chat,
            switch_theme=self._switch_theme,
        )
        self.commands = build_registry(self.ctx)
        self._suggestions: list[tuple[str, str]] = []
        self._suggestion_index = 0
        self._suggestion_navigated = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield VerticalScroll(id="chat")
            yield Static(id="suggestions")
            yield _ChatInput(
                id="input",
                placeholder="Message… (Shift+Enter for newline · / for commands)",
            )
            yield Static(id="status")

    async def on_mount(self) -> None:
        self.register_theme(JTECH_DARK)
        self.register_theme(JTECH_LIGHT)
        self.theme = textual_theme_name(self.settings.theme)
        for msg in self.session.messages:
            self._append(msg["role"], msg["content"])
        if not self.settings.base_url:
            self.push_message(
                "system", "No server configured — run /settings to set base_url and model."
            )
        self.query_one("#suggestions", Static).display = False
        self._render_status()
        self._focus_input()
        if self.session.messages and self.server.context_length:
            await self._init_token_count()

    def push_message(self, role: str, text: str) -> tuple[Static, Markdown]:
        return self._append(role, text)

    def _append(self, role: str, text: str) -> tuple[Static, Markdown]:
        chat = self.query_one("#chat", VerticalScroll)
        label = Static(role.upper(), classes=f"bubble-label {role}")
        chat.mount(label)
        md = Markdown(text or "", classes=f"bubble {role}")
        chat.mount(md)
        chat.scroll_end(animate=False)
        return label, md

    def _append_plain(self, role: str, text: str, *, hidden: bool = False) -> tuple[Static, Static]:
        """Mount a dimmed label + plain-text body pair (the reasoning bubble).

        A plain ``Static`` body keeps per-token updates cheap — no Markdown
        re-parse on every token or timer tick.
        """
        chat = self.query_one("#chat", VerticalScroll)
        label = Static(role.upper(), classes=f"bubble-label {role}")
        chat.mount(label)
        body = Static(text or "", classes=f"bubble {role}")
        chat.mount(body)
        if hidden:
            label.display = False
            body.display = False
        chat.scroll_end(animate=False)
        return label, body

    def _focus_input(self) -> None:
        try:
            self.query_one("#input", _ChatInput).focus()
        except Exception:  # noqa: BLE001, S110 - input may not be mounted yet
            pass

    async def _init_token_count(self) -> None:
        """Count session tokens on startup so the footer is accurate immediately."""
        text = " ".join(m["content"] for m in self.session.messages)
        if not text:
            return
        count = await asyncio.to_thread(server_info.fetch_token_count, self.settings, text)
        if count:
            self._prompt_tokens = count
            self._render_status()

    def on_input_changed(self, _event: Input.Changed) -> None:
        self._update_suggestions()

    def _update_suggestions(self) -> None:
        value = self.query_one("#input", _ChatInput).value
        if not value.startswith("/") or " " in value:
            self._hide_suggestions()
            return
        matches = self.commands.completions(value)[:10]
        if not matches:
            self._hide_suggestions()
            return
        self._suggestions = matches
        self._suggestion_index = 0
        self._suggestion_navigated = False
        self._render_suggestions()

    def _hide_suggestions(self) -> None:
        if not self._suggestions:
            return
        self._suggestions = []
        self._suggestion_index = 0
        self.query_one("#suggestions", Static).display = False

    def _render_suggestions(self) -> None:
        items = [(f"/{name}", help_text) for name, help_text in self._suggestions]
        box = self.query_one("#suggestions", Static)
        box.display = True
        box.update(render_menu_rows(items, self._suggestion_index))

    def action_suggestion_up(self) -> None:
        # While the command menu is open, Up cycles it; otherwise Up recalls
        # the next queued message into the input for editing.
        if self._suggestions and isinstance(self.focused, _ChatInput):
            self._cycle_suggestion(-1)
        else:
            self._recall_queued()

    def action_suggestion_down(self) -> None:
        self._cycle_suggestion(1)

    def _recall_queued(self) -> None:
        """Up with an empty input: pull the next queued message back into the
        input. It is not submitted — the user edits it (or clears it to cancel
        it), then Enter sends it (re-queueing at the back if a reply is still
        in flight). The escape hatch from a wrong queue entry.
        """
        if not self._queue or not isinstance(self.focused, _ChatInput):
            return
        inp = self.query_one("#input", _ChatInput)
        if inp.value.strip():
            return  # never clobber unsent text in the input
        text = self._queue[0]
        self._remove_queue_entry(0)  # clears the "Queued" line + status count
        inp.value = text
        inp.cursor_position = len(inp.value)

    def _cycle_suggestion(self, direction: int) -> None:
        if not self._suggestions or not isinstance(self.focused, _ChatInput):
            return
        self._suggestion_index = (self._suggestion_index + direction) % len(
            self._suggestions
        )
        self._suggestion_navigated = True
        self._render_suggestions()

    @property
    def suggestion_navigated(self) -> bool:
        """True once the user has moved the highlight (Enter then runs the row)."""
        return self._suggestion_navigated

    def accepted_suggestion(self) -> str | None:
        """The highlighted suggestion as ``"/name "`` or None when the list is hidden."""
        if not self._suggestions or not isinstance(self.focused, _ChatInput):
            return None
        name, _ = self._suggestions[self._suggestion_index]
        return f"/{name} "

    def apply_completion(self, completed: str) -> None:
        inp = self.query_one("#input", _ChatInput)
        inp.value = completed
        inp.cursor_position = len(completed)
        self._hide_suggestions()
        inp.focus()

    def _render_status(self, running: str | None = None) -> None:
        parts: list[str] = []
        if running:
            parts.append(running)
        if self.settings.base_url:
            parts.append(self.settings.base_url)
        if self.server.context_length:
            if self._prompt_tokens:
                total = self.server.context_length
                remaining_pct = max(0, (1 - self._prompt_tokens / total) * 100)
                parts.append(
                    f"ctx {self._prompt_tokens // 1000}k"
                    f"/{total // 1000}k ({remaining_pct:.0f}% left)"
                )
            else:
                parts.append("ctx n/a")
        if self._queue:
            parts.append(f"queue: {len(self._queue)}")
        self.query_one("#status", Static).update("  ·  ".join(parts))

    def _save(self) -> None:
        if self.session.persist:
            self.ctx.save()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.query_one("#input", _ChatInput).value = ""
        if not value:
            self._focus_input()
            return
        if value == "'''":
            self.run_worker(self._heredoc())
            return
        if value.startswith("/"):
            self._dispatch_command(value)
        else:
            self.run_worker(self._send_message(value))
        self._focus_input()

    def _dispatch_command(self, raw: str) -> None:
        try:
            result = self.commands.handle(raw)
        except SystemExit:
            self.exit()
            return
        # Async handlers (e.g. /write, /stats) return a coroutine: run it as a
        # worker so the event loop (and the token fetch / multiline editor)
        # never blocks the UI.
        if result is not None:
            self.run_worker(result)

    async def _heredoc(self) -> None:
        content = (await self._enter_multiline("'''")).strip()
        if content:
            await self._send_message(content)

    def on_input_to_multiline(self, event: InputToMultiline) -> None:
        self.query_one("#input", _ChatInput).value = ""
        self.run_worker(self._switch_to_multiline(event.value))

    async def _switch_to_multiline(self, prefill: str) -> None:
        content = (await self._enter_multiline("'''", prefill=prefill)).strip()
        if content:
            await self._send_message(content)

    def _enqueue(self, text: str) -> None:
        """Queue a message sent while a reply is still streaming.

        Shown immediately as a dim "Queued" system line, with a live count in
        the status bar. The queue drains in order once the in-flight reply
        finishes or is stopped with Esc.
        """
        self._queue.append(text)
        # No count in the line: it would go stale the moment a message ahead of
        # it is recalled. The status bar carries the live count.
        self._queue_lines.append(self.push_message("system", f"Queued: {text}"))
        self._render_status()

    def _remove_queue_entry(self, index: int) -> None:
        """Drop queue entry ``index`` and its "Queued" line from the chat.

        The line is transient status, not history — the real record is the
        USER bubble that appears when the message actually sends.
        """
        self._queue.pop(index)
        for w in self._queue_lines.pop(index):
            if w.is_mounted:
                w.remove()
        self._render_status()

    async def _enter_multiline(self, terminator: str = "'''", prefill: str = "") -> str:
        self._multiline_terminator = terminator
        self._multiline_future = asyncio.get_running_loop().create_future()
        inp = self.query_one("#input", _ChatInput)
        ta = _MultilineInput(classes="multiline", id="multiline-input")
        inp.display = False
        self.mount(ta)
        self._multiline_textarea = ta
        if prefill:
            ta.text = prefill
        ta.focus()
        return await self._multiline_future

    def on_multiline_submit(self, _event: MultilineSubmit) -> None:
        ta = self._multiline_textarea
        if ta is None:
            return
        lines = ta.text.splitlines()
        if lines and lines[-1].strip() == self._multiline_terminator:
            self._resolve_multiline("\n".join(lines[:-1]))

    def on_multiline_cancel(self, _event: MultilineCancel) -> None:
        self._resolve_multiline("")

    def _resolve_multiline(self, content: str) -> None:
        ta = self._multiline_textarea
        if ta is None:
            return
        ta.remove()
        self._multiline_textarea = None
        inp = self.query_one("#input", _ChatInput)
        inp.display = True
        inp.focus()
        fut = self._multiline_future
        self._multiline_future = None
        if fut is not None and not fut.done():
            fut.set_result(content)

    async def _send_message(self, content: str) -> None:
        content = content.strip()
        if not content:
            return
        if self._generating or self._tool_rounds_active:
            # A reply is in flight: queue instead of running two generations
            # concurrently (the in-flight _stream_reply drains the queue).
            self._enqueue(content)
            return
        self.session.add("user", content)
        self._save()
        self._append("user", content)
        await self._stream_reply()

    async def _stream_reply(self) -> None:
        reply = await self._stream_once()
        if reply is not None:
            await self._tool_rounds(reply)

        # Drain queued messages in order. Each send re-enters _stream_reply and
        # drains again, so the while loop re-checks for anything queued in the
        # meantime; frames unwind between iterations (no deep recursion).
        while self._queue:
            text = self._queue[0]
            self._remove_queue_entry(0)  # the USER bubble replaces the line
            await self._send_message(text)

    async def _stream_once(self) -> str | None:
        """One stream: returns the reply text, or None when stopped or errored."""
        parts: list[str] = []
        reason_parts: list[str] = []
        timings: dict | None = None
        error: BaseException | None = None
        done = asyncio.Event()
        started = time.monotonic()
        mode = self.settings.reasoning
        got_token = False
        tick_count = 0
        reason_visible = False
        last_content = ""
        last_reason = ""
        self._stop_event.clear()
        self._generating = True

        def reason_display(reason: str) -> str:
            """Reasoning text as shown: ``tail`` mode keeps only the last 500 chars."""
            return ("…" + reason[-500:]) if mode == "tail" else reason

        reason_pair: tuple[Static, Static] | None = None
        if mode != "hide":
            # Mounted *first* (hidden) so that, once revealed, it renders above
            # the AI bubble: the thinking happens before the answer, the chat
            # bottom stays the answer, and the newest reasoning line grows just
            # above the pinned AI label — in view as long as we follow along.
            reason_pair = self._append_plain("reasoning", "", hidden=True)
        label, ai_md = self._append("ai", "")
        # Stream the answer through Markdown.append (parses only the new lines,
        # mounts only the new blocks) instead of Markdown.update per token,
        # which re-parses and re-mounts the whole document while holding the
        # widget lock: quadratic re-parse work (UI lag) and fire-and-forget
        # update tasks that surface as "GatheringFuture exception was never
        # retrieved" CancelledError noise when they are cancelled at exit.
        ai_stream = Markdown.get_stream(ai_md)
        stream_closed = False

        def drop_reason_bubble() -> None:
            if reason_pair is None:
                return
            for w in reason_pair:
                if w.is_mounted:
                    w.remove()

        def paint() -> None:
            """Re-render the live bubbles + status label. UI thread only."""
            nonlocal reason_visible, last_content, last_reason, stream_closed
            if not label.is_mounted:  # a queued tick may run after the stop teardown
                return
            reason = "".join(reason_parts)
            content = "".join(parts)
            moved = False  # any layout change: new lines land below the viewport
            # reasoning bubble — separate from the answer, dim, plain text
            if reason_pair is not None:
                if reason and not reason_visible:
                    reason_pair[0].display = True
                    reason_pair[1].display = True
                    reason_visible = True
                    last_reason = reason_display(reason)
                    reason_pair[1].update(last_reason)
                    moved = True
                if reason_visible:
                    if content and mode == "transient":
                        # option 2: visible while reasoning, gone once the answer starts
                        drop_reason_bubble()
                        reason_visible = False
                        moved = True
                    else:
                        shown = reason_display(reason)
                        if shown != last_reason:
                            reason_pair[1].update(shown)
                            last_reason = shown
                            moved = True
            # answer bubble — content only, never reasoning. Only the new delta
            # goes to the stream; it coalesces fragments and appends, so the
            # widget never re-parses the whole document per token.
            if not stream_closed and content and content != last_content:
                delta = content[len(last_content):]
                last_content = content
                # paint always runs on the event-loop thread (timer ticks and
                # call_from_thread both marshal here); schedule the tiny write
                # coroutine without blocking this repaint.
                asyncio.ensure_future(ai_stream.write(delta))
                moved = True
            # status label — time-based (tick-driven) so it advances even while
            # the stream is silent
            elapsed = int(time.monotonic() - started)
            frame = SPINNER_FRAMES[tick_count % len(SPINNER_FRAMES)]
            if not got_token:
                label.update(f"AI  ·  waiting {elapsed}s")
            elif not content:
                label.update(f"AI  ·  {frame}  thinking… {len(reason)}")
            else:
                label.update(f"AI  ·  {frame}  {len(content)}")
            if moved:
                # layout grew: the chat bottom (AI label + newest reasoning
                # line / answer) moved down — follow it so it stays in view
                self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

        def tick() -> None:
            nonlocal tick_count
            tick_count += 1
            paint()

        timer = self.set_interval(1.0, tick)

        def consume() -> None:
            """Consume the blocking stream generator off the event loop thread."""
            nonlocal error, got_token, timings
            try:
                messages = self.session.messages_with_system(self.settings.system_prompt)
                for item in stream_reply(self.settings, messages):
                    if isinstance(item, tuple):
                        kind, payload = item
                        if kind == "reasoning":
                            reason_parts.append(payload)
                        elif kind == "usage":
                            self._prompt_tokens = payload["prompt_tokens"]
                        else:
                            timings = payload
                            self._prompt_tokens = int(payload.get("prompt_n", 0))
                    else:
                        parts.append(item)
                    got_token = True
                    self.call_from_thread(paint)
                    if self._stop_event.is_set():
                        break
            except Exception as e:  # noqa: BLE001 - report connection failures cleanly
                error = e
            finally:
                self.call_from_thread(done.set)

        threading.Thread(target=consume, daemon=True).start()
        await done.wait()

        try:
            timer.stop()
            # Settle the stream before touching the bubble: stop() cancels the
            # stream task and flushes pending fragments, but an in-flight
            # append (shielded inside the stream task) may outlive it; holding
            # the widget lock until it is fully done is what makes the
            # remove/replace below safe.
            stream_closed = True
            await ai_stream.stop()
            async with ai_md.lock:
                pass
            if self._stop_event.is_set():
                drop_reason_bubble()
                label.remove()
                ai_md.remove()
                self.push_message("system", "Generation stopped.")
                return None
            elif error is not None:
                if mode == "transient":
                    drop_reason_bubble()
                label.update("AI")
                ai_md.add_class("error")
                # Awaited (not fire-and-forget): a dangling update task is what
                # produced the "GatheringFuture never retrieved" noise.
                await ai_md.update(f"{CONNECTION_ERROR}\n\n{error}")
                self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
                return None
            else:
                if mode == "transient":
                    # covers the reasoning-only stream where no content token ever
                    # arrived to trigger the in-paint removal
                    drop_reason_bubble()
                label.update(self._done_label(timings))
                reply = "".join(parts)
                if reply.strip():
                    self.session.add("assistant", reply)
                    self._save()
                self.ctx.last_reply = reply
                return reply
        finally:
            self._generating = False
            self._render_status()

    async def _tool_rounds(self, reply: str) -> None:
        """Process `` ```cmd `` blocks in a reply, re-streaming after each round.

        Each round: gate every block (blacklist -> mode -> allowlist -> prompt),
        run or note the outcome, append the result to the session as a system
        message, then re-stream so the model can react. Bounded by
        MAX_TOOL_ROUNDS so a looping model cannot run forever. While active,
        newly submitted messages queue (they drain once the turn ends).
        """
        self._tool_rounds_active = True
        try:
            rounds = 0
            nudges = 0
            while rounds < MAX_TOOL_ROUNDS:
                blocks = extract_cmd_blocks(reply)
                if not blocks:
                    # Model stopped mid-task: nudge it (max 2, ephemeral).
                    if nudges < 2:
                        nudges += 1
                        nudge = "[system] Continue your task or respond to the user."
                        self.session.add("system", nudge)
                        reply = await self._stream_once()
                        # Remove the ephemeral nudge so it doesn't pollute history.
                        self.session.messages.remove({"role": "system", "content": nudge})
                        if reply is None:
                            return
                        continue
                    return
                if len(blocks) > MAX_BLOCKS_PER_REPLY:
                    self.push_message(
                        "system",
                        f"{len(blocks) - MAX_BLOCKS_PER_REPLY} extra command block(s) "
                        f"ignored (max {MAX_BLOCKS_PER_REPLY} per reply).",
                    )
                await self._process_blocks(blocks[:MAX_BLOCKS_PER_REPLY])
                rounds += 1
                reply = await self._stream_once()
                if reply is None:
                    return
        finally:
            self._tool_rounds_active = False

    async def _process_blocks(self, blocks: list[str]) -> None:
        for block in blocks:
            if not block.strip():
                await self._note_command(block, "empty command — not run")
                continue
            decision = decide(block, self.cmd, self._project_root)
            if decision.action == "blocked":
                await self._note_command(block, f"blocked — {decision.reason}")
                continue
            if decision.action == "ask":
                choice = await self._prompt_for_command(block, decision.reason)
                if choice is CmdChoice.DECLINE:
                    await self._note_command(block, "declined by the user")
                    continue
                if choice is CmdChoice.ALWAYS:
                    self._add_allow_rule(allow_rule_for(block))
            result = await self._exec_command(block)
            self.push_message("system", self._cmd_bubble(block, result))
            self.session.add("system", format_result(block, result=result))
            self._save()

    async def _note_command(self, command: str, note: str) -> None:
        """Feed a non-executed command outcome back to the model and show it."""
        self.push_message("system", f"$ {command}\n→ {note}")
        self.session.add("system", format_result(command, note=note))
        self._save()

    async def _prompt_for_command(self, command: str, reason: str) -> CmdChoice:
        return await self.push_screen_wait(CommandPrompt(command, reason))

    def _add_allow_rule(self, rule: str | None) -> None:
        if not rule or rule in self.cmd.allow:
            return
        self.cmd.allow.append(rule)
        try:
            self.cmd.mode = self.settings.cmd_mode
            save_settings(self.settings, self.config_path, cmd=self.cmd)
            self.push_message("system", f"Always-allow saved: {rule}")
        except OSError:
            pass

    async def _exec_command(self, command: str) -> ExecResult:
        """Run one command in a thread; Esc kills it, the timeout kills it."""
        self._render_status(running="running command…")
        self._cmd_interrupted = False
        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=self._project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as e:
            self._render_status()
            return ExecResult(127, str(e))
        self._running_proc = proc
        try:
            try:
                out, _ = await asyncio.to_thread(
                    proc.communicate, timeout=self.cmd.timeout
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
                return ExecResult(124, "", timed_out=True)
            out = out or ""
            if self._cmd_interrupted:
                self._cmd_interrupted = False
                return ExecResult(130, out.strip("\n"), interrupted=True)
            text, truncated = truncate_output(out, self.cmd.max_output)
            return ExecResult(proc.returncode, text, truncated=truncated)
        finally:
            self._running_proc = None
            self._render_status()

    @staticmethod
    def _cmd_bubble(command: str, result: ExecResult) -> str:
        if result.timed_out:
            return f"$ {command}\n\n**timed out**"
        if result.interrupted:
            return f"$ {command}\n\n**interrupted**"
        body = result.output if result.output else "(no output)"
        return f"$ {command} — exit {result.exit_code}\n\n```\n{body}\n```"

    @staticmethod
    def _done_label(timings: dict | None) -> str:
        """Final AI label; includes llama.cpp prompt-eval stats when reported."""
        if not timings:
            return "AI"
        prompt_n = int(timings.get("prompt_n", 0))
        prompt_s = float(timings.get("prompt_ms", 0)) / 1000
        prompt_tps = float(timings.get("prompt_per_second", 0))
        return f"AI  ·  prompt {prompt_n:,} tok · {prompt_s:.1f}s · {prompt_tps:.0f} t/s"

    def action_stop_stream(self) -> None:
        if self._generating:
            self._stop_event.set()
        elif self._running_proc is not None:
            # a command is running: kill it (communicate returns the partial output)
            self._cmd_interrupted = True
            self._running_proc.kill()
        else:
            self._hide_suggestions()

    def _open_settings(self) -> None:
        self.push_screen(
            SettingsScreen(self.settings, self.config_path, self._on_settings_saved, cmd=self.cmd)
        )

    def _switch_theme(self) -> None:
        name = textual_theme_name(self.settings.theme)
        if name != self.theme:
            self.theme = name

    def _on_settings_saved(self) -> None:
        self._switch_theme()
        self._render_status()

    def _clear_chat(self) -> None:
        self.query_one("#chat", VerticalScroll).remove_children()
        self._prompt_tokens = 0
        self._render_status()

    def action_clear_chat(self) -> None:
        self.commands.handle("/clear")

    def action_settings(self) -> None:
        self._open_settings()
