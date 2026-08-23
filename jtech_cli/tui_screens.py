"""Modal screens for command approval and settings editing."""

from __future__ import annotations

import enum
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, TextArea

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    SETTING_DESCRIPTIONS,
    SETTINGS,
    Settings,
    load_cmd_policy,
    save_settings,
)
from jtech_cli.tui_widgets import (
    FieldCancel,
    FieldCommit,
    _FieldInput,
    _PromptEditor,
    render_menu_rows,
)


class CmdChoice(enum.Enum):
    ALLOW = "allow"
    ALWAYS = "always"
    DECLINE = "decline"


class CommandPrompt(ModalScreen[CmdChoice]):
    """Approval prompt for one AI-requested shell command."""

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


class SettingsScreen(ModalScreen[None]):
    """Menu-style settings editor with immediate validation and persistence."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "row_up", "Prev", show=False),
        Binding("down", "row_down", "Next", show=False),
        Binding("enter", "edit_row", "Edit", show=False),
        Binding("escape", "close_modal", "Close", show=False),
    ]

    _ROWS: ClassVar[tuple[tuple[str, str], ...]] = tuple(
        (setting.name, setting.label) for setting in SETTINGS
    )
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
        self._row: str | None = None

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
            except ValueError as error:
                self.notify(str(error), severity="error")
                return
        try:
            cmd = self._cmd
            if cmd is None:
                # Preserve allow/timeout entries when no policy was injected.
                cmd = load_cmd_policy(self._config_path)
            cmd.mode = self._settings.cmd_mode
            save_settings(self._settings, self._config_path, cmd=cmd)
        except OSError as error:
            self.notify(f"Could not save settings: {error}", severity="warning")
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
            (label, self._preview(self._row_value(key)))
            for key, label in self._ROWS
        ]
        self.query_one("#settings-rows", Static).update(
            render_menu_rows(items, self._cursor)
        )
        key, _ = self._ROWS[self._cursor]
        self.query_one("#settings-help", Static).update(
            SETTING_DESCRIPTIONS.get(key, "")
        )

    @staticmethod
    def _preview(value: str) -> str:
        return value if len(value) <= 60 else value[:59] + "…"
