"""Modal screens for command approval, settings, profiles, and quit confirmation."""

from __future__ import annotations

import enum
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Protocol

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen, ScreenResultType
from textual.widgets import Static, TextArea

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    SETTING_DESCRIPTIONS,
    SETTINGS,
    Profile,
    ProfileError,
    Profiles,
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


class _ChatModal(ModalScreen[ScreenResultType]):
    """Base modal that keeps the application's contextual ``Ctrl+C`` reachable.

    Textual truncates the non-priority binding chain at the innermost modal
    (``Screen._modal_binding_chain``), so a modal otherwise swallows ``Ctrl+C``
    once its own copy action skips, and the app never sees it. Both
    destinations are declared for the one key: ``_check_bindings()`` walks them
    in order and stops at the first that does not raise ``SkipAction``, so a
    non-empty selection still copies and only an unconsumed press reaches the
    app.

    ``screen.copy_text`` is re-declared rather than reimplemented; it has to be
    named again only because a subclass that declares a key replaces every
    base-class binding for it. ``Cmd+C`` is untouched — ``Screen`` registers it
    under the separate ``super+c`` key.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "screen.copy_text", "Copy selected text", show=False),
        Binding("ctrl+c", "app.contextual_ctrl_c", "Quit", show=False),
    ]


class CmdChoice(enum.Enum):
    ALLOW = "allow"
    ALWAYS = "always"
    DECLINE = "decline"


class CommandPrompt(_ChatModal[CmdChoice]):
    """Approval prompt for one AI-requested shell command.

    Any agent may be the requester, and only one prompt is ever mounted at a
    time, so the title names who is asking: the authorization target has to be
    unambiguous whichever activity stream happens to be on screen.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "allow_once", "Allow", show=False),
        Binding("a", "always_allow", "Always allow", show=False),
        Binding("n", "decline_cmd", "Decline", show=False),
        Binding("escape", "decline_cmd", "Decline", show=False),
    ]

    def __init__(
        self, command: str, reason: str, *, requester: str = "Primary"
    ) -> None:
        """Args:
            command: The command awaiting a decision.
            reason: Why the policy is asking rather than running it.
            requester: The label of the agent that asked. The default keeps
                standalone callers unchanged.
        """
        super().__init__()
        self._command = command
        self._reason = reason
        self._requester = requester

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="cmd-dialog"):
            # A ``Text`` value, not markup: an agent label is data, and console
            # markup in it must not be able to restyle or forge this dialog.
            yield Static(
                Text(f"Run command for {self._requester}?"), classes="dialog-title"
            )
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


class SettingsScreen(_ChatModal[None]):
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
    _HINT_PROMPT_EDITING = "Enter save · Ctrl+J newline · Esc cancel"

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
            hint = self._HINT_PROMPT_EDITING
        else:
            editor = _FieldInput(self._row_value(key), id="settings-field")
            hint = self._HINT_EDITING
        await self.query_one("#settings-editor", Vertical).mount(editor)
        self._row = key
        self._set_hint(hint)
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
            self._settings.set_prompt_inline(value)
        else:
            try:
                self._settings.set(key, value.strip())
            except ValueError as error:
                self.notify(str(error), severity="error")
                return
        try:
            cmd = self._cmd
            if cmd is None:
                # Preserve the allowlist and output limit when no policy was
                # injected.
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


class CommitProfiles(Protocol):
    """Persist a candidate catalog, raising if it cannot be stored.

    ``activated`` marks an explicit user selection, which the app treats as
    superseding any ``--base-url``/``--model`` override for the session.
    """

    async def __call__(
        self, candidate: Profiles, *, activated: bool = False
    ) -> None: ...


class ProfilesScreen(_ChatModal[None]):
    """Manage API profiles: list, activate, add, edit, rename, and delete.

    Deliberately probes nothing. Connectivity is transient and a local server
    may legitimately be stopped while its profile is edited, so an online/offline
    column here would be invented state; first-run setup remains the one flow
    that requires a successful probe.

    The screen owns presentation only. ``commit`` persists a candidate catalog
    and raises on failure, and the local copy advances only once it succeeded —
    so a failed save leaves the previous catalog selected and the modal open.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "row_up", "Prev", show=False),
        Binding("down", "row_down", "Next", show=False),
        Binding("enter", "select_row", "Select", show=False),
        Binding("escape", "back", "Back", show=False),
    ]

    ADD_ROW = "Add profile…"
    ACTIONS: ClassVar[tuple[str, ...]] = ("Activate", "Edit", "Delete", "Back")
    CONFIRM: ClassVar[tuple[str, ...]] = ("Confirm delete", "Cancel")
    # (widget id, label, Profile field)
    FIELDS: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("profile-name", "Name", "name"),
        ("profile-url", "Base URL", "base_url"),
        ("profile-model", "Model", "model"),
        ("profile-key", "API key variable", "api_key_env"),
    )

    _HINT_LIST = "↑/↓ move · Enter select · Esc close"
    _HINT_ACTIONS = "↑/↓ move · Enter run · Esc back"
    _HINT_EDIT = "↑/↓ field · Enter save · Esc cancel"
    _HINT_CONFIRM = "↑/↓ move · Enter choose · Esc back"

    _EDIT_HELP = (
        "Model may be blank to use the server's only model. "
        "API key variable may be blank for a local server; it names an "
        "environment variable, never the key itself."
    )

    def __init__(self, profiles: Profiles, commit: CommitProfiles) -> None:
        super().__init__()
        self._profiles = profiles
        self._commit = commit
        self._state = "list"
        self._cursor = 0
        # The profile the action/edit/confirm states act on; None while adding.
        self._target: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="profiles-dialog"):
            yield Static("Profiles", classes="dialog-title")
            yield Static(id="profiles-rows")
            yield Vertical(id="profiles-editor")
            yield Static(id="profiles-help")
            yield Static(self._HINT_LIST, id="profiles-hint")

    def on_mount(self) -> None:
        self._refresh_view()

    # --- rendering ---------------------------------------------------------

    def _list_rows(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for profile in self._profiles.items:
            marker = " (active)" if profile.name == self._profiles.active_name else ""
            detail = profile.base_url
            if profile.model:
                detail += f" · {profile.model}"
            if profile.api_key_env:
                detail += f" · ${profile.api_key_env}"
            rows.append((f"{profile.name}{marker}", detail))
        rows.append((self.ADD_ROW, ""))
        return rows

    def _rows(self) -> list[tuple[str, str]]:
        if self._state == "list":
            return self._list_rows()
        if self._state == "actions":
            return [(action, "") for action in self.ACTIONS]
        if self._state == "confirm":
            return [(choice, "") for choice in self.CONFIRM]
        return []

    def _help_text(self) -> str:
        if self._state == "edit":
            target = f"Editing {self._target}" if self._target else "New profile"
            return f"{target}. {self._EDIT_HELP}"
        if self._state == "actions":
            return f"Profile: {self._target}"
        if self._state == "confirm":
            return f"Delete profile {self._target}? The active profile cannot be deleted."
        if not self._profiles.items:
            return "No profiles yet — choose Add profile… to configure an endpoint."
        return "The active profile is used for every new turn."

    def _hint(self) -> str:
        return {
            "list": self._HINT_LIST,
            "actions": self._HINT_ACTIONS,
            "edit": self._HINT_EDIT,
            "confirm": self._HINT_CONFIRM,
        }[self._state]

    def _refresh_view(self) -> None:
        rows = self._rows()
        self.query_one("#profiles-rows", Static).update(
            render_menu_rows(rows, self._cursor) if rows else ""
        )
        self.query_one("#profiles-help", Static).update(self._help_text())
        self.query_one("#profiles-hint", Static).update(self._hint())

    # --- navigation --------------------------------------------------------

    def action_row_up(self) -> None:
        self._move(-1)

    def action_row_down(self) -> None:
        self._move(1)

    def _move(self, direction: int) -> None:
        if self._state == "edit":
            # Arrows walk the form's fields; Input itself only handles left/right.
            if direction < 0:
                self.focus_previous()
            else:
                self.focus_next()
            return
        rows = self._rows()
        if not rows:
            return
        self._cursor = (self._cursor + direction) % len(rows)
        self._refresh_view()

    async def action_select_row(self) -> None:
        if self._state == "list":
            await self._select_list_row()
        elif self._state == "actions":
            await self._run_action(self.ACTIONS[self._cursor])
        elif self._state == "confirm":
            await self._resolve_confirm(self.CONFIRM[self._cursor])

    async def action_back(self) -> None:
        if self._state == "list":
            self.app.pop_screen()
        elif self._state == "edit":
            target = self._target
            await self._close_editor()
            if target is None:
                self._to_list()
            else:
                self._to_actions()
        else:
            self._to_list(focus=self._target)

    # --- state transitions -------------------------------------------------

    async def _select_list_row(self) -> None:
        if self._cursor >= len(self._profiles.items):
            await self._open_editor(None)
            return
        self._target = self._profiles.items[self._cursor].name
        self._to_actions()

    def _to_actions(self) -> None:
        self._state = "actions"
        self._cursor = 0
        self._refresh_view()

    def _to_list(self, focus: str | None = None) -> None:
        self._state = "list"
        self._target = None
        names = self._profiles.names
        self._cursor = names.index(focus) if focus in names else 0
        self._refresh_view()

    async def _run_action(self, action: str) -> None:
        if action == "Back":
            self._to_list(focus=self._target)
        elif action == "Edit":
            await self._open_editor(self._target)
        elif action == "Activate":
            await self._activate()
        elif action == "Delete":
            self._state = "confirm"
            self._cursor = len(self.CONFIRM) - 1  # default to Cancel
            self._refresh_view()

    async def _resolve_confirm(self, choice: str) -> None:
        if choice != "Confirm delete":
            self._to_actions()
            return
        target = self._target
        try:
            # Profiles.delete owns the active-delete rule; do not restate it here.
            candidate = self._profiles.delete(target)
        except ProfileError as error:
            self.notify(str(error), severity="error")
            self._to_actions()
            return
        if await self._persist(candidate):
            self._to_list()

    async def _activate(self) -> None:
        try:
            candidate = self._profiles.activate(self._target)
        except ProfileError as error:
            self.notify(str(error), severity="error")
            return
        if await self._persist(candidate, activated=True):
            self._to_list(focus=self._target)

    # --- editing -----------------------------------------------------------

    async def _open_editor(self, target: str | None) -> None:
        """Mount the four-field form for ``target``, or for a new profile."""
        self._target = target
        self._state = "edit"
        profile = self._profiles.get(target) if target is not None else None
        editor = self.query_one("#profiles-editor", Vertical)
        await editor.remove_children()
        widgets: list[Static | _FieldInput] = []
        for widget_id, label, attribute in self.FIELDS:
            widgets.append(Static(label))
            value = getattr(profile, attribute) if profile is not None else ""
            widgets.append(_FieldInput(value, id=widget_id))
        await editor.mount(*widgets)
        self._refresh_view()
        editor.query_one(f"#{self.FIELDS[0][0]}", _FieldInput).focus()

    async def _close_editor(self) -> None:
        await self.query_one("#profiles-editor", Vertical).remove_children()

    async def on_field_commit(self, _event: FieldCommit) -> None:
        if self._state != "edit":
            return
        values = {
            attribute: self.query_one(f"#{widget_id}", _FieldInput).value.strip()
            for widget_id, _label, attribute in self.FIELDS
        }
        try:
            # One construction point: Profile owns every field rule, and its
            # message is what the user sees.
            profile = Profile(**values)
            candidate = (
                self._profiles.replace(self._target, profile)
                if self._target is not None
                else self._profiles.add(profile)
            )
        except ProfileError as error:
            self.notify(str(error), severity="error")
            return  # the editor stays open with the rejected input
        if await self._persist(candidate):
            await self._close_editor()
            self._to_list(focus=profile.name)

    async def on_field_cancel(self, _event: FieldCancel) -> None:
        if self._state != "edit":
            return
        target = self._target
        await self._close_editor()
        if target is None:
            self._to_list()
        else:
            self._to_actions()

    # --- persistence -------------------------------------------------------

    async def _persist(self, candidate: Profiles, *, activated: bool = False) -> bool:
        """Store ``candidate``, adopting it locally only once it is saved."""
        try:
            await self._commit(candidate, activated=activated)
        except (OSError, ProfileError) as error:
            self.notify(str(error), severity="error")
            return False
        self._profiles = candidate
        return True


class QuitScreen(ModalScreen[bool]):
    """Confirm an ordinary quit; a second Ctrl+C confirms immediately.

    The screen decides nothing about the application's lifetime: it returns
    ``True`` to exit and ``False`` to stay, and ``ChatApp`` owns the exit.

    Its ``Ctrl+C`` is declared ``priority=True`` deliberately. Every other
    ``Ctrl+C`` in the app is selection-first; this one is the panic exit the
    user pressed twice, so it must win even over a selection inside the dialog.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left,up,shift+tab", "move(-1)", "Previous", show=False),
        Binding("right,down,tab", "move(1)", "Next", show=False),
        Binding("enter", "choose", "Choose", show=False),
        Binding("escape", "stay", "Stay", show=False),
        Binding(
            "ctrl+c",
            "panic_quit",
            "Quit now",
            show=False,
            priority=True,
        ),
    ]

    CHOICES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Stay", "Return to JTech CLI"),
        ("Quit", "Exit JTech CLI"),
    )

    _HINT = "Arrows/Tab choose · Enter confirm · Esc stay · Ctrl+C quit now"

    def __init__(self) -> None:
        super().__init__()
        # Index 0 is Stay: the default must never be the destructive choice.
        self._cursor = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-dialog"):
            yield Static("Quit JTech CLI?", classes="dialog-title")
            yield Static(id="quit-rows")
            yield Static(self._HINT, id="quit-hint")

    def on_mount(self) -> None:
        self._render_choices()

    def action_move(self, direction: int) -> None:
        self._cursor = (self._cursor + direction) % len(self.CHOICES)
        self._render_choices()

    def action_choose(self) -> None:
        self.dismiss(self._cursor == 1)

    def action_stay(self) -> None:
        self.dismiss(False)

    def action_panic_quit(self) -> None:
        self.dismiss(True)

    def _render_choices(self) -> None:
        self.query_one("#quit-rows", Static).update(
            render_menu_rows(self.CHOICES, self._cursor)
        )
