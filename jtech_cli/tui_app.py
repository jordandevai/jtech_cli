"""Chat application lifecycle, agent orchestration, and command policy.

The app owns everything that is app-wide: the Primary conversation and its
composer, the catalog of dispatched agents, profile lookup and resolution,
serialized command approval, settings persistence, and the workspace. One
conversation's own model/command loop lives in
:class:`~jtech_cli.tui_runtime.AutonomousRuntime`, which Primary and every
subagent instantiate identically.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from rich.console import RenderableType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Static

from jtech_cli import server_info
from jtech_cli.cmd_tools import (
    AgentDispatch,
    CmdPolicy,
    allow_rule_for,
    decide,
    duplicate_agent_keys,
)
from jtech_cli.commands import CommandContext, build_registry
from jtech_cli.config import (
    CONFIG_PATH,
    Profile,
    ProfileError,
    Profiles,
    ResolvedProfile,
    Settings,
    resolve_profile,
    save_settings,
)
from jtech_cli.llm_client import stream_reply
from jtech_cli.prompts import compose_coordinator_prompt, compose_worker_prompt
from jtech_cli.server_info import ServerInfo, fetch_server_info
from jtech_cli.session import Session
from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT, textual_theme_name
from jtech_cli.tui_runtime import (
    CONNECTION_ERROR,
    RENDER_ERROR,
    SPINNER_FRAMES,
    AgentOutcome,
    AgentRunState,
    AutonomousRuntime,
    CommandAuthorization,
    RunPhase,
    StreamReply,
)
from jtech_cli.tui_screens import (
    CmdChoice,
    CommandPrompt,
    ProfilesScreen,
    QuitScreen,
    SettingsScreen,
)
from jtech_cli.tui_widgets import (
    AgentStatus,
    AgentSummary,
    AgentTaskSummary,
    AgentWorkspace,
    InputToMultiline,
    MultilineCancel,
    MultilineSubmit,
    OutputSink,
    PlainTail,
    Transcript,
    TranscriptRecord,
    _ChatInput,
    _MultilineInput,
    render_menu_rows,
)

NO_PROFILE = "No API profile is configured — run /profiles to add one."
BUSY_GENERATING = "A reply is streaming — press Esc to stop it before changing profiles."
BUSY_TOOL_ROUND = "A tool round is running — wait for it to finish before changing profiles."
PRIMARY_AGENT_ID = "primary"
SUBAGENT_READONLY = "Read only — subagents communicate with their dispatcher."
SUBAGENT_CLEAR_BLOCKED = (
    "Subagent activity is read only; switch to Primary to clear chat."
)
AGENT_STOPPED = "Agent stopped before completing the task."

FetchServerInfo = Callable[[Profile], ServerInfo]
FetchTokenCount = Callable[[Profile, str], int | None]

#: Every non-terminal phase an agent row and its current task row can show.
#: ``stopped`` maps to ``failed``: an assignment that did not finish is not a
#: success, and Primary is told so.
_PHASE_STATUS: dict[RunPhase, AgentStatus] = {
    "starting": "running",
    "streaming": "running",
    "tool": "running",
    "command": "running",
    "waiting": "waiting",
    "completed": "completed",
    "failed": "failed",
    "stopped": "failed",
}


class _DispatchRejected(Exception):
    """One dispatch call refused before any task or model request exists."""


@dataclass(slots=True)
class _ManagedAgent:
    """One dispatched agent's identity, conversation, and current assignment.

    The profile *name* is retained, never a resolved credential-bearing value:
    each new task re-resolves that name and pins its current endpoint, model,
    and credential, exactly as a Primary turn does.
    """

    agent_key: str
    agent_label: str
    profile_name: str
    session: Session
    transcript: Transcript
    tasks: list[AgentTaskSummary]
    runtime: AutonomousRuntime | None = None
    active_task_id: str | None = None


class ChatApp(App):
    """Full-screen chat app with injected network boundaries."""

    CSS_PATH = Path(__file__).parent / "resources" / "styles" / "tui.css"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "contextual_ctrl_c", "Copy / Clear / Quit"),
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
        no_discover: bool = False,
        stream_reply_fn: StreamReply | None = None,
        fetch_server_info_fn: FetchServerInfo | None = None,
        fetch_token_count_fn: FetchTokenCount | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.session = session
        self.server = server
        self.config_path = config_path
        self._no_discover = no_discover
        self._stream_reply_fn = stream_reply if stream_reply_fn is None else stream_reply_fn
        self._fetch_server_info_fn = (
            fetch_server_info
            if fetch_server_info_fn is None
            else fetch_server_info_fn
        )
        self._fetch_token_count_fn = (
            server_info.fetch_token_count
            if fetch_token_count_fn is None
            else fetch_token_count_fn
        )
        self.cmd = cmd if cmd is not None else CmdPolicy()
        self.settings.cmd_mode = self.cmd.mode
        self._project_root = Path.cwd().resolve()
        self._primary_runtime: AutonomousRuntime | None = None
        # Insertion ordered by first dispatch. Primary is not a member:
        # its session, transcript, and composer belong to the app itself.
        self._agents: dict[str, _ManagedAgent] = {}
        # One approval modal at a time, and the policy re-read inside the
        # lock: a rule another agent just saved must decide this request too.
        self._approval_lock = asyncio.Lock()
        self._multiline_textarea: _MultilineInput | None = None
        # ``None`` is the resolved value for a cancel, so it is distinct from a
        # submitted empty editor: only the latter is content the user chose.
        self._multiline_future: asyncio.Future[str | None] | None = None
        self._prompt_tokens = 0
        self._queue: list[str] = []
        self._queue_lines: list[PlainTail] = []
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
            open_profiles=self._open_profiles,
            switch_profile=self._switch_profile,
            effective_prompt=self._primary_system_prompt,
        )
        self.commands = build_registry(self.ctx)
        self._suggestions: list[tuple[str, str]] = []
        self._suggestion_index = 0
        self._suggestion_navigated = False
        # Ownership of the accepted Primary turn, not a limit or a round count:
        # queued messages drain through nested ``_send_message()`` calls, so an
        # inner turn must not paint ``idle`` while its outer drain is still
        # running.
        self._primary_turn_depth = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            # The Primary transcript is created here, not by the workspace, so
            # ``#chat`` stays the stable id every existing selector uses.
            yield AgentWorkspace(
                AgentSummary(PRIMARY_AGENT_ID, "Primary", "idle"),
                Transcript(id="chat", classes="agent-activity"),
                id="agent-workspace",
            )
            yield Static(id="suggestions")
            yield _ChatInput(
                id="input",
                placeholder="Message… (Enter send · Shift+Enter newline · / commands)",
            )
            readonly = Static(SUBAGENT_READONLY, id="subagent-readonly")
            readonly.display = False
            yield readonly
            yield Static(id="status")

    @property
    def _workspace(self) -> AgentWorkspace:
        """The mounted activity/sidebar split."""
        return self.query_one("#agent-workspace", AgentWorkspace)

    @property
    def viewing_primary(self) -> bool:
        """Whether the displayed activity stream is Primary's.

        Derived from the workspace rather than mirrored into a second boolean,
        so a selection change has exactly one place to be recorded.
        """
        workspace = self._workspace
        return workspace.selected_agent_id == workspace.primary_agent_id

    async def add_agent_view(
        self,
        summary: AgentSummary,
        records: Sequence[TranscriptRecord] = (),
    ) -> Transcript:
        """Register one subagent's presentation and return its transcript.

        The presentation seam a future orchestration runtime calls. It does no
        scheduling, persistence, network I/O, or thread marshalling: callers
        invoke it on Textual's event loop and retain the returned `Transcript`
        as that agent's activity stream for the workspace lifetime.

        Args:
            summary: The new agent's presentation data.
            records: Completed activity to seed.

        Raises:
            ValueError: if ``summary`` is invalid or already registered.
        """
        return await self._workspace.add_agent(summary, records)

    def update_agent_view(self, summary: AgentSummary) -> None:
        """Replace one registered agent's label, status, or tasks.

        The presentation seam a future orchestration runtime calls on Textual's
        event loop. It repaints one sidebar row and nothing else — no
        transcript is replaced, reloaded, scrolled, shown, or hidden, and the
        user's selection is never changed.

        Raises:
            ValueError: if ``summary`` is invalid.
            KeyError: if its agent id is not registered.
        """
        self._workspace.update_agent(summary)

    def on_agent_workspace_agent_selected(
        self, event: AgentWorkspace.AgentSelected
    ) -> None:
        """Follow the visible activity stream with the composer's visibility.

        Presentation only: the composer still targets Primary whatever is shown,
        so this changes what is displayed and no runtime state at all.
        """
        self._show_primary_composer(
            event.agent_id == event.workspace.primary_agent_id
        )

    def _show_primary_composer(self, show: bool) -> None:
        """Show the Primary composer, or the read-only subagent notice.

        Hiding is display-only: the input value and selection, the suggestion
        data, the multi-line text and its unresolved future, the queue, and the
        Primary session are all left exactly as they were, so returning to
        Primary restores the draft the user left behind. A disabled ``Input`` is
        deliberately not used — it still looks like a destination.
        """
        readonly = self.query_one("#subagent-readonly", Static)
        suggestions = self.query_one("#suggestions", Static)
        chat_input = self.query_one("#input", _ChatInput)
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
        self._update_suggestions()

    def _set_primary_agent_status(self, status: AgentStatus) -> None:
        """Write one Primary status through the workspace's own summary.

        The workspace stays the single source of truth for the summary, so an
        orchestration-supplied Primary label or task list survives a local
        status change with no synchronization code.
        """
        workspace = self._workspace
        summary = workspace.summary_for(PRIMARY_AGENT_ID)
        workspace.update_agent(dataclasses.replace(summary, status=status))

    async def on_mount(self) -> None:
        self.register_theme(JTECH_DARK)
        self.register_theme(JTECH_LIGHT)
        self.theme = textual_theme_name(self.settings.theme)
        # Replayed history becomes completed content in one rendering: it costs
        # no label or Markdown widget per stored message, however long it is.
        history: list[TranscriptRecord] = []
        for msg in self.session.messages:
            if msg.get("_debug_only") and self.settings.debug_level != "system":
                continue
            history.append(
                TranscriptRecord(role=msg["role"], content=msg["content"])
            )
        chat = self.query_one("#chat", Transcript)
        chat.load(history)
        if history:
            chat.scroll_end(animate=False)
        profile = self.settings.active_profile
        if profile is None:
            self.push_message("system", NO_PROFILE)
        if self.settings.prompt_notice:
            self.push_message("system", self.settings.prompt_notice)
        self.query_one("#suggestions", Static).display = False
        self._render_status()
        self._focus_input()
        if profile is not None and self.session.messages and self.server.context_length:
            await self._init_token_count(profile)
        if not self._no_discover and profile is not None:
            self.call_later(self._discover_server, profile)

    async def _discover_server(self, profile: Profile) -> None:
        """Refresh metadata for ``profile`` without downgrading known values."""
        try:
            info = await asyncio.to_thread(self._fetch_server_info_fn, profile)
        except ProfileError as error:
            self.push_message("system", str(error))
            return
        if self.settings.active_profile != profile:
            # The endpoint changed while this probe was in flight. A late answer
            # from the previous one must not describe the current one.
            return
        if not info.known:
            detail = f" ({info.error})" if info.error else ""
            self.push_message(
                "system",
                f"Could not reach {profile.base_url}{detail} — model and context "
                "info unavailable. Check the endpoint in /profiles.",
            )
            return
        self.server.models = info.models
        self.server.context_length = info.context_length
        self.server.error = None
        self._render_status()
        if self.session.messages and self.server.context_length:
            await self._init_token_count(profile)

    def push_message(self, role: str, text: str) -> None:
        """Add one already-complete message to the visible transcript."""
        self.query_one("#chat", Transcript).append(
            TranscriptRecord(role=role, content=text)
        )

    def _focus_input(self) -> None:
        inputs = self.query(_ChatInput)
        if inputs:
            inputs[0].focus()

    async def _init_token_count(self, profile: Profile) -> None:
        """Count session tokens for ``profile`` so the footer is accurate.

        The count describes one endpoint's tokenizer, so a result that arrives
        after the user switched profiles is discarded rather than applied to the
        new one — the same staleness rule discovery uses.
        """
        history = self.session.messages_with_system("")
        text = " ".join(message["content"] for message in history)
        if not text:
            return
        try:
            count = await asyncio.to_thread(self._fetch_token_count_fn, profile, text)
        except ProfileError as error:
            if self.settings.active_profile == profile:
                self.push_message("system", str(error))
            return
        if count and self.settings.active_profile == profile:
            self._prompt_tokens = count
            self._render_status()

    def on_input_changed(self, _event: Input.Changed) -> None:
        self._update_suggestions()

    def _update_suggestions(self) -> None:
        value = self.query_one("#input", _ChatInput).value
        if not value.startswith("/") or " " in value:
            self._hide_suggestions()
            return
        matches = self.commands.completions(value)
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
        if self._suggestions and isinstance(self.focused, _ChatInput):
            self._cycle_suggestion(-1)
        else:
            self._recall_queued()

    def action_suggestion_down(self) -> None:
        self._cycle_suggestion(1)

    def _recall_queued(self) -> None:
        """Recall the next queued message without submitting it."""
        if not self._queue or not isinstance(self.focused, _ChatInput):
            return
        inp = self.query_one("#input", _ChatInput)
        if inp.value.strip():
            return
        text = self._queue[0]
        self._remove_queue_entry(0)
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
        return self._suggestion_navigated

    def accepted_suggestion(self) -> str | None:
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
        profile = self.settings.active_profile
        if profile is not None:
            suffix = " (override)" if self.settings.profile_is_overridden else ""
            parts.append(f"profile: {profile.name}{suffix}")
            parts.append(profile.base_url)
            model = profile.model or self.server.model
            if model:
                parts.append(f"model: {model}")
        if self.server.context_length:
            if self._prompt_tokens:
                total = self.server.context_length
                remaining_pct = max(0, (1 - self._prompt_tokens / total) * 100)
                parts.append(
                    f"ctx {self._prompt_tokens // 1000}k"
                    f"/{total // 1000}k ({remaining_pct:.0f}% left)"
                )
            else:
                parts.append(f"ctx {self.server.context_length}")
        if self._queue:
            parts.append(f"queue: {len(self._queue)}")
        self.query_one("#status", Static).update("  ·  ".join(parts))

    def _record_message(
        self,
        role: str,
        content: str,
        *,
        include_in_context: bool = True,
        debug_only: bool = False,
        model_role: str | None = None,
        model_content: str | None = None,
    ) -> None:
        """Persist one message, surfacing an I/O failure in the transcript.

        Presenting the failure is UI work, so it lives here rather than in
        ``Session``. The message stays in memory when the append fails — a
        save failure is worth saying out loud, but it is not a reason to drop
        the live conversation — and the warning never joins model context.
        """
        try:
            self.session.add(
                role,
                content,
                include_in_context=include_in_context,
                debug_only=debug_only,
                model_role=model_role,
                model_content=model_content,
            )
        except OSError as error:
            self.push_message("system", f"Could not save history: {error}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.query_one("#input", _ChatInput).value = ""
        if not value:
            self._focus_input()
            return
        if value == "'''":
            self.run_worker(self._open_multiline_message())
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
        if result is not None:
            self.run_worker(result)

    async def _open_multiline_message(self) -> None:
        """Compose a chat message in the multi-line editor, then send it."""
        content = await self._enter_multiline()
        if content is None:
            return
        message = content.strip()
        if message:
            await self._send_message(message)

    def on_input_to_multiline(self, _event: InputToMultiline) -> None:
        """Take over the promoted draft, synchronously, before anything else can.

        The editor is mounted here rather than inside the worker: a worker only
        starts on a later pass of the message pump, and a burst of terminal
        events could reach this handler again first and mount a second editor
        under the same id.
        """
        inp = self.query_one("#input", _ChatInput)
        if self._multiline_textarea is not None:
            # Reachable when an editor is open but hidden behind a selected
            # subagent. Decline the promotion rather than mount a second
            # editor, but release the flags so the composer still works.
            inp.cancel_promotion()
            return
        draft = inp.take_promotion()
        future = self._open_multiline_editor(draft.value, draft.cursor_offset)
        self.run_worker(self._switch_to_multiline(future))
        if draft.submit:
            # Enter arrived while the draft was still in the single-line input.
            # The user asked to send; the racing editor must not change that.
            self._resolve_multiline(draft.value)

    async def _switch_to_multiline(
        self, future: asyncio.Future[str | None]
    ) -> None:
        """Send whatever the promoted editor produces, unless it was cancelled."""
        content = await future
        if content is None:
            return
        message = content.strip()
        if message:
            await self._send_message(message)

    def _enqueue(self, text: str) -> None:
        """Queue a message while a reply or tool round is in flight.

        The notice is literal app text rather than model Markdown, so it is a
        plain live entry: it can be withdrawn again by recall or by draining.
        """
        chat = self.query_one("#chat", Transcript)
        self._queue.append(text)
        self._queue_lines.append(chat.begin_plain("system", f"Queued: {text}"))
        self._render_status()

    def _remove_queue_entry(self, index: int) -> None:
        """Drop a queued message and its transient chat line."""
        self._queue.pop(index)
        self.query_one("#chat", Transcript).remove(self._queue_lines.pop(index))
        self._render_status()

    async def _enter_multiline(
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
        return await self._open_multiline_editor(prefill, cursor_offset)

    def _open_multiline_editor(
        self, prefill: str, cursor_offset: int | None
    ) -> asyncio.Future[str | None]:
        """Mount the editor and claim ownership in one synchronous step.

        Returning the future rather than reading ``self._multiline_future``
        later matters: ``_resolve_multiline()`` clears that attribute, and a
        submit can land before the awaiting worker has even started.
        """
        if self._multiline_textarea is not None:
            raise RuntimeError("A multi-line editor is already open.")
        future: asyncio.Future[str | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._multiline_future = future
        inp = self.query_one("#input", _ChatInput)
        textarea = _MultilineInput(classes="multiline", id="multiline-input")
        inp.display = False
        self.mount(textarea)
        self._multiline_textarea = textarea
        if prefill:
            textarea.text = prefill
        offset = len(prefill) if cursor_offset is None else cursor_offset
        offset = max(0, min(offset, len(prefill)))
        textarea.move_cursor(textarea.document.get_location_from_index(offset))
        textarea.focus()
        return future

    def on_multiline_submit(self, _event: MultilineSubmit) -> None:
        textarea = self._multiline_textarea
        if textarea is None:
            return
        self._resolve_multiline(textarea.text)

    def on_multiline_cancel(self, _event: MultilineCancel) -> None:
        self._resolve_multiline(None)

    def _resolve_multiline(self, content: str | None) -> None:
        textarea = self._multiline_textarea
        if textarea is None:
            return
        textarea.remove()
        self._multiline_textarea = None
        inp = self.query_one("#input", _ChatInput)
        inp.display = True
        inp.focus()
        future = self._multiline_future
        self._multiline_future = None
        if future is not None and not future.done():
            future.set_result(content)

    async def _send_message(self, content: str) -> None:
        """Run one accepted Primary turn, or queue the message behind one.

        The accepted turn owns the Primary ``running``/``idle`` status. Depth,
        not a flag, because ``_stream_reply()`` drains the queue through nested
        calls to this method: only the outermost accepted turn may report the
        agent idle again, and every exit — a provider error, a rendering error,
        cancellation, a command round, an agent batch, a nudge, or a drained
        queue — releases exactly one acquired depth.
        """
        content = content.strip()
        if not content:
            return
        if self._primary_runtime is not None:
            self._enqueue(content)
            return
        self._primary_turn_depth += 1
        if self._primary_turn_depth == 1:
            self._set_primary_agent_status("running")
        try:
            self._record_message("user", content)
            self.push_message("user", content)
            await self._stream_reply()
        finally:
            self._primary_turn_depth -= 1
            if self._primary_turn_depth == 0:
                self._set_primary_agent_status("idle")

    def _resolve_turn_profile(self) -> ResolvedProfile:
        """Pin one endpoint, model, and credential for this whole user turn.

        Raises:
            ProfileError: if no profile is selected, no model resolves, or the
                credential is unavailable — before any provider thread starts.
        """
        profile = self.settings.active_profile
        if profile is None:
            raise ProfileError(NO_PROFILE)
        return resolve_profile(
            profile, discovered_model=self.server.model, environ=os.environ
        )

    async def _stream_reply(self) -> None:
        """Run one user turn against a single immutable profile snapshot.

        Every completion in the turn — the first answer, each command
        continuation, each agent batch, and each nudge — is handed the same
        ``ResolvedProfile``, so an endpoint change can never take effect
        halfway through a turn.
        """
        try:
            profile = self._resolve_turn_profile()
        except ProfileError as error:
            self.push_message("system", str(error))
            profile = None
        if profile is not None:
            runtime = AutonomousRuntime(
                self._primary_run_state(profile),
                host=self,
                stream_reply_fn=self._stream_reply_fn,
                cmd_policy=self.cmd,
                project_root=self._project_root,
            )
            self._primary_runtime = runtime
            try:
                await runtime.run()
            finally:
                self._primary_runtime = None
        while self._queue:
            text = self._queue[0]
            self._remove_queue_entry(0)
            await self._send_message(text)

    def _primary_run_state(self, profile: ResolvedProfile) -> AgentRunState:
        """The run state for one accepted Primary turn.

        The footer figures are seeded from the app, not zeroed: a new turn
        inherits what the last one measured until the provider reports again.
        """
        state = AgentRunState(
            agent_key=PRIMARY_AGENT_ID,
            agent_label=self._workspace.summary_for(PRIMARY_AGENT_ID).label,
            kind="primary",
            session=self.session,
            transcript=self.query_one("#chat", Transcript),
            profile=profile,
            temperature=self.settings.temperature,
            system_prompt=self._primary_system_prompt,
            reasoning_mode=lambda: self.settings.reasoning,
            debug_level=lambda: self.settings.debug_level,
        )
        state.prompt_tokens = self._prompt_tokens
        state.last_reply = self.ctx.last_reply
        return state

    # -------------------------------------------------------------- prompts

    def _primary_system_prompt(self) -> str:
        """The prompt the next Primary completion will actually carry."""
        active = self.settings.active_profile
        return compose_coordinator_prompt(
            self.settings.effective_system_prompt(),
            profile_names=[
                profile.name for profile in self._available_dispatch_profiles()
            ],
            active_profile_name=active.name if active is not None else None,
        )

    def _worker_system_prompt(self) -> str:
        """The prompt every subagent completion carries."""
        return compose_worker_prompt(self.settings.effective_system_prompt())

    # ------------------------------------------------------------- profiles

    def _available_dispatch_profiles(self) -> tuple[Profile, ...]:
        """Every profile an agent may be dispatched to, in advertised order.

        A session-only ``--base-url`` override comes first and hides a
        configured profile of the same name, so one name never advertises two
        endpoints.
        """
        configured = self.settings.profiles.items
        override = self.settings.profile_override
        if override is None:
            return tuple(configured)
        return (
            override,
            *(item for item in configured if item.name != override.name),
        )

    def _profile_for_dispatch(self, name: str) -> Profile:
        """The profile called ``name``, with the same override precedence.

        Raises:
            ProfileError: if no available profile has that name.
        """
        override = self.settings.profile_override
        if override is not None and override.name == name:
            return override
        return self.settings.profiles.get(name)

    async def _resolve_agent_profile(self, profile: Profile) -> ResolvedProfile:
        """Pin ``profile`` to one model and credential for one agent task.

        Discovery runs off the event loop and only when the profile configures
        no model. It never writes ``self.server`` or the Primary footer: a
        subagent's endpoint is not the one the status bar describes.

        Raises:
            ProfileError: if the credential is unavailable, the endpoint cannot
                be reached, or no single model resolves. There is no
                active-profile or model fallback.
        """
        if profile.model:
            return resolve_profile(profile, discovered_model=None, environ=os.environ)
        active = self.settings.active_profile
        if active is not None and profile == active and self.server.model:
            return resolve_profile(
                profile, discovered_model=self.server.model, environ=os.environ
            )
        info = await asyncio.to_thread(self._fetch_server_info_fn, profile)
        if not info.models and info.error:
            raise ProfileError(
                f"Profile {profile.name!r}: {profile.base_url} could not be "
                f"reached to discover its model ({info.error})"
            )
        return resolve_profile(
            profile, discovered_model=info.model, environ=os.environ
        )

    # ------------------------------------------------------------- dispatch

    def _new_task_id(self) -> str:
        """A fresh opaque task id.

        A method so a test can make ids deterministic without an id factory in
        the public constructor. No user or model value ever enters it, and it
        is never used as a DOM id or a filesystem path.
        """
        return "task-" + uuid.uuid4().hex

    async def dispatch_agents(
        self, run: AgentRunState, calls: tuple[AgentDispatch, ...]
    ) -> tuple[AgentOutcome, ...]:
        """Run one whole dispatch batch and return its results in call order.

        Distinct keys start together and run concurrently; one failure never
        cancels a sibling; and the results are appended to the coordinator in
        the order it wrote the calls, whatever order they finish in, so
        provider timing cannot make its context nondeterministic.

        Raises:
            ValueError: if a non-Primary run reaches here, or the batch repeats
                an agent key. The runtime refuses both first; this is the
                authority boundary saying so again, before anything is created.
        """
        if run.kind != "primary":
            raise ValueError("only Primary may dispatch agents")
        duplicates = duplicate_agent_keys(calls)
        if duplicates:
            raise ValueError(
                f"one response cannot dispatch {', '.join(duplicates)} twice"
            )

        outcomes: list[AgentOutcome | None] = [None] * len(calls)
        started: list[tuple[int, _ManagedAgent, AgentDispatch, str]] = []
        for index, call in enumerate(calls):
            task_id = self._new_task_id()
            try:
                managed = await self._begin_agent_task(call, task_id)
            except _DispatchRejected as rejection:
                outcomes[index] = self._setup_outcome(call, task_id, str(rejection))
                continue
            except Exception as error:  # noqa: BLE001 - one call, not the batch
                # Setting one agent up is that call's own work. An unexpected
                # failure here must not take its siblings down with it, so it
                # becomes that call's failed result like any other.
                outcomes[index] = self._setup_outcome(
                    call, task_id, f"{type(error).__name__}: {error}"
                )
                continue
            started.append((index, managed, call, task_id))

        self._set_run_phase(run, "waiting")
        try:
            results = await asyncio.gather(
                *(
                    self._dispatch_one(managed, call, task_id)
                    for _, managed, call, task_id in started
                ),
                return_exceptions=True,
            )
        finally:
            self._set_run_phase(run, "tool")

        for (index, managed, call, task_id), result in zip(
            started, results, strict=True
        ):
            if isinstance(result, asyncio.CancelledError):
                # Shutdown, not a task failure: it must unwind, not be reported
                # to the model as a completed batch.
                raise result
            if isinstance(result, BaseException):
                message = f"{type(result).__name__}: {result}"
                managed.transcript.append(
                    TranscriptRecord(role="system", content=message, error=True)
                )
                self._set_agent_task_status(call.agent_key, task_id, "failed")
                outcomes[index] = AgentOutcome(
                    agent_key=call.agent_key,
                    agent_label=managed.agent_label,
                    task_id=task_id,
                    task_label=call.task_label,
                    status="failed",
                    content=message,
                )
                continue
            outcomes[index] = result
        # Every call has an outcome by construction: it was rejected, its setup
        # failed, it returned one, or its exception was converted above.
        # Checked, not filtered — a filter would silently hand the coordinator
        # a shorter batch than it dispatched — and checked at runtime rather
        # than asserted, because assertions vanish under `python -O`.
        settled: list[AgentOutcome] = []
        for index, outcome in enumerate(outcomes):
            if outcome is None:
                raise RuntimeError(
                    f"dispatch call {index} for agent {calls[index].agent_key!r} "
                    "produced no outcome"
                )
            settled.append(outcome)
        return tuple(settled)

    def _setup_outcome(
        self, call: AgentDispatch, task_id: str, content: str
    ) -> AgentOutcome:
        """Report one call that never reached a runtime, failing any row it made.

        A rejection creates nothing, so there is usually no row to correct. An
        unexpected setup failure can leave a task already committed to a live
        agent: that row is marked failed here rather than left running for the
        rest of the session.

        The result carries the identity the *call* asked for, so the model can
        match every outcome to the call it wrote. A label conflict is exactly
        the case where the existing agent's label differs, and answering a
        rejected ``Renamed`` call with ``Coder`` would hide which call failed.
        Where setup got far enough to touch an agent, the guards above have
        already proved the two labels equal.
        """
        managed = self._agents.get(call.agent_key)
        if managed is not None and any(
            task.task_id == task_id for task in managed.tasks
        ):
            self._set_agent_task_status(call.agent_key, task_id, "failed")
        return AgentOutcome(
            agent_key=call.agent_key,
            agent_label=call.agent_label,
            task_id=task_id,
            task_label=call.task_label,
            status="failed",
            content=content,
        )

    async def _begin_agent_task(
        self, call: AgentDispatch, task_id: str
    ) -> _ManagedAgent:
        """Create or continue one agent and append its new running task.

        A new key gets one in-memory session, one workspace view, and the task
        recorded exactly once — the transcript record seeded into the view is
        the presentation of that same message, not a second model message.

        Raises:
            _DispatchRejected: if the key exists with a different label or
                profile, or is already running a task. Nothing is mutated.
        """
        task = AgentTaskSummary(task_id, call.task_label, "running")
        managed = self._agents.get(call.agent_key)
        if managed is None:
            session = Session(persist=False)
            transcript = await self.add_agent_view(
                AgentSummary(call.agent_key, call.agent_label, "running", (task,)),
                (TranscriptRecord(role="user", content=call.task),),
            )
            # Same rule as a continuation: the assignment joins the worker's
            # conversation only once its presentation exists. The seeded
            # transcript record is that message's presentation, not a second
            # model message.
            session.add("user", call.task)
            managed = _ManagedAgent(
                agent_key=call.agent_key,
                agent_label=call.agent_label,
                profile_name=call.profile_name,
                session=session,
                transcript=transcript,
                tasks=[task],
            )
            self._agents[call.agent_key] = managed
            return managed
        if managed.agent_label != call.agent_label:
            raise _DispatchRejected(
                f"Agent {call.agent_key!r} already exists with the label "
                f"{managed.agent_label!r}. An agent key keeps its label for the "
                "session; use a new key for a differently labelled agent."
            )
        if managed.profile_name != call.profile_name:
            raise _DispatchRejected(
                f"Agent {call.agent_key!r} already exists on profile "
                f"{managed.profile_name!r}. An agent key keeps its profile for "
                "the session; use a new key to work on another profile."
            )
        if managed.runtime is not None:
            raise _DispatchRejected(
                f"Agent {call.agent_key!r} is still working on its current task. "
                "Wait for its result before sending it another one."
            )
        # Presentation first, model context last. The sidebar and transcript
        # writes are the fallible ones, so the task row is committed only after
        # the sidebar accepted it and the assignment joins the worker's
        # conversation only after both succeeded: a setup that fails must never
        # leave an unanswered instruction in the context of the agent's next
        # task.
        self.update_agent_view(
            AgentSummary(
                call.agent_key,
                managed.agent_label,
                "running",
                (*managed.tasks, task),
            )
        )
        managed.tasks.append(task)
        managed.transcript.append(TranscriptRecord(role="user", content=call.task))
        managed.session.add("user", call.task)
        return managed

    async def _dispatch_one(
        self, managed: _ManagedAgent, call: AgentDispatch, task_id: str
    ) -> AgentOutcome:
        """Resolve one task's profile, run it, and map its typed outcome.

        A pre-stream failure is written into the worker's own transcript before
        the task is marked failed: a worker that never reached a provider still
        has to show why.
        """
        def finish(
            status: Literal["completed", "failed"], content: str
        ) -> AgentOutcome:
            """Mark this task terminal and describe it to the coordinator."""
            self._set_agent_task_status(call.agent_key, task_id, status)
            return AgentOutcome(
                agent_key=call.agent_key,
                agent_label=managed.agent_label,
                task_id=task_id,
                task_label=call.task_label,
                status=status,
                content=content,
            )

        try:
            profile = self._profile_for_dispatch(call.profile_name)
            resolved = await self._resolve_agent_profile(profile)
        except ProfileError as error:
            managed.transcript.append(
                TranscriptRecord(role="system", content=str(error), error=True)
            )
            return finish("failed", str(error))

        runtime = AutonomousRuntime(
            AgentRunState(
                agent_key=call.agent_key,
                agent_label=managed.agent_label,
                kind="subagent",
                session=managed.session,
                transcript=managed.transcript,
                profile=resolved,
                temperature=self.settings.temperature,
                system_prompt=self._worker_system_prompt,
                reasoning_mode=lambda: self.settings.reasoning,
                debug_level=lambda: self.settings.debug_level,
            ),
            host=self,
            stream_reply_fn=self._stream_reply_fn,
            cmd_policy=self.cmd,
            project_root=self._project_root,
        )
        managed.runtime = runtime
        managed.active_task_id = task_id
        try:
            result = await runtime.run()
        finally:
            managed.runtime = None
            managed.active_task_id = None
        if result.status == "completed":
            return finish("completed", result.final_text)
        if result.status == "failed":
            return finish("failed", result.error)
        return finish("failed", AGENT_STOPPED)

    def _set_agent_task_status(
        self, agent_key: str, task_id: str, status: AgentStatus
    ) -> None:
        """Repaint one agent row and exactly one of its task rows.

        Only one task of an agent runs at a time, so the agent row always shows
        the status of the task named here; earlier task rows keep the terminal
        status they finished with.

        Raises:
            KeyError: if the agent or the task is unknown. That is an internal
                inconsistency, not a condition to skip past.
        """
        managed = self._agents.get(agent_key)
        if managed is None:
            raise KeyError(f"Unknown agent key: {agent_key!r}")
        for index, task in enumerate(managed.tasks):
            if task.task_id == task_id:
                managed.tasks[index] = dataclasses.replace(task, status=status)
                break
        else:
            raise KeyError(f"Agent {agent_key!r} has no task {task_id!r}")
        self.update_agent_view(
            AgentSummary(
                agent_key, managed.agent_label, status, tuple(managed.tasks)
            )
        )

    # ------------------------------------------------------ runtime callbacks

    def _set_run_phase(self, run: AgentRunState, phase: RunPhase) -> None:
        """Move one run to ``phase`` from outside its own runtime."""
        run.phase = phase
        self.runtime_changed(run)

    def runtime_changed(self, run: AgentRunState) -> None:
        """Follow one run's observable state into the UI it owns.

        The single synchronization point: no timer polls a runtime field, and
        nothing else reads one to paint with.
        """
        if not self.is_running:
            # The app has already left. Every runtime is still unwinding its
            # own ``finally`` — releasing the generating flag, clearing a
            # process — and those notifications land here after the widgets
            # are gone. There is nothing to paint and nothing left to read
            # what would be recorded, so this is a finished state rather than
            # a failure to report. Same rule as the unmounted live bubble a
            # ``/clear`` leaves behind mid-stream.
            return
        if run.kind == "primary":
            self._prompt_tokens = run.prompt_tokens
            self.ctx.last_reply = run.last_reply
            self._render_status(
                running="running command…" if run.phase == "command" else None
            )
            wanted: AgentStatus | None = None
            if run.phase == "waiting":
                wanted = "waiting"
            elif self._primary_turn_depth:
                # The accepted turn still owns the transition to idle, so a
                # finished phase never paints it here.
                wanted = "running"
            if (
                wanted is not None
                and self._workspace.summary_for(PRIMARY_AGENT_ID).status != wanted
            ):
                # Every phase change reaches here; only a real transition is
                # worth a sidebar repaint.
                self._set_primary_agent_status(wanted)
            return
        managed = self._agents[run.agent_key]
        task_id = managed.active_task_id
        if task_id is None:
            raise KeyError(
                f"Agent {run.agent_key!r} reported {run.phase!r} with no active task"
            )
        # The task the runtime is actually working on, never the last row of
        # the list: a continuation appends, and earlier rows keep their own
        # terminal status.
        self._set_agent_task_status(run.agent_key, task_id, _PHASE_STATUS[run.phase])

    # ------------------------------------------------------------- approvals

    async def authorize_command(
        self, run: AgentRunState, command: str
    ) -> CommandAuthorization:
        """Decide one command for ``run`` under the live global policy.

        Serialized for every runtime: one approval modal exists at a time, and
        the decision is re-evaluated after the lock is acquired, because the
        agent ahead in the queue may have saved an allow rule that covers this
        command too.
        """
        async with self._approval_lock:
            decision = decide(command, self.cmd, self._project_root)
            if decision.action == "run":
                return CommandAuthorization("run")
            if decision.action == "blocked":
                return CommandAuthorization("blocked", decision.reason)
            self._set_run_phase(run, "waiting")
            try:
                choice = await self.push_screen_wait(
                    CommandPrompt(command, decision.reason, requester=run.agent_label)
                )
            finally:
                self._set_run_phase(run, "tool")
            if choice is CmdChoice.DECLINE:
                return CommandAuthorization("declined", "declined by the user")
            if choice is CmdChoice.ALWAYS:
                self._add_allow_rule(allow_rule_for(command, self.cmd.allow), run)
            return CommandAuthorization("run")

    def _add_allow_rule(self, rule: str | None, run: AgentRunState) -> None:
        """Persist one always-allow rule, reporting it to the run that earned it."""
        if not rule or rule in self.cmd.allow:
            return
        self.cmd.allow.append(rule)
        try:
            self._save(self.settings)
            note = f"Always-allow saved: {rule}"
        except OSError as error:
            note = f"Could not save always-allow rule: {error}"
        run.transcript.append(TranscriptRecord(role="system", content=note))

    # ------------------------------------------------------------ stop / busy

    def action_stop_stream(self) -> None:
        # A subagent view has no writable target and no cancellation of its own:
        # stopping the hidden Primary turn from here would be invisible work.
        # The visible read-only notice is the documented explanation.
        if not self.viewing_primary:
            return
        runtime = self._primary_runtime
        if runtime is None or not (
            runtime.state.generating or runtime.state.running_proc is not None
        ):
            self._hide_suggestions()
            return
        runtime.request_stop()

    @property
    def _generating(self) -> bool:
        """Whether the Primary turn is consuming a provider stream right now.

        Derived from the run that owns the flag rather than mirrored into a
        second boolean, so there is exactly one place it can be true.
        """
        runtime = self._primary_runtime
        return runtime is not None and runtime.state.generating

    @property
    def _tool_rounds_active(self) -> bool:
        """Whether the Primary turn is between completions, running its tools."""
        runtime = self._primary_runtime
        return runtime is not None and runtime.state.tool_rounds_active

    def _busy(self) -> str | None:
        """Why a profile change is refused right now, or ``None`` when idle.

        The current turn must keep one endpoint/model/credential identity, so a
        switch that is visible in the footer but does not affect the running
        turn is refused outright rather than half-applied. A live agent batch
        is a tool round: Primary stays in it until every result is back.
        """
        runtime = self._primary_runtime
        if runtime is None:
            return None
        if runtime.state.generating:
            return BUSY_GENERATING
        return BUSY_TOOL_ROUND

    def _save(self, settings: Settings) -> None:
        """Write ``settings`` with the complete live command policy."""
        self.cmd.mode = self.settings.cmd_mode
        save_settings(settings, self.config_path, cmd=self.cmd)

    def _open_profiles(self) -> None:
        busy = self._busy()
        if busy:
            self.push_message("system", busy)
            return
        self.push_screen(ProfilesScreen(self.settings.profiles, self._commit_profiles))

    def _adopt_profiles(
        self, candidate: Profiles, previous: Profile | None, *, activated: bool
    ) -> None:
        """Adopt a catalog that has just been persisted successfully.

        The single place the live catalog advances, so ``/profile`` and the
        modal's Activate action cannot drift apart.
        """
        self.settings.profiles = candidate
        if activated:
            # An explicit selection supersedes a --base-url/--model override,
            # but only once that selection is actually stored.
            self.settings.profile_override = None
        self._after_profile_change(previous)

    async def _commit_profiles(
        self, candidate: Profiles, *, activated: bool = False
    ) -> None:
        """Persist ``candidate``, then adopt it as the live catalog.

        Persistence comes first, so a failed save needs no live-state rollback:
        the previous catalog is still the only one anything has seen.
        ``activated`` marks the modal's Activate action, which is an explicit
        selection and therefore retires a CLI endpoint override.

        Raises:
            ProfileError: if a turn is in progress.
            OSError: if the config file cannot be written.
        """
        busy = self._busy()
        if busy:
            self.push_message("system", busy)
            raise ProfileError(busy)
        previous = self.settings.active_profile
        replaced = dataclasses.replace(self.settings, profiles=candidate)
        if activated:
            replaced = dataclasses.replace(replaced, profile_override=None)
        self._save(replaced)
        self._adopt_profiles(candidate, previous, activated=activated)

    async def _switch_profile(self, name: str) -> None:
        """Activate ``name`` and persist the selection for the next launch."""
        busy = self._busy()
        if busy:
            self.push_message("system", busy)
            return
        try:
            candidate = self.settings.profiles.activate(name)
        except ProfileError as error:
            self.push_message("system", str(error))
            return
        previous = self.settings.active_profile
        try:
            self._save(
                dataclasses.replace(
                    self.settings, profiles=candidate, profile_override=None
                )
            )
        except OSError as error:
            self.push_message("system", f"Could not save profile selection: {error}")
            return
        self._adopt_profiles(candidate, previous, activated=True)
        self.push_message("system", f"Profile: {name}")

    def _after_profile_change(self, previous: Profile | None) -> None:
        """Invalidate endpoint-derived state when the selected endpoint changed."""
        current = self.settings.active_profile
        if current == previous:
            self._render_status()
            return
        # CommandContext shares this ServerInfo, so clear it in place rather
        # than rebinding: /models and /stats read the same object.
        self.server.models = []
        self.server.context_length = None
        self.server.error = None
        self._prompt_tokens = 0
        self._render_status()
        if current is not None and not self._no_discover:
            self.call_later(self._discover_server, current)

    def _open_settings(self) -> None:
        self.push_screen(
            SettingsScreen(
                self.settings,
                self.config_path,
                self._on_settings_saved,
                cmd=self.cmd,
            )
        )

    def _switch_theme(self) -> None:
        name = textual_theme_name(self.settings.theme)
        if name != self.theme:
            self.theme = name
            # Live bubbles follow the theme through CSS; completed history is
            # rendered content, so it has to be rebuilt for the new colors —
            # in every registered stream, hidden ones included.
            self._workspace.refresh_theme()

    def _on_settings_saved(self) -> None:
        self._switch_theme()
        self._render_status()

    def _add_system(self, content: str) -> None:
        """Persist a visible runtime event with model-facing observation framing."""
        self._record_message(
            "system",
            content,
            model_role="user",
            model_content=f"[JTECH runtime event]\n{content}",
        )
        if self.settings.debug_level == "system":
            self.push_message("system", content)

    def _clear_chat(self) -> None:
        self.query_one("#chat", Transcript).clear()
        self._prompt_tokens = 0
        self._render_status()

    def action_clear_chat(self) -> None:
        if not self.viewing_primary:
            self.notify(SUBAGENT_CLEAR_BLOCKED)
            return
        self.commands.handle("/clear")

    def action_settings(self) -> None:
        self._open_settings()

    def action_contextual_ctrl_c(self) -> None:
        """Clear the chat composer, or confirm quit once it is already empty.

        Selection copy is not handled here. Textual's ``Input``, ``TextArea``,
        and ``Screen`` copy actions raise ``SkipAction`` only when there is
        nothing selected, so reaching this action *is* the evidence that no
        selection existed — re-checking it here would be a second clipboard
        path that could disagree with the first.

        A non-quit modal owns the screen when the stack is deeper than one. Its
        fields are not the chat composer, so a global shortcut must not erase an
        unsaved profile or settings edit: the confirmation opens above it and
        leaves both that modal and the suspended draft untouched.

        A selected subagent hides the Primary composer for the same reason: a
        draft the user cannot see must not be cleared by a key they pressed to
        copy or to leave, so that branch goes straight to the confirmation.
        """
        if len(self.screen_stack) == 1 and self.viewing_primary:
            textarea = self._multiline_textarea
            if textarea is not None:
                if textarea.text != "":
                    textarea.text = ""
                    textarea.focus()
                    return
            else:
                # Whitespace is user-entered content: clear it rather than
                # treating it as an empty composer that should offer to quit.
                chat_input = self.query_one("#input", _ChatInput)
                if chat_input.value != "":
                    chat_input.value = ""
                    # Input.Changed -> _update_suggestions() hides the menu.
                    chat_input.focus()
                    return
        self.push_screen(QuitScreen(), self._complete_quit)

    def _complete_quit(self, confirmed: bool) -> None:
        """Exit only on an explicit confirmation from the quit screen.

        Staying needs no work: dismissing the modal restores the previous screen
        and its focus, which may be settings, profiles, or a command prompt
        rather than the chat input.
        """
        if confirmed:
            self.exit()

    def exit(
        self,
        result: object | None = None,
        return_code: int = 0,
        message: RenderableType | None = None,
    ) -> None:
        """Signal every live runtime, then exit exactly as Textual would.

        The one place the app leaves, so ``Ctrl+Q``, ``/exit``, the confirmed
        quit, and a panic exit all stop the same runtimes and kill the same
        child processes without a separate shutdown command. It does not await,
        poll, or delay the exit: the runners' own ``finally`` blocks finish the
        unwinding when Textual cancels them.
        """
        runtime = self._primary_runtime
        if runtime is not None:
            runtime.request_stop()
        for managed in self._agents.values():
            if managed.runtime is not None:
                managed.runtime.request_stop()
        super().exit(result, return_code, message)


__all__ = [
    "BUSY_GENERATING",
    "BUSY_TOOL_ROUND",
    "CONNECTION_ERROR",
    "NO_PROFILE",
    "PRIMARY_AGENT_ID",
    "RENDER_ERROR",
    "SPINNER_FRAMES",
    "SUBAGENT_CLEAR_BLOCKED",
    "SUBAGENT_READONLY",
    "ChatApp",
    "FetchServerInfo",
    "FetchTokenCount",
    "StreamReply",
]
