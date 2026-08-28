"""Chat application lifecycle, streaming orchestration, and command execution."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Markdown, Static

from jtech_cli import server_info
from jtech_cli.cmd_tools import (
    CmdPolicy,
    ExecResult,
    allow_rule_for,
    decide,
    format_result,
    parse_jtech_reply,
    timeout_partial_output,
    truncate_output,
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
from jtech_cli.llm_client import StreamItem, stream_reply
from jtech_cli.prompts import COMMAND_DECLINED_PROMPT, NUDGE_PROMPT
from jtech_cli.server_info import ServerInfo, fetch_server_info
from jtech_cli.session import Session
from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT, textual_theme_name
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
    AgentWorkspace,
    InputToMultiline,
    MarkdownTail,
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

CONNECTION_ERROR = "Connection failed — check the endpoint in /profiles"
RENDER_ERROR = "Could not render the reply — it was not added to the conversation"
NO_PROFILE = "No API profile is configured — run /profiles to add one."
BUSY_GENERATING = "A reply is streaming — press Esc to stop it before changing profiles."
BUSY_TOOL_ROUND = "A tool round is running — wait for it to finish before changing profiles."
SPINNER_FRAMES = "-\\|/"
PRIMARY_AGENT_ID = "primary"
SUBAGENT_READONLY = "Read only — subagents communicate with their dispatcher."
SUBAGENT_CLEAR_BLOCKED = (
    "Subagent activity is read only; switch to Primary to clear chat."
)

StreamReply = Callable[[ResolvedProfile, float, list[dict]], Iterator[StreamItem]]
FetchServerInfo = Callable[[Profile], ServerInfo]
FetchTokenCount = Callable[[Profile, str], int | None]


@dataclass(frozen=True)
class _StreamEnd:
    """Terminal marker the provider thread enqueues exactly once."""

    error: Exception | None = None


_QueuedStreamItem = StreamItem | _StreamEnd


class _StreamInbox:
    """Ordered handoff from the provider thread to the Textual event loop.

    The provider iterator is synchronous and runs off the event loop, so it may
    not touch widgets or stream state. It only appends here. Every event is
    enqueued once, drained once in source order, and at most one wake-up is
    outstanding for a non-empty queue: the idle-to-pending transition and the
    drain that clears it share one lock, so an item can neither slip into the
    gap between draining and clearing nor schedule a redundant callback.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._items: deque[_QueuedStreamItem] = deque()
        self._lock = threading.Lock()
        self._ready = asyncio.Event()
        self._notified = False

    def put(self, item: _QueuedStreamItem) -> None:
        """Enqueue one event from any thread, waking the consumer if idle."""
        with self._lock:
            self._items.append(item)
            notify = not self._notified
            self._notified = True
        if notify:
            self._loop.call_soon_threadsafe(self._ready.set)

    async def get_batch(self) -> list[_QueuedStreamItem]:
        """Await and return every event queued since the last drain, in order."""
        await self._ready.wait()
        with self._lock:
            self._ready.clear()
            batch = list(self._items)
            self._items.clear()
            self._notified = False
        return batch


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
        self._running_proc: subprocess.Popen | None = None
        self._cmd_interrupted = False
        self._tool_rounds_active = False
        self._multiline_textarea: _MultilineInput | None = None
        self._multiline_future: asyncio.Future[str] | None = None
        self._multiline_terminator = "'''"
        self._generating = False
        self._stop_event = threading.Event()
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
                placeholder="Message… (Shift+Enter for newline · / for commands)",
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

    async def _enter_multiline(self, terminator: str = "'''", prefill: str = "") -> str:
        self._multiline_terminator = terminator
        self._multiline_future = asyncio.get_running_loop().create_future()
        inp = self.query_one("#input", _ChatInput)
        textarea = _MultilineInput(classes="multiline", id="multiline-input")
        inp.display = False
        self.mount(textarea)
        self._multiline_textarea = textarea
        if prefill:
            textarea.text = prefill
        textarea.focus()
        return await self._multiline_future

    def on_multiline_submit(self, _event: MultilineSubmit) -> None:
        textarea = self._multiline_textarea
        if textarea is None:
            return
        lines = textarea.text.splitlines()
        if lines and lines[-1].strip() == self._multiline_terminator:
            self._resolve_multiline("\n".join(lines[:-1]))

    def on_multiline_cancel(self, _event: MultilineCancel) -> None:
        self._resolve_multiline("")

    def _resolve_multiline(self, content: str) -> None:
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
        cancellation, a command round, a nudge, or a drained queue — releases
        exactly one acquired depth.
        """
        content = content.strip()
        if not content:
            return
        if self._generating or self._tool_rounds_active:
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
        continuation, and each nudge — is handed the same ``ResolvedProfile``,
        so an endpoint change can never take effect halfway through a turn.
        """
        try:
            profile = self._resolve_turn_profile()
        except ProfileError as error:
            self.push_message("system", str(error))
            profile = None
        if profile is not None:
            temperature = self.settings.temperature
            reply = await self._stream_once(profile, temperature)
            if reply is not None:
                await self._tool_rounds(reply, profile, temperature)
        while self._queue:
            text = self._queue[0]
            self._remove_queue_entry(0)
            await self._send_message(text)

    async def _stream_once(
        self, profile: ResolvedProfile, temperature: float
    ) -> str | None:
        """Consume one stream and return its answer, or ``None`` on failure."""
        parts: list[str] = []
        timings: dict | None = None
        error: Exception | None = None
        started = time.monotonic()
        mode = self.settings.reasoning
        content_chars = 0
        reasoning_chars = 0
        got_item = False
        got_content = False
        tick_count = 0
        reason_visible = False
        reason_dropped = False
        reason_tail = ""
        reason_text = Text()
        self._stop_event.clear()
        self._generating = True

        chat = self.query_one("#chat", Transcript)
        reason_entry: PlainTail | None = None
        if mode != "hide":
            reason_entry = chat.begin_plain("reasoning", hidden=True)
        ai_entry: MarkdownTail = chat.begin_markdown("ai")
        label, ai_md = ai_entry.label, ai_entry.body
        ai_stream = Markdown.get_stream(ai_md)
        # Anchoring lets Textual keep the newest output in view by itself and
        # release that hold the moment the user scrolls, so the stream never
        # has to issue a scroll request of its own per chunk.
        chat.anchor()

        def drop_reason_bubble() -> None:
            """Withdraw the reasoning entry; the transcript owns its widgets."""
            if reason_entry is not None:
                chat.remove(reason_entry)

        def close_reason_bubble() -> None:
            """End the reasoning entry the way its mode asks for.

            ``transient`` never keeps anything, and neither retaining mode has
            anything worth keeping if no reasoning ever became visible; both
            withdraw the entry instead of compacting an empty one.
            """
            if reason_entry is None:
                return
            if mode == "transient" or not reason_visible:
                chat.remove(reason_entry)
                return
            displayed = "…" + reason_tail if mode == "tail" else reason_text.plain
            chat.finalize(
                reason_entry,
                TranscriptRecord("reasoning", displayed, format="plain"),
            )

        def render_reasoning() -> None:
            """Push the accumulated reasoning body into its bubble.

            ``tail`` renders a bounded string; the retaining modes hand over one
            mutable ``Text`` that grows by append, so no mode re-joins every
            fragment received so far.
            """
            assert reason_entry is not None
            if mode == "tail":
                reason_entry.body.update("…" + reason_tail)
            else:
                reason_entry.body.update(reason_text)

        def apply_reasoning(delta: str) -> None:
            """Fold one batch of reasoning deltas into the reasoning bubble."""
            nonlocal reason_visible, reason_dropped, reason_tail
            if reason_entry is None:
                return
            if delta:
                if mode == "tail":
                    reason_tail = (reason_tail + delta)[-500:]
                else:
                    reason_text.append(delta)
            if reasoning_chars and not reason_visible and not reason_dropped:
                reason_entry.label.display = True
                reason_entry.body.display = True
                reason_visible = True
            if not reason_visible:
                return
            if got_content and mode == "transient":
                drop_reason_bubble()
                reason_visible = False
                reason_dropped = True
            elif delta:
                render_reasoning()

        def update_label() -> None:
            """Repaint the AI status line from the running counters only."""
            if not label.is_mounted:
                return
            elapsed = int(time.monotonic() - started)
            frame = SPINNER_FRAMES[tick_count % len(SPINNER_FRAMES)]
            if not got_item:
                label.update(f"AI  ·  waiting {elapsed}s")
            elif not got_content:
                label.update(f"AI  ·  {frame}  thinking… {reasoning_chars}")
            else:
                label.update(f"AI  ·  {frame}  {content_chars}")

        def tick() -> None:
            nonlocal tick_count
            tick_count += 1
            update_label()

        timer = self.set_interval(1.0, tick)
        inbox = _StreamInbox(asyncio.get_running_loop())
        # This stream's own stop signal, separate from the app-wide Escape
        # flag: it says "nobody is reading the inbox any more", which is true
        # on every exit, not only the ones the user asked for.
        abandoned = threading.Event()

        def consume() -> None:
            """Provider thread: enqueue every event, then one end marker."""
            producer_error: Exception | None = None
            try:
                messages = self.session.messages_with_system(
                    self.settings.effective_system_prompt()
                )
                for item in self._stream_reply_fn(profile, temperature, messages):
                    inbox.put(item)
                    if abandoned.is_set() or self._stop_event.is_set():
                        break
            except Exception as exc:  # noqa: BLE001 - report connection failures cleanly
                producer_error = exc
            finally:
                inbox.put(_StreamEnd(producer_error))

        producer = threading.Thread(target=consume, daemon=True)
        producer.start()

        render_error: Exception | None = None
        try:
            try:
                ended = False
                while not ended:
                    batch = await inbox.get_batch()
                    content_deltas: list[str] = []
                    reason_deltas: list[str] = []
                    for item in batch:
                        if isinstance(item, _StreamEnd):
                            error = item.error
                            ended = True
                            break
                        got_item = True
                        if isinstance(item, tuple):
                            kind, payload = item
                            if kind == "reasoning":
                                reasoning_chars += len(payload)
                                reason_deltas.append(payload)
                            elif kind == "usage":
                                self._prompt_tokens = payload["prompt_tokens"]
                            elif kind == "timings":
                                timings = payload
                                prompt_n = payload.get("prompt_n")
                                if prompt_n:
                                    self._prompt_tokens = int(prompt_n)
                        else:
                            parts.append(item)
                            content_chars += len(item)
                            if item:
                                got_content = True
                                content_deltas.append(item)
                    # /clear can unmount the live bubbles mid-stream; the answer
                    # keeps accumulating for the model, but there is nothing
                    # left to paint.
                    if label.is_mounted:
                        apply_reasoning("".join(reason_deltas))
                        combined_delta = "".join(content_deltas)
                        if combined_delta:
                            # Awaited, not detached: the next batch cannot be
                            # drained — and the stream cannot be finalized —
                            # until this lands.
                            await ai_stream.write(combined_delta)
                    update_label()
            except Exception as exc:  # noqa: BLE001 - a broken renderer ends the turn
                # Rendering happens on the event loop now, so a failure here has
                # no provider thread to surface it. Leaving it to propagate
                # would latch _generating and queue every later message behind
                # a turn that can never finish, so it is reported like any
                # other stream failure and the turn ends.
                render_error = exc
            finally:
                timer.stop()

            await ai_stream.stop()
            async with ai_md.lock:
                pass
            if self._stop_event.is_set():
                drop_reason_bubble()
                chat.remove(ai_entry)
                self.push_message("system", "Generation stopped.")
                return None
            failure = error if render_error is None else render_error
            if failure is not None:
                headline = CONNECTION_ERROR if render_error is None else RENDER_ERROR
                error_text = f"{headline}\n\n{failure}"
                # A /clear during the turn already closed these handles; the
                # failure still ends the turn, it just has nothing to paint.
                if ai_entry.state == "live":
                    label.update("AI")
                    ai_md.add_class("error")
                    await ai_md.update(error_text)
                close_reason_bubble()
                chat.finalize(
                    ai_entry,
                    TranscriptRecord("ai", error_text, label="AI", error=True),
                )
                chat.scroll_end(animate=False)
                return None
            reply = "".join(parts)
            if reply.strip():
                self._record_message("assistant", reply)
            done_label = self._done_label(timings)
            if ai_entry.state == "live":
                # A queue notice can hold this bubble on screen past the turn,
                # so the widget gets the done label too, not just the record.
                label.update(done_label)
            close_reason_bubble()
            chat.finalize(ai_entry, TranscriptRecord("ai", reply, label=done_label))
            self.ctx.last_reply = reply
            return reply
        finally:
            abandoned.set()
            try:
                # Every path but a render failure has already seen _StreamEnd,
                # so the producer is done. That one has not: releasing the turn
                # with it still running would let the next request overlap it
                # and grow an inbox nobody drains. Join off the event loop —
                # Escape already makes a turn wait for its provider this way.
                if producer.is_alive():
                    await asyncio.to_thread(producer.join)
            finally:
                self._generating = False
                self._render_status()

    async def _tool_rounds(
        self, reply: str, profile: ResolvedProfile, temperature: float
    ) -> None:
        """Run model rounds until one reply carries prose and no command calls.

        Command output must reach the model before the next completion decision,
        so this loop owns the only normal exit: a reply with non-whitespace text
        and no parsed commands. Any command call — however often it repeats, and
        whatever result it produced — costs another model round, and an empty
        reply is nudged rather than treated as an answer. There is no command,
        round, repetition, or retry budget. ``None`` is a cancellation or a
        provider failure that ``_stream_once`` has already reported.

        ``profile`` and ``temperature`` are the turn's captured values: no round
        here re-reads the live active profile.
        """
        self._tool_rounds_active = True
        try:
            while reply is not None:
                commands = parse_jtech_reply(reply).commands
                if commands:
                    await self._process_commands(commands)
                    reply = await self._stream_once(profile, temperature)
                    continue
                if reply.strip():
                    return
                reply = await self._nudge(profile, temperature)
        finally:
            self._tool_rounds_active = False

    async def _nudge(
        self, profile: ResolvedProfile, temperature: float
    ) -> str | None:
        """Request a continuation after the model returned an empty reply."""
        if self.settings.debug_level == "system":
            # Keep an auditable event without feeding it into future prompts.
            self._record_message(
                "system",
                NUDGE_PROMPT,
                include_in_context=False,
                debug_only=True,
            )
            self.push_message("system", NUDGE_PROMPT)
        with self.session.ephemeral("system", NUDGE_PROMPT):
            return await self._stream_once(profile, temperature)

    async def _process_commands(self, commands: list[str]) -> None:
        """Run every parsed command in source order, feeding each result back."""
        for command in commands:
            if not command.strip():
                self._note_command(command, "empty command — not run")
                continue
            decision = decide(command, self.cmd, self._project_root)
            if decision.action == "blocked":
                self._note_command(command, f"blocked — {decision.reason}")
                continue
            if decision.action == "ask":
                choice = await self._prompt_for_command(command, decision.reason)
                if choice is CmdChoice.DECLINE:
                    self._note_command(command, "declined by the user")
                    self._add_system(COMMAND_DECLINED_PROMPT)
                    continue
                if choice is CmdChoice.ALWAYS:
                    self._add_allow_rule(allow_rule_for(command, self.cmd.allow))
            result = await self._exec_command(command)
            self.push_message("system", self._cmd_bubble(command, result))
            self._add_system(format_result(command, result=result))

    def _note_command(self, command: str, note: str) -> None:
        self.push_message("system", f"$ {command}\n→ {note}")
        self._add_system(format_result(command, note=note))

    async def _prompt_for_command(self, command: str, reason: str) -> CmdChoice:
        """Ask the user about one command, reporting Primary as waiting.

        The surrounding accepted turn still owns the final transition to
        ``idle``; this only marks the stretch where Primary cannot progress
        without the user.
        """
        self._set_primary_agent_status("waiting")
        try:
            return await self.push_screen_wait(CommandPrompt(command, reason))
        finally:
            self._set_primary_agent_status("running")

    def _add_allow_rule(self, rule: str | None) -> None:
        if not rule or rule in self.cmd.allow:
            return
        self.cmd.allow.append(rule)
        try:
            self._save(self.settings)
            self.push_message("system", f"Always-allow saved: {rule}")
        except OSError as error:
            self.push_message("system", f"Could not save always-allow rule: {error}")

    async def _exec_command(self, command: str) -> ExecResult:
        """Run one command in a worker; Esc and timeout preserve partial output."""
        self._render_status(running="running command…")
        self._cmd_interrupted = False
        try:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                ["bash", "-c", command],
                cwd=self._project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as error:
            self._render_status()
            return ExecResult(127, str(error))
        self._running_proc = proc
        try:
            try:
                out, _ = await asyncio.to_thread(
                    proc.communicate, timeout=self.cmd.timeout
                )
            except subprocess.TimeoutExpired as error:
                proc.kill()
                await asyncio.to_thread(proc.wait)
                partial, truncated = truncate_output(
                    timeout_partial_output(error), self.cmd.max_output
                )
                return ExecResult(124, partial, timed_out=True, truncated=truncated)
            out = out or ""
            if self._cmd_interrupted:
                self._cmd_interrupted = False
                text, truncated = truncate_output(out, self.cmd.max_output)
                return ExecResult(130, text, interrupted=True, truncated=truncated)
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
        if not timings:
            return "AI"
        prompt_n = int(timings.get("prompt_n", 0))
        prompt_s = float(timings.get("prompt_ms", 0)) / 1000
        prompt_tps = float(timings.get("prompt_per_second", 0))
        return f"AI  ·  prompt {prompt_n:,} tok · {prompt_s:.1f}s · {prompt_tps:.0f} t/s"

    def action_stop_stream(self) -> None:
        # A subagent view has no writable target and no cancellation of its own:
        # stopping the hidden Primary turn from here would be invisible work.
        # The visible read-only notice is the documented explanation.
        if not self.viewing_primary:
            return
        if self._generating:
            self._stop_event.set()
        elif self._running_proc is not None:
            self._cmd_interrupted = True
            self._running_proc.kill()
        else:
            self._hide_suggestions()

    def _busy(self) -> str | None:
        """Why a profile change is refused right now, or ``None`` when idle.

        The current turn must keep one endpoint/model/credential identity, so a
        switch that is visible in the footer but does not affect the running
        turn is refused outright rather than half-applied.
        """
        if self._generating:
            return BUSY_GENERATING
        if self._tool_rounds_active:
            return BUSY_TOOL_ROUND
        return None

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
