"""The chat application: its lifecycle, its Primary turn, and its wiring.

What is left here is what only an ``App`` can do — composing the widget tree,
handling Textual messages and key actions, driving one accepted Primary turn,
and exiting — plus the small surfaces other modules and the widgets name:
``RuntimeHost`` for every runtime, ``SuggestionHost`` and ``MessagePusher``
for the composer input and command output.

Everything else it used to carry now belongs to a collaborator it constructs
and hands itself to, each declaring the narrow port it needs:
:class:`~jtech_cli.tui_agents.AgentCoordinator` for dispatched agents,
:class:`~jtech_cli.tui_composer.Composer` for unsent input,
:class:`~jtech_cli.tui_profiles.ProfileManager` for the active endpoint,
:class:`~jtech_cli.tui_status.StatusView` for the transcript and footer, and
:class:`~jtech_cli.tui_approvals.CommandApprovals` for command policy. One
conversation's own model/command loop lives in
:class:`~jtech_cli.tui_runtime.AutonomousRuntime`, which Primary and every
subagent instantiate identically.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from rich.console import RenderableType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Static

from jtech_cli import server_info
from jtech_cli.cmd_tools import AgentDispatch, CmdPolicy
from jtech_cli.commands import CommandContext, build_registry
from jtech_cli.config import (
    CONFIG_PATH,
    ProfileError,
    ResolvedProfile,
    Settings,
)
from jtech_cli.llm_client import stream_reply
from jtech_cli.prompts import compose_coordinator_prompt
from jtech_cli.server_info import (
    FetchServerInfo,
    FetchTokenCount,
    ServerInfo,
    fetch_server_info,
)
from jtech_cli.session import Session
from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT, textual_theme_name
from jtech_cli.tui_agents import AgentCoordinator
from jtech_cli.tui_approvals import CommandApprovals
from jtech_cli.tui_composer import Composer
from jtech_cli.tui_profiles import (
    BUSY_GENERATING,
    BUSY_TOOL_ROUND,
    NO_PROFILE,
    ProfileManager,
)
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
from jtech_cli.tui_screens import QuitScreen
from jtech_cli.tui_status import StatusView
from jtech_cli.tui_widgets import (
    AgentStatus,
    AgentSummary,
    AgentWorkspace,
    InputToMultiline,
    MultilineCancel,
    MultilineSubmit,
    OutputSink,
    Transcript,
    TranscriptRecord,
    _ChatInput,
)

PRIMARY_AGENT_ID = "primary"
SUBAGENT_READONLY = "Read only — subagents communicate with their dispatcher."
SUBAGENT_CLEAR_BLOCKED = (
    "Subagent activity is read only; switch to Primary to clear chat."
)


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
        self.no_discover = no_discover
        self.stream_reply_fn = stream_reply if stream_reply_fn is None else stream_reply_fn
        self.fetch_server_info_fn = (
            fetch_server_info
            if fetch_server_info_fn is None
            else fetch_server_info_fn
        )
        self.fetch_token_count_fn = (
            server_info.fetch_token_count
            if fetch_token_count_fn is None
            else fetch_token_count_fn
        )
        self.cmd = cmd if cmd is not None else CmdPolicy()
        self.settings.cmd_mode = self.cmd.mode
        self.project_root = Path.cwd().resolve()
        self.primary_runtime: AutonomousRuntime | None = None
        self.agents = AgentCoordinator(self)
        self.composer = Composer(self)
        self.status = StatusView(self)
        self.profile_manager = ProfileManager(self)
        self.approvals = CommandApprovals(self)
        self.ctx = CommandContext(
            session=session,
            settings=settings,
            console=OutputSink(self),
            server=server,
            cmd=self.cmd,
            config_path=config_path,
            enter_multiline=self.composer.enter_multiline,
            refresh_footer=self.status.render,
            open_settings=self.profile_manager.open_settings,
            clear_chat=self.status.clear,
            switch_theme=self.profile_manager.switch_theme,
            open_profiles=self.profile_manager.open_modal,
            switch_profile=self.profile_manager.switch,
            effective_prompt=self._primary_system_prompt,
        )
        self.commands = build_registry(self.ctx)
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
                placeholder="Message… (Enter send · Ctrl+J newline · / commands)",
            )
            readonly = Static(SUBAGENT_READONLY, id="subagent-readonly")
            readonly.display = False
            yield readonly
            yield Static(id="status")

    @property
    def _workspace(self) -> AgentWorkspace:
        """The mounted activity/sidebar split."""
        return self.query_one("#agent-workspace", AgentWorkspace)

    # ------------------------------------------------- widgets collaborators use

    def primary_transcript(self) -> Transcript:
        """Primary's own activity stream."""
        return self.query_one("#chat", Transcript)

    def status_bar(self) -> Static:
        """The footer widget."""
        return self.query_one("#status", Static)

    def chat_input(self) -> _ChatInput:
        """The single-line composer."""
        return self.query_one("#input", _ChatInput)

    def suggestion_box(self) -> Static:
        """The completion menu."""
        return self.query_one("#suggestions", Static)

    def refresh_theme(self) -> None:
        """Rebuild rendered history in every registered activity stream."""
        self._workspace.refresh_theme()

    def readonly_notice(self) -> Static:
        """The notice shown in place of the composer for a subagent view."""
        return self.query_one("#subagent-readonly", Static)

    def queue_depth(self) -> int:
        """How many messages are waiting behind the accepted turn."""
        return len(self.composer.queue)

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
        self.composer.show(
            event.agent_id == event.workspace.primary_agent_id
        )

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
                TranscriptRecord.from_message(
                    role=msg["role"],
                    content=msg["content"],
                )
            )
        chat = self.primary_transcript()
        chat.load(history)
        if history:
            chat.scroll_end(animate=False)
        profile = self.settings.active_profile
        if profile is None:
            self.push_message("system", NO_PROFILE)
        if self.settings.prompt_notice:
            self.push_message("system", self.settings.prompt_notice)
        self.suggestion_box().display = False
        self.status.render()
        self._focus_input()
        if profile is not None and self.session.messages and self.server.context_length:
            await self.profile_manager.count_tokens(profile)
        if not self.no_discover and profile is not None:
            self.call_later(self.profile_manager.discover, profile)

    def push_message(self, role: str, text: str) -> None:
        """Add one already-complete message to the visible transcript.

        Kept on the app because ``tui_widgets.MessagePusher`` names it: an
        ``OutputSink`` handed to a command handler pushes through this.
        """
        self.status.push(role, text)

    def _focus_input(self) -> None:
        inputs = self.query(_ChatInput)
        if inputs:
            inputs[0].focus()

    def on_input_changed(self, _event: Input.Changed) -> None:
        self.composer.update_suggestions()

    def action_suggestion_up(self) -> None:
        self.composer.navigate_up()

    def action_suggestion_down(self) -> None:
        self.composer.cycle(1)

    # ``suggestion_navigated``, ``accepted_suggestion``, and
    # ``apply_completion`` are named by ``tui_widgets.SuggestionHost``: the
    # composer input calls them on the app, so they stay here and forward.

    @property
    def suggestion_navigated(self) -> bool:
        return self.composer.suggestion_navigated

    def accepted_suggestion(self) -> str | None:
        return self.composer.accepted_suggestion()

    def apply_completion(self, completed: str) -> None:
        self.composer.apply_completion(completed)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.chat_input().value = ""
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
        content = await self.composer.enter_multiline()
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
        inp = self.chat_input()
        if self.composer.multiline is not None:
            # Reachable when an editor is open but hidden behind a selected
            # subagent. Decline the promotion rather than mount a second
            # editor, but release the flags so the composer still works.
            inp.cancel_promotion()
            return
        draft = inp.take_promotion()
        future = self.composer.open_editor(draft.value, draft.cursor_offset)
        self.run_worker(self._switch_to_multiline(future))
        if draft.submit:
            # Enter arrived while the draft was still in the single-line input.
            # The user asked to send; the racing editor must not change that.
            self.composer.resolve(draft.value)

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

    def on_multiline_submit(self, _event: MultilineSubmit) -> None:
        textarea = self.composer.multiline
        if textarea is None:
            return
        self.composer.resolve(textarea.text)

    def on_multiline_cancel(self, _event: MultilineCancel) -> None:
        self.composer.resolve(None)

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
        if self.primary_runtime is not None:
            self.composer.enqueue(content)
            return
        self._primary_turn_depth += 1
        if self._primary_turn_depth == 1:
            self._set_primary_agent_status("running")
        try:
            self.status.record("user", content)
            self.push_message("user", content)
            await self._stream_reply()
        finally:
            self._primary_turn_depth -= 1
            if self._primary_turn_depth == 0:
                self._set_primary_agent_status("idle")

    async def _stream_reply(self) -> None:
        """Run one user turn against a single immutable profile snapshot.

        Every completion in the turn — the first answer, each command
        continuation, each agent batch, and each nudge — is handed the same
        ``ResolvedProfile``, so an endpoint change can never take effect
        halfway through a turn.
        """
        try:
            profile = self.profile_manager.resolve_turn_profile()
        except ProfileError as error:
            self.push_message("system", str(error))
            profile = None
        if profile is not None:
            runtime = AutonomousRuntime(
                self._primary_run_state(profile),
                host=self,
                stream_reply_fn=self.stream_reply_fn,
                cmd_policy=self.cmd,
                project_root=self.project_root,
            )
            self.primary_runtime = runtime
            try:
                await runtime.run()
            finally:
                self.primary_runtime = None
        while self.composer.queue:
            await self._send_message(self.composer.pop_next())

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
            transcript=self.primary_transcript(),
            profile=profile,
            temperature=self.settings.temperature,
            system_prompt=self._primary_system_prompt,
            reasoning_mode=lambda: self.settings.reasoning,
            debug_level=lambda: self.settings.debug_level,
        )
        state.prompt_tokens = self.status.prompt_tokens
        state.last_reply = self.ctx.last_reply
        return state

    # -------------------------------------------------------------- prompts

    def _primary_system_prompt(self) -> str:
        """The prompt the next Primary completion will actually carry."""
        active = self.settings.active_profile
        return compose_coordinator_prompt(
            self.settings.effective_system_prompt(),
            profile_names=[
                profile.name for profile in self.agents.available_profiles()
            ],
            active_profile_name=active.name if active is not None else None,
        )

    # ------------------------------------------------------ runtime callbacks

    def set_run_phase(self, run: AgentRunState, phase: RunPhase) -> None:
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
            self.status.prompt_tokens = run.prompt_tokens
            self.ctx.last_reply = run.last_reply
            self.status.render(
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
        self.agents.task_changed(run)

    # --------------------------------------------------- RuntimeHost surface

    async def dispatch_agents(
        self, run: AgentRunState, calls: tuple[AgentDispatch, ...]
    ) -> tuple[AgentOutcome, ...]:
        """Run one dispatch batch. Named by ``RuntimeHost``."""
        return await self.agents.dispatch(run, calls)

    async def authorize_command(
        self, run: AgentRunState, command: str
    ) -> CommandAuthorization:
        """Decide one command for ``run``. Named by ``RuntimeHost``."""
        return await self.approvals.authorize(run, command)

    # ------------------------------------------------------------ stop / busy

    def action_stop_stream(self) -> None:
        # A subagent view has no writable target and no cancellation of its own:
        # stopping the hidden Primary turn from here would be invisible work.
        # The visible read-only notice is the documented explanation.
        if not self.viewing_primary:
            return
        runtime = self.primary_runtime
        if runtime is None or not (
            runtime.state.generating or runtime.state.running_proc is not None
        ):
            self.composer.hide_suggestions()
            return
        runtime.request_stop()

    @property
    def generating(self) -> bool:
        """Whether the Primary turn is consuming a provider stream right now.

        Derived from the run that owns the flag rather than mirrored into a
        second boolean, so there is exactly one place it can be true.
        """
        runtime = self.primary_runtime
        return runtime is not None and runtime.state.generating

    @property
    def tool_rounds_active(self) -> bool:
        """Whether the Primary turn is between completions, running its tools."""
        runtime = self.primary_runtime
        return runtime is not None and runtime.state.tool_rounds_active

    def action_clear_chat(self) -> None:
        if not self.viewing_primary:
            self.notify(SUBAGENT_CLEAR_BLOCKED)
            return
        self.commands.handle("/clear")

    def action_settings(self) -> None:
        self.profile_manager.open_settings()

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
            textarea = self.composer.multiline
            if textarea is not None:
                if textarea.text != "":
                    textarea.text = ""
                    textarea.focus()
                    return
            else:
                # Whitespace is user-entered content: clear it rather than
                # treating it as an empty composer that should offer to quit.
                chat_input = self.chat_input()
                if chat_input.value != "":
                    chat_input.value = ""
                    # Input.Changed -> Composer.update_suggestions() hides it.
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
        runtime = self.primary_runtime
        if runtime is not None:
            runtime.request_stop()
        self.agents.request_stop_all()
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
