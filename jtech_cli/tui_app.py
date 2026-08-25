"""Chat application lifecycle, streaming orchestration, and command execution."""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
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
from jtech_cli.config import CONFIG_PATH, Settings, save_settings
from jtech_cli.llm_client import StreamItem, stream_reply
from jtech_cli.prompts import COMMAND_DECLINED_PROMPT, NUDGE_PROMPT
from jtech_cli.server_info import ServerInfo, fetch_server_info
from jtech_cli.session import Session
from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT, textual_theme_name
from jtech_cli.tui_screens import CmdChoice, CommandPrompt, SettingsScreen
from jtech_cli.tui_widgets import (
    InputToMultiline,
    MultilineCancel,
    MultilineSubmit,
    OutputSink,
    _ChatInput,
    _MultilineInput,
    render_menu_rows,
)

CONNECTION_ERROR = "Connection failed — check base_url in /settings"
RENDER_ERROR = "Could not render the reply — it was not added to the conversation"
SPINNER_FRAMES = "-\\|/"

StreamReply = Callable[[Settings, list[dict]], Iterator[StreamItem]]
FetchServerInfo = Callable[[Settings], ServerInfo]
FetchTokenCount = Callable[[Settings, str], int | None]


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


def _message_widgets(role: str, text: str) -> tuple[Static, Markdown]:
    """Build the label/body pair for one transcript message.

    Construction only — mounting, scrolling, and persistence stay with the
    caller, so replayed history can be mounted in one batch while a live
    message still follows the transcript on its own.
    """
    label = Static(role.upper(), classes=f"bubble-label {role}")
    markdown = Markdown(text or "", classes=f"bubble {role}")
    return label, markdown


class ChatApp(App):
    """Full-screen chat app with injected network boundaries."""

    CSS_PATH = Path(__file__).parent / "resources" / "styles" / "tui.css"

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
        # Replayed history is mounted as one batch and scrolled once: going
        # through _append() would mount and scroll per stored message.
        history: list[Static | Markdown] = []
        for msg in self.session.messages:
            if msg.get("_debug_only") and self.settings.debug_level != "system":
                continue
            history.extend(_message_widgets(msg["role"], msg["content"]))
        if history:
            chat = self.query_one("#chat", VerticalScroll)
            await chat.mount(*history)
            chat.scroll_end(animate=False)
        if not self.settings.base_url:
            self.push_message(
                "system", "No server configured — run /settings to set base_url and model."
            )
        if self.settings.prompt_notice:
            self.push_message("system", self.settings.prompt_notice)
        self.query_one("#suggestions", Static).display = False
        self._render_status()
        self._focus_input()
        if self.session.messages and self.server.context_length:
            await self._init_token_count()
        if not self._no_discover and self.settings.base_url:
            self.call_later(self._discover_server)

    async def _discover_server(self) -> None:
        """Refresh server metadata in place without downgrading known values."""
        info = await asyncio.to_thread(self._fetch_server_info_fn, self.settings)
        if not info.known:
            self.push_message(
                "system",
                f"Could not reach {self.settings.base_url} — model and context "
                "info unavailable. Check base_url in /settings.",
            )
            return
        self.server.models = info.models
        self.server.context_length = info.context_length
        if self.server.model and not self.settings.model:
            self.settings.model = self.server.model
        self._render_status()
        if self.session.messages and self.server.context_length:
            await self._init_token_count()

    def push_message(self, role: str, text: str) -> tuple[Static, Markdown]:
        return self._append(role, text)

    def _append(self, role: str, text: str) -> tuple[Static, Markdown]:
        chat = self.query_one("#chat", VerticalScroll)
        label, markdown = _message_widgets(role, text)
        chat.mount(label, markdown)
        chat.scroll_end(animate=False)
        return label, markdown

    def _append_plain(
        self, role: str, text: str, *, hidden: bool = False
    ) -> tuple[Static, Static]:
        """Mount a cheap-to-update plain-text bubble, optionally hidden."""
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
        inputs = self.query(_ChatInput)
        if inputs:
            inputs[0].focus()

    async def _init_token_count(self) -> None:
        """Count session tokens on startup so the footer is accurate."""
        history = self.session.messages_with_system("")
        text = " ".join(message["content"] for message in history)
        if not text:
            return
        count = await asyncio.to_thread(self._fetch_token_count_fn, self.settings, text)
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
        if self.settings.base_url:
            parts.append(self.settings.base_url)
        if self.settings.model:
            parts.append(f"model: {self.settings.model}")
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
        """Queue a message while a reply or tool round is in flight."""
        self._queue.append(text)
        self._queue_lines.append(self.push_message("system", f"Queued: {text}"))
        self._render_status()

    def _remove_queue_entry(self, index: int) -> None:
        """Drop a queued message and its transient chat line."""
        self._queue.pop(index)
        for widget in self._queue_lines.pop(index):
            if widget.is_mounted:
                widget.remove()
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
        content = content.strip()
        if not content:
            return
        if self._generating or self._tool_rounds_active:
            self._enqueue(content)
            return
        self._record_message("user", content)
        self._append("user", content)
        await self._stream_reply()

    async def _stream_reply(self) -> None:
        reply = await self._stream_once()
        if reply is not None:
            await self._tool_rounds(reply)
        while self._queue:
            text = self._queue[0]
            self._remove_queue_entry(0)
            await self._send_message(text)

    async def _stream_once(self) -> str | None:
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

        chat = self.query_one("#chat", VerticalScroll)
        reason_pair: tuple[Static, Static] | None = None
        if mode != "hide":
            reason_pair = self._append_plain("reasoning", "", hidden=True)
        label, ai_md = self._append("ai", "")
        ai_stream = Markdown.get_stream(ai_md)
        # Anchoring lets Textual keep the newest output in view by itself and
        # release that hold the moment the user scrolls, so the stream never
        # has to issue a scroll request of its own per chunk.
        chat.anchor()

        def drop_reason_bubble() -> None:
            if reason_pair is None:
                return
            for widget in reason_pair:
                if widget.is_mounted:
                    widget.remove()

        def render_reasoning() -> None:
            """Push the accumulated reasoning body into its bubble.

            ``tail`` renders a bounded string; the retaining modes hand over one
            mutable ``Text`` that grows by append, so no mode re-joins every
            fragment received so far.
            """
            assert reason_pair is not None
            if mode == "tail":
                reason_pair[1].update("…" + reason_tail)
            else:
                reason_pair[1].update(reason_text)

        def apply_reasoning(delta: str) -> None:
            """Fold one batch of reasoning deltas into the reasoning bubble."""
            nonlocal reason_visible, reason_dropped, reason_tail
            if reason_pair is None:
                return
            if delta:
                if mode == "tail":
                    reason_tail = (reason_tail + delta)[-500:]
                else:
                    reason_text.append(delta)
            if reasoning_chars and not reason_visible and not reason_dropped:
                reason_pair[0].display = True
                reason_pair[1].display = True
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
                for item in self._stream_reply_fn(self.settings, messages):
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
                label.remove()
                ai_md.remove()
                self.push_message("system", "Generation stopped.")
                return None
            failure = error if render_error is None else render_error
            if failure is not None:
                if mode == "transient":
                    drop_reason_bubble()
                label.update("AI")
                ai_md.add_class("error")
                headline = CONNECTION_ERROR if render_error is None else RENDER_ERROR
                await ai_md.update(f"{headline}\n\n{failure}")
                chat.scroll_end(animate=False)
                return None
            if mode == "transient":
                drop_reason_bubble()
            label.update(self._done_label(timings))
            reply = "".join(parts)
            if reply.strip():
                self._record_message("assistant", reply)
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

    async def _tool_rounds(self, reply: str) -> None:
        """Run model rounds until one reply carries prose and no command calls.

        Command output must reach the model before the next completion decision,
        so this loop owns the only normal exit: a reply with non-whitespace text
        and no parsed commands. Any command call — however often it repeats, and
        whatever result it produced — costs another model round, and an empty
        reply is nudged rather than treated as an answer. There is no command,
        round, repetition, or retry budget. ``None`` is a cancellation or a
        provider failure that ``_stream_once`` has already reported.
        """
        self._tool_rounds_active = True
        try:
            while reply is not None:
                commands = parse_jtech_reply(reply).commands
                if commands:
                    await self._process_commands(commands)
                    reply = await self._stream_once()
                    continue
                if reply.strip():
                    return
                reply = await self._nudge()
        finally:
            self._tool_rounds_active = False

    async def _nudge(self) -> str | None:
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
            return await self._stream_once()

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
        return await self.push_screen_wait(CommandPrompt(command, reason))

    def _add_allow_rule(self, rule: str | None) -> None:
        if not rule or rule in self.cmd.allow:
            return
        self.cmd.allow.append(rule)
        try:
            self.cmd.mode = self.settings.cmd_mode
            save_settings(self.settings, self.config_path, cmd=self.cmd)
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
        if self._generating:
            self._stop_event.set()
        elif self._running_proc is not None:
            self._cmd_interrupted = True
            self._running_proc.kill()
        else:
            self._hide_suggestions()

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
        self.query_one("#chat", VerticalScroll).remove_children()
        self._prompt_tokens = 0
        self._render_status()

    def action_clear_chat(self) -> None:
        self.commands.handle("/clear")

    def action_settings(self) -> None:
        self._open_settings()
