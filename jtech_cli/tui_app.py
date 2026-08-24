"""Chat application lifecycle, streaming orchestration, and command execution."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

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
from jtech_cli.prompts import (
    BLOCKS_DROPPED_PROMPT,
    COMMAND_DECLINED_PROMPT,
    NUDGE_PROMPT,
    REPEATED_COMMAND_PROMPT,
)
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
SPINNER_FRAMES = "-\\|/"
MAX_BLOCKS_PER_REPLY = 5

StreamReply = Callable[[Settings, list[dict]], Iterator[StreamItem]]
FetchServerInfo = Callable[[Settings], ServerInfo]
FetchTokenCount = Callable[[Settings, str], int | None]


@dataclass(frozen=True)
class CommandBatch:
    """Execution count and outcome signature for one model command response."""

    executed: int
    signature: tuple[tuple[str, str], ...]


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
        for msg in self.session.messages:
            if msg.get("_debug_only") and self.settings.debug_level != "system":
                continue
            self._append(msg["role"], msg["content"])
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
        label = Static(role.upper(), classes=f"bubble-label {role}")
        chat.mount(label)
        markdown = Markdown(text or "", classes=f"bubble {role}")
        chat.mount(markdown)
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
        self.session.add("user", content)
        self._save()
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
            return ("…" + reason[-500:]) if mode == "tail" else reason

        reason_pair: tuple[Static, Static] | None = None
        if mode != "hide":
            reason_pair = self._append_plain("reasoning", "", hidden=True)
        label, ai_md = self._append("ai", "")
        ai_stream = Markdown.get_stream(ai_md)
        stream_closed = False

        def drop_reason_bubble() -> None:
            if reason_pair is None:
                return
            for widget in reason_pair:
                if widget.is_mounted:
                    widget.remove()

        def paint() -> None:
            nonlocal reason_visible, last_content, last_reason, stream_closed
            if not label.is_mounted:
                return
            reason = "".join(reason_parts)
            content = "".join(parts)
            moved = False
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
                        drop_reason_bubble()
                        reason_visible = False
                        moved = True
                    else:
                        shown = reason_display(reason)
                        if shown != last_reason:
                            reason_pair[1].update(shown)
                            last_reason = shown
                            moved = True
            if not stream_closed and content and content != last_content:
                delta = content[len(last_content) :]
                last_content = content
                asyncio.ensure_future(ai_stream.write(delta))
                moved = True
            elapsed = int(time.monotonic() - started)
            frame = SPINNER_FRAMES[tick_count % len(SPINNER_FRAMES)]
            if not got_token:
                label.update(f"AI  ·  waiting {elapsed}s")
            elif not content:
                label.update(f"AI  ·  {frame}  thinking… {len(reason)}")
            else:
                label.update(f"AI  ·  {frame}  {len(content)}")
            if moved:
                self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

        def tick() -> None:
            nonlocal tick_count
            tick_count += 1
            paint()

        timer = self.set_interval(1.0, tick)

        def consume() -> None:
            nonlocal error, got_token, timings
            try:
                messages = self.session.messages_with_system(
                    self.settings.effective_system_prompt()
                )
                for item in self._stream_reply_fn(self.settings, messages):
                    if isinstance(item, tuple):
                        kind, payload = item
                        if kind == "reasoning":
                            reason_parts.append(payload)
                        elif kind == "usage":
                            self._prompt_tokens = payload["prompt_tokens"]
                        elif kind == "timings":
                            timings = payload
                            prompt_n = payload.get("prompt_n")
                            if prompt_n:
                                self._prompt_tokens = int(prompt_n)
                    else:
                        parts.append(item)
                    got_token = True
                    self.call_from_thread(paint)
                    if self._stop_event.is_set():
                        break
            except Exception as exc:  # noqa: BLE001 - report connection failures cleanly
                error = exc
            finally:
                self.call_from_thread(done.set)

        threading.Thread(target=consume, daemon=True).start()
        await done.wait()

        try:
            timer.stop()
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
            if error is not None:
                if mode == "transient":
                    drop_reason_bubble()
                label.update("AI")
                ai_md.add_class("error")
                await ai_md.update(f"{CONNECTION_ERROR}\n\n{error}")
                self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
                return None
            if mode == "transient":
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
        """Process standalone command calls until the model stops or repeats a result.

        The model may add commentary after a command prefix. A prose-only reply
        is terminal. An empty reply after a command is the one explicit recovery
        case: nudge once, then require another command call or finish the turn.
        Repeating a command batch with identical outcomes is a no-progress loop.
        """
        self._tool_rounds_active = True
        try:
            seen_batches: set[tuple[tuple[str, str], ...]] = set()
            while True:
                commands = parse_jtech_reply(reply).commands
                if not commands:
                    return
                if len(commands) > MAX_BLOCKS_PER_REPLY:
                    self._report_blocks_dropped(len(commands))
                batch = await self._process_commands(
                    commands[:MAX_BLOCKS_PER_REPLY]
                )
                if batch.signature in seen_batches:
                    self._report_no_progress()
                    return
                seen_batches.add(batch.signature)
                reply = await self._stream_once()
                if reply is None:
                    return
                if batch.executed and not reply.strip():
                    reply = await self._nudge()
                    if reply is None:
                        return
        finally:
            self._tool_rounds_active = False

    def _report_blocks_dropped(self, total: int) -> None:
        dropped = total - MAX_BLOCKS_PER_REPLY
        self.push_message(
            "system",
            f"{dropped} extra command call(s) ignored "
            f"(max {MAX_BLOCKS_PER_REPLY} per reply).",
        )
        self._add_system(
            BLOCKS_DROPPED_PROMPT.format(kept=MAX_BLOCKS_PER_REPLY, dropped=dropped)
        )

    def _report_no_progress(self) -> None:
        self.push_message(
            "system",
            "Repeated command round detected with identical tool results; "
            "stopping to prevent a no-progress loop.",
        )
        self._add_system(REPEATED_COMMAND_PROMPT)

    async def _nudge(self) -> str | None:
        """Request one continuation after a command round ended empty."""
        if self.settings.debug_level == "system":
            # Keep an auditable event without feeding it into future prompts.
            self.session.add(
                "system",
                NUDGE_PROMPT,
                include_in_context=False,
                debug_only=True,
            )
            self._save()
            self.push_message("system", NUDGE_PROMPT)
        try:
            with self.session.ephemeral("system", NUDGE_PROMPT):
                return await self._stream_once()
        finally:
            self._save()

    async def _process_commands(self, commands: list[str]) -> CommandBatch:
        executed = 0
        outcomes: list[tuple[str, str]] = []
        for command in commands:
            if not command.strip():
                note = "empty command — not run"
                self._note_command(command, note)
                outcomes.append((command, f"note:{note}"))
                continue
            decision = decide(command, self.cmd, self._project_root)
            if decision.action == "blocked":
                note = f"blocked — {decision.reason}"
                self._note_command(command, note)
                outcomes.append((command, f"note:{note}"))
                continue
            if decision.action == "ask":
                choice = await self._prompt_for_command(command, decision.reason)
                if choice is CmdChoice.DECLINE:
                    note = "declined by the user"
                    self._note_command(command, note)
                    self._add_system(COMMAND_DECLINED_PROMPT)
                    outcomes.append((command, f"note:{note}"))
                    continue
                if choice is CmdChoice.ALWAYS:
                    self._add_allow_rule(allow_rule_for(command, self.cmd.allow))
            result = await self._exec_command(command)
            executed += 1
            self.push_message("system", self._cmd_bubble(command, result))
            self._add_system(format_result(command, result=result))
            outcomes.append((command, self._result_signature(result)))
        return CommandBatch(executed, tuple(outcomes))

    @staticmethod
    def _result_signature(result: ExecResult) -> str:
        """Fingerprint a tool result without retaining its potentially large output."""
        output_hash = hashlib.sha256(result.output.encode("utf-8")).hexdigest()
        return (
            f"exit={result.exit_code};timed_out={result.timed_out};"
            f"interrupted={result.interrupted};truncated={result.truncated};"
            f"output_sha256={output_hash}"
        )

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
        self.session.add(
            "system",
            content,
            model_role="user",
            model_content=f"[JTECH runtime event]\n{content}",
        )
        self._save()
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
