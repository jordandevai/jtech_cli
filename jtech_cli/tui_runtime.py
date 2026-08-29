"""One autonomous conversation: provider streaming, tool rounds, and shell.

Primary and every subagent run this same loop. The class owns exactly one
conversation's mutable state — its stop signal, live process, generating flag,
transcript target, and continuation state — so concurrent runs cannot overwrite
each other's. Everything app-wide (command policy persistence, approval modals,
profile lookup, the agent catalog, result ordering) stays behind
:class:`RuntimeHost`.

The stop rule is the whole contract: a run continues after every recognized
tool call and after every empty response, and ends normally only when the model
emits a non-whitespace response containing no recognized tool call. There is no
command, round, repetition, retry, concurrency, or elapsed-time budget here.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from rich.text import Text
from textual.widgets import Markdown

from jtech_cli.cmd_tools import (
    AgentDispatch,
    CmdPolicy,
    ExecResult,
    ToolProtocolError,
    duplicate_agent_keys,
    format_result,
    parse_jtech_reply,
    truncate_output,
)
from jtech_cli.config import ResolvedProfile
from jtech_cli.llm_client import StreamItem
from jtech_cli.prompts import (
    COMMAND_DECLINED_PROMPT,
    NUDGE_PROMPT,
    PromptResourceError,
    PromptSourceError,
)
from jtech_cli.session import Session
from jtech_cli.tui_widgets import (
    MarkdownTail,
    PlainTail,
    Transcript,
    TranscriptRecord,
)

CONNECTION_ERROR = "Connection failed — check the endpoint in /profiles"
RENDER_ERROR = "Could not render the reply — it was not added to the conversation"
PROMPT_ERROR = "The system prompt could not be composed — check /system and /prompt"
SPINNER_FRAMES = "-\\|/"

MIXED_TOOLS_ERROR = (
    "Tool protocol error: one response cannot mix jtech_cmd and jtech_agent "
    "calls. No call from the response was executed. Emit one tool kind in the "
    "corrected response."
)
SUBAGENT_DISPATCH_ERROR = (
    "Tool protocol error: a subagent cannot dispatch agents, so no call from "
    "the response was executed. Finish the assignment yourself with shell "
    "commands and report the result."
)

StreamReply = Callable[[ResolvedProfile, float, list[dict]], Iterator[StreamItem]]

RunKind = Literal["primary", "subagent"]
RunPhase = Literal[
    "starting",
    "streaming",
    "tool",
    "waiting",
    "command",
    "completed",
    "failed",
    "stopped",
]
RunExit = Literal["completed", "failed", "stopped"]
CompletionExit = Literal["reply", "failed", "stopped"]
AuthorizationAction = Literal["run", "declined", "blocked"]

_EXIT_PHASE: dict[RunExit, RunPhase] = {
    "completed": "completed",
    "failed": "failed",
    "stopped": "stopped",
}


def parse_errors_message(errors: Sequence[ToolProtocolError]) -> str:
    """The model-facing diagnostic for a reply the runtime refuses to execute."""
    # The headline names neither a cause nor a remedy. A batch is either all
    # syntax errors or all wrapped calls that are themselves well-formed, and
    # the two need opposite corrections: re-emit the call, or stop trying to
    # emit it at all because it was only ever an example. Prescribing one here
    # overrides the other — telling a model to "emit the calls again" is how a
    # requested syntax example becomes a command that runs. Each error carries
    # its own reason and its own remedy; this only says nothing ran.
    lines = [
        "Tool protocol error: no call from this response was executed. "
        "Address each issue below:"
    ]
    lines.extend(f"- line {error.line}: {error.message}" for error in errors)
    return "\n".join(lines)


def duplicate_keys_message(keys: Sequence[str]) -> str:
    """The model-facing diagnostic for one reply dispatching a key twice."""
    return (
        "Tool protocol error: one response cannot dispatch the same agent key "
        f"twice ({', '.join(keys)}). One agent key is one conversation and "
        "cannot have two concurrent writers. No call from the response was "
        "executed. Dispatch each key once, and send a follow-up task only "
        "after its result arrives."
    )


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """One dispatched task's result, as the coordinator receives it."""

    agent_key: str
    agent_label: str
    task_id: str
    task_label: str
    status: Literal["completed", "failed"]
    content: str

    def payload(self) -> str:
        """The framed observation appended to the coordinator's context.

        JSON for the variable data, built field by field in this type's own
        order: agent text is only ever the string value of ``content``, so no
        model output can forge an envelope field or a second result.
        """
        body = json.dumps(
            {
                "agent_key": self.agent_key,
                "agent_label": self.agent_label,
                "task_id": self.task_id,
                "task_label": self.task_label,
                "status": self.status,
                "content": self.content,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"[JTECH agent result]\n{body}"

    def display(self) -> str:
        """The one-line presentation form shown in the coordinator's stream."""
        return f"{self.agent_label} {self.status}: {self.task_label}"


@dataclass(frozen=True, slots=True)
class CommandAuthorization:
    """The host's decision about one shell command for one run."""

    action: AuthorizationAction
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """How one whole run ended.

    Raises:
        ValueError: on an internally inconsistent outcome — a completed run
            with no answer, or a failure with no reason. An invalid state is
            reported, never degraded into empty output.
    """

    status: RunExit
    final_text: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if self.status == "completed" and not self.final_text.strip():
            raise ValueError("a completed run must carry its final response")
        if self.status == "failed" and not self.error.strip():
            raise ValueError("a failed run must carry its failure description")
        if self.status == "stopped" and (self.final_text or self.error):
            raise ValueError("a stopped run carries no response and no error")
        if self.status != "completed" and self.final_text:
            raise ValueError("only a completed run carries a final response")
        if self.status != "failed" and self.error:
            raise ValueError("only a failed run carries a failure description")


@dataclass(frozen=True, slots=True)
class _CompletionOutcome:
    """How one model completion ended.

    ``reply`` deliberately admits empty text: an empty response is the
    documented nudge trigger, not a completed run, so the loop — not this
    type — decides what an empty answer means.

    Raises:
        ValueError: on an internally inconsistent outcome.
    """

    status: CompletionExit
    text: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if self.status == "failed" and not self.error.strip():
            raise ValueError("a failed completion must carry its failure description")
        if self.status == "stopped" and (self.text or self.error):
            raise ValueError("a stopped completion carries no text and no error")
        if self.status != "reply" and self.text:
            raise ValueError("only a reply completion carries text")
        if self.status != "failed" and self.error:
            raise ValueError("only a failed completion carries a failure description")


@dataclass(slots=True)
class AgentRunState:
    """Everything one run owns, and nothing another run may touch.

    The injected callables are evaluated at each completion boundary rather
    than captured once, so a global prompt, reasoning, or debug change applies
    to the next round instead of mutating a stream already in progress.
    """

    agent_key: str
    agent_label: str
    kind: RunKind
    session: Session
    transcript: Transcript
    profile: ResolvedProfile
    temperature: float
    system_prompt: Callable[[], str]
    reasoning_mode: Callable[[], str]
    debug_level: Callable[[], str]
    phase: RunPhase = "starting"
    prompt_tokens: int = 0
    last_reply: str = ""
    generating: bool = False
    tool_rounds_active: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    running_proc: subprocess.Popen[str] | None = None
    command_interrupted: bool = False


class RuntimeHost(Protocol):
    """The app-wide authority one runtime defers to.

    Deliberately three methods: policy and approval, dispatch, and one change
    notification. Everything else a run needs, it owns.
    """

    async def authorize_command(
        self, run: AgentRunState, command: str
    ) -> CommandAuthorization:
        """Decide one command for ``run``, prompting the user if the policy asks."""
        ...

    async def dispatch_agents(
        self, run: AgentRunState, calls: tuple[AgentDispatch, ...]
    ) -> tuple[AgentOutcome, ...]:
        """Run one whole dispatch batch and return its results in call order."""
        ...

    def runtime_changed(self, run: AgentRunState) -> None:
        """Note that ``run`` mutated one of its observable fields."""
        ...


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


def _kill_command_group(proc: subprocess.Popen[str]) -> None:
    """Kill the whole process group ``proc`` leads, not just the shell.

    ``bash -c`` is a shell, so a pipeline stage, a background job, or any other
    descendant survives a signal aimed only at the shell's own pid — and keeps
    reading and writing the project after the CLI reported the command
    stopped, or after the CLI itself exited. It also keeps the command's
    stdout pipe open, so the parent's ``communicate()`` blocks on a command
    the user already stopped.

    Commands are started with ``start_new_session=True``, so the shell is a
    session and group leader and its pid *is* its group id: nothing outside the
    command can be in the group this signals.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The group is already gone. Nothing survived it, so nothing to stop.
        pass


async def _abandon_launch(launch: asyncio.Future[subprocess.Popen[str]]) -> None:
    """Stop a command whose owner was cancelled while it was still starting.

    Cancelling ``asyncio.to_thread()`` abandons the *await*, not the thread:
    the ``Popen`` still completes, and this coroutine was the only thing that
    would ever have owned the child. Without this, an exit landing inside the
    launch window leaves a command running with nobody responsible for it.
    """
    try:
        proc = await launch
    except OSError:
        # The child was never created, so there is nothing to stop. The error
        # itself has no reader left: the run that asked for the command is
        # already unwinding.
        return
    _kill_command_group(proc)
    await asyncio.to_thread(proc.wait)


class AutonomousRuntime:
    """One conversation's unlimited model/command loop and its rendering."""

    def __init__(
        self,
        state: AgentRunState,
        *,
        host: RuntimeHost,
        stream_reply_fn: StreamReply,
        cmd_policy: CmdPolicy,
        project_root: Path,
    ) -> None:
        self._state = state
        self._host = host
        self._stream_reply_fn = stream_reply_fn
        self._cmd = cmd_policy
        self._project_root = project_root

    @property
    def state(self) -> AgentRunState:
        """This run's mutable state. Only its owner and the host read it."""
        return self._state

    # ------------------------------------------------------------- lifecycle

    async def run(self) -> RunOutcome:
        """Run until final prose, a typed failure, or an explicit stop.

        The caller has already recorded the user message or task that starts
        this run; this requests the first completion and applies the loop.
        """
        try:
            first = await self._stream_once()
            outcome = await self._tool_rounds(first)
        except asyncio.CancelledError:
            # Application teardown. Release this run's external work and let the
            # cancellation unwind untouched: a status write here has no reader
            # left, and an exception raised by one would replace the
            # cancellation the caller must see.
            self.request_stop()
            raise
        except Exception:
            self._set_phase("failed")
            raise
        self._set_phase(_EXIT_PHASE[outcome.status])
        return outcome

    def request_stop(self) -> None:
        """Stop this run's live external work: its stream, or its command.

        Those are the only resources a run holds outside the event loop, so
        this is the whole of stopping one — for ``Esc`` on Primary and for
        application exit alike.
        """
        state = self._state
        if state.generating:
            state.stop_event.set()
        elif state.running_proc is not None:
            state.command_interrupted = True
            _kill_command_group(state.running_proc)

    # ------------------------------------------------------------- tool loop

    async def _tool_rounds(self, first: _CompletionOutcome) -> RunOutcome:
        """Continue the conversation until one reply is a final answer.

        Tool output must reach the model before its next decision, so this loop
        owns the only normal exit: a reply with non-whitespace text and no
        recognized tool call. Any call — however often it repeats, and whatever
        it produced — costs another round, and an empty reply is nudged rather
        than treated as an answer. There is no round, repetition, retry, or
        elapsed-time budget anywhere in this control flow.
        """
        self._set_tool_rounds_active(True)
        try:
            outcome = first
            while True:
                if outcome.status == "stopped":
                    return RunOutcome("stopped")
                if outcome.status == "failed":
                    return RunOutcome("failed", error=outcome.error)
                reply = outcome.text
                parsed = parse_jtech_reply(reply)
                if parsed.errors:
                    # Atomic: a reply with any diagnostic executes none of its
                    # calls, so a malformed batch is never half-run.
                    self._record_protocol_error(parse_errors_message(parsed.errors))
                elif parsed.commands and parsed.dispatches:
                    self._record_protocol_error(MIXED_TOOLS_ERROR)
                elif parsed.commands:
                    await self._process_commands(parsed.commands)
                elif parsed.dispatches:
                    refusal = self._dispatch_refusal(parsed.dispatches)
                    if refusal is not None:
                        self._record_protocol_error(refusal)
                    else:
                        outcomes = await self._host.dispatch_agents(
                            self._state, tuple(parsed.dispatches)
                        )
                        self._record_agent_outcomes(outcomes)
                elif reply.strip():
                    return RunOutcome("completed", final_text=reply)
                else:
                    outcome = await self._nudge()
                    continue
                outcome = await self._stream_once()
        finally:
            self._set_tool_rounds_active(False)

    def _dispatch_refusal(self, dispatches: Sequence[AgentDispatch]) -> str | None:
        """Why this batch cannot run at all, or ``None`` when it may.

        Checked before any view, task, session, or model request exists, so a
        refused batch leaves nothing behind.
        """
        if self._state.kind != "primary":
            return SUBAGENT_DISPATCH_ERROR
        duplicates = duplicate_agent_keys(dispatches)
        if duplicates:
            return duplicate_keys_message(duplicates)
        return None

    async def _nudge(self) -> _CompletionOutcome:
        """Request a continuation after the model returned an empty reply."""
        state = self._state
        if state.debug_level() == "system":
            # Keep an auditable event without feeding it into future prompts.
            self._record_message(
                "system",
                NUDGE_PROMPT,
                include_in_context=False,
                debug_only=True,
            )
            self._show("system", NUDGE_PROMPT)
        with state.session.ephemeral("system", NUDGE_PROMPT):
            return await self._stream_once()

    # ------------------------------------------------------------- streaming

    async def _stream_once(self) -> _CompletionOutcome:
        """Consume one provider stream into this run's transcript.

        The request is prepared first, before a single piece of live state
        exists. Composing the prompt is fallible, and a failure after the
        generating flag, the AI entry, and the label timer were opened would
        latch all three: a bubble stuck on "live" forever and a run that never
        releases its turn. Nothing that can fail is done after that point
        without a ``finally`` that closes it.
        """
        state = self._state
        try:
            # Snapshotted on the event loop: a provider thread must never read
            # a live Session.messages list the loop can mutate underneath it.
            messages = state.session.messages_with_system(state.system_prompt())
        except (PromptResourceError, PromptSourceError) as error:
            failure = f"{PROMPT_ERROR}\n\n{error}"
            state.transcript.append(
                TranscriptRecord(role="system", content=failure, error=True)
            )
            return _CompletionOutcome("failed", error=failure)

        parts: list[str] = []
        timings: dict | None = None
        error: Exception | None = None
        started = time.monotonic()
        mode = state.reasoning_mode()
        content_chars = 0
        reasoning_chars = 0
        got_item = False
        got_content = False
        tick_count = 0
        reason_visible = False
        reason_dropped = False
        reason_tail = ""
        reason_text = Text()
        state.stop_event.clear()
        self._set_phase("streaming")
        self._set_generating(True)

        chat = state.transcript
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

        # The timer belongs to the transcript it repaints, so it is scoped to
        # one run's own widget rather than to the whole app.
        timer = chat.set_interval(1.0, tick)
        inbox = _StreamInbox(asyncio.get_running_loop())
        # This stream's own stop signal, separate from the run's stop flag: it
        # says "nobody is reading the inbox any more", which is true on every
        # exit, not only the ones the user asked for.
        abandoned = threading.Event()
        profile = state.profile
        temperature = state.temperature

        def consume() -> None:
            """Provider thread: enqueue every event, then one end marker."""
            producer_error: Exception | None = None
            try:
                for item in self._stream_reply_fn(profile, temperature, messages):
                    inbox.put(item)
                    if abandoned.is_set() or state.stop_event.is_set():
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
                                self._set_prompt_tokens(payload["prompt_tokens"])
                            elif kind == "timings":
                                timings = payload
                                prompt_n = payload.get("prompt_n")
                                if prompt_n:
                                    self._set_prompt_tokens(int(prompt_n))
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
                # would latch ``generating`` and queue every later message
                # behind a turn that can never finish, so it is reported like
                # any other stream failure and the completion ends.
                render_error = exc
            finally:
                timer.stop()

            await ai_stream.stop()
            async with ai_md.lock:
                pass
            if state.stop_event.is_set():
                drop_reason_bubble()
                chat.remove(ai_entry)
                self._show("system", "Generation stopped.")
                return _CompletionOutcome("stopped")
            failure = error if render_error is None else render_error
            if failure is not None:
                headline = CONNECTION_ERROR if render_error is None else RENDER_ERROR
                error_text = f"{headline}\n\n{failure}"
                # A /clear during the turn already closed these handles; the
                # failure still ends the completion, it just has nothing to paint.
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
                return _CompletionOutcome("failed", error=error_text)
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
            self._set_last_reply(reply)
            return _CompletionOutcome("reply", text=reply)
        finally:
            abandoned.set()
            try:
                # Every path but a render failure has already seen _StreamEnd,
                # so the producer is done. That one has not: releasing the run
                # with it still running would let the next request overlap it
                # and grow an inbox nobody drains. Join off the event loop —
                # Escape already makes a run wait for its provider this way.
                if producer.is_alive():
                    await asyncio.to_thread(producer.join)
            finally:
                self._set_generating(False)

    # -------------------------------------------------------------- commands

    async def _process_commands(self, commands: Sequence[str]) -> None:
        """Run every parsed command in source order, feeding each result back.

        Sequential within one run by design: one conversation's commands have
        an order the model chose. Separate runs execute concurrently.
        """
        self._set_phase("tool")
        for command in commands:
            if not command.strip():
                self._note_command(command, "empty command — not run")
                continue
            authorization = await self._host.authorize_command(self._state, command)
            if authorization.action == "blocked":
                self._note_command(command, f"blocked — {authorization.reason}")
                continue
            if authorization.action == "declined":
                self._note_command(command, authorization.reason)
                self._add_system(COMMAND_DECLINED_PROMPT)
                continue
            # Opened before the process is awaited: an authorized command has
            # to look acknowledged from the moment it starts. Until it exits,
            # nothing else in this run's own stream says it is running.
            entry = self._state.transcript.begin_markdown(
                "system", f"$ {command}\n\nrunning…"
            )
            try:
                result = await self._exec_command(command)
                presentation = self._cmd_bubble(command, result)
                # Finalizing records the value; it does not redraw the widget,
                # and compaction stops behind any removable notice still ahead
                # of this entry, so the body is what has to be updated or the
                # visible tail keeps saying "running…" after the process ended.
                # Gated on the state because /clear may have unmounted the body
                # while the command was still running.
                if entry.state == "live":
                    await entry.body.update(presentation)
            except asyncio.CancelledError:
                # Teardown. The placeholder is presentation only, so it leaves
                # with the run rather than being finalized into a result the
                # command never reported.
                self._state.transcript.remove(entry)
                raise
            self._state.transcript.finalize(
                entry, TranscriptRecord(role="system", content=presentation)
            )
            self._add_system(format_result(command, result=result))

    def _note_command(self, command: str, note: str) -> None:
        self._show("system", f"$ {command}\n→ {note}")
        self._add_system(format_result(command, note=note))

    async def _exec_command(self, command: str) -> ExecResult:
        """Run one command in a worker; a stop keeps the output it produced.

        There is no elapsed-time deadline. The command ends when it exits,
        when the user stops it, or when this run is cancelled: a build or a
        test suite takes as long as it takes.
        """
        state = self._state
        self._set_phase("command")
        state.command_interrupted = False
        try:
            # Kept as a task, not a bare await: cancellation must be able to
            # come back for the child this is about to create.
            launch = asyncio.ensure_future(
                asyncio.to_thread(
                    functools.partial(
                        subprocess.Popen,
                        ["bash", "-c", command],
                        cwd=self._project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        # Its own session, so the command owns a process group
                        # this runtime can stop whole. Without it, stopping the
                        # shell orphans everything the command started.
                        start_new_session=True,
                    )
                )
            )
            try:
                # Shielded, so a cancellation arriving mid-launch reaches this
                # coroutine while the launch itself still finishes and stays
                # claimable below.
                proc = await asyncio.shield(launch)
            except asyncio.CancelledError:
                # Shielded again: a second cancellation while unwinding must
                # not be what leaves a command running.
                await asyncio.shield(_abandon_launch(launch))
                raise
            except OSError as error:
                return ExecResult(127, str(error))
            self._set_running_proc(proc)
            try:
                try:
                    out, _ = await asyncio.to_thread(proc.communicate)
                except asyncio.CancelledError:
                    # Cancelling the await does not stop the child, and the
                    # ``finally`` below is about to release ownership of it —
                    # after which ``request_stop()`` has nothing left to find.
                    # This is the last place that can stop the command, so it
                    # does not delegate the job to whoever cancelled it.
                    #
                    # The kill is synchronous and lands before this yields:
                    # the command must not depend on this coroutine being
                    # resumed, or on the loop still scheduling work, to die.
                    # Only the reap waits, shielded so a second cancellation
                    # cannot skip it.
                    _kill_command_group(proc)
                    await asyncio.shield(asyncio.to_thread(proc.wait))
                    raise
                out = out or ""
                if state.command_interrupted:
                    state.command_interrupted = False
                    text, truncated = truncate_output(out, self._cmd.max_output)
                    return ExecResult(130, text, interrupted=True, truncated=truncated)
                text, truncated = truncate_output(out, self._cmd.max_output)
                return ExecResult(proc.returncode, text, truncated=truncated)
            finally:
                self._set_running_proc(None)
        finally:
            self._set_phase("tool")

    @staticmethod
    def _cmd_bubble(command: str, result: ExecResult) -> str:
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

    # ------------------------------------------------------------- recording

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
        """Persist one message to this run's session, surfacing an I/O failure.

        Presenting the failure is UI work, so it lives here rather than in
        ``Session``. The message stays in memory when the append fails — a save
        failure is worth saying out loud, but it is not a reason to drop the
        live conversation — and the warning never joins model context.
        """
        try:
            self._state.session.add(
                role,
                content,
                include_in_context=include_in_context,
                debug_only=debug_only,
                model_role=model_role,
                model_content=model_content,
            )
        except OSError as error:
            self._show("system", f"Could not save history: {error}")

    def _show(self, role: str, content: str) -> None:
        """Add one already-complete message to this run's own transcript."""
        self._state.transcript.append(TranscriptRecord(role=role, content=content))

    def _add_system(self, content: str) -> None:
        """Persist a visible runtime event with model-facing observation framing."""
        self._record_message(
            "system",
            content,
            model_role="user",
            model_content=f"[JTECH runtime event]\n{content}",
        )
        if self._state.debug_level() == "system":
            self._show("system", content)

    def _record_protocol_error(self, content: str) -> None:
        """Report a refused tool reply to the model.

        A protocol error is a runtime event with the same framing and the same
        visibility as any other: the model always sees it, and the transcript
        shows it at the system debug level.
        """
        self._add_system(content)

    def _record_agent_outcomes(self, outcomes: Sequence[AgentOutcome]) -> None:
        """Append one result observation per dispatched call, in call order.

        Each is a ``system`` message for presentation and a ``user`` message for
        the model, so the full worker answer survives in the coordinator's
        context and durable history while the reader inspects the agent's own
        transcript instead of a wall of relayed text.
        """
        for outcome in outcomes:
            self._record_message(
                "system",
                outcome.display(),
                debug_only=True,
                model_role="user",
                model_content=outcome.payload(),
            )
            if self._state.debug_level() == "system":
                self._show("system", outcome.display())

    # ---------------------------------------------------------- state changes

    def _notify(self) -> None:
        self._host.runtime_changed(self._state)

    def _set_phase(self, phase: RunPhase) -> None:
        self._state.phase = phase
        self._notify()

    def _set_generating(self, generating: bool) -> None:
        self._state.generating = generating
        self._notify()

    def _set_tool_rounds_active(self, active: bool) -> None:
        self._state.tool_rounds_active = active
        self._notify()

    def _set_prompt_tokens(self, tokens: int) -> None:
        self._state.prompt_tokens = tokens
        self._notify()

    def _set_last_reply(self, reply: str) -> None:
        self._state.last_reply = reply
        self._notify()

    def _set_running_proc(self, proc: subprocess.Popen[str] | None) -> None:
        self._state.running_proc = proc
        self._notify()


__all__ = [
    "CONNECTION_ERROR",
    "MIXED_TOOLS_ERROR",
    "RENDER_ERROR",
    "SPINNER_FRAMES",
    "SUBAGENT_DISPATCH_ERROR",
    "AgentOutcome",
    "AgentRunState",
    "AutonomousRuntime",
    "CommandAuthorization",
    "RunOutcome",
    "RuntimeHost",
    "StreamReply",
    "duplicate_keys_message",
    "parse_errors_message",
]
