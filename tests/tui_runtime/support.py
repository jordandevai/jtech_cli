"""Helpers shared by more than one module of the runtime test package."""

import asyncio
import pathlib

from textual.app import App, ComposeResult

from jtech_cli.cmd_tools import AgentDispatch, CmdPolicy
from jtech_cli.config import ResolvedProfile
from jtech_cli.session import Session
from jtech_cli.tui_runtime import (
    AgentOutcome,
    AgentRunState,
    AutonomousRuntime,
    CommandAuthorization,
)
from jtech_cli.tui_widgets import MarkdownTail, Transcript


PROFILE = ResolvedProfile(
    name="local", base_url="http://host:9000/v1", model="qwen3", api_key="none"
)


class Harness(App):
    """A minimal host app: one mounted transcript and nothing else."""

    def compose(self) -> ComposeResult:
        yield Transcript(id="chat")


class FakeHost:
    """A `RuntimeHost` that records every call and answers from a script."""

    def __init__(
        self,
        *,
        authorization: CommandAuthorization | None = None,
        outcomes: tuple[AgentOutcome, ...] = (),
    ) -> None:
        self.authorization = authorization or CommandAuthorization("run")
        self.outcomes = outcomes
        self.authorized: list[str] = []
        self.dispatched: list[tuple[AgentDispatch, ...]] = []
        self.phases: list[str] = []
        self.changes = 0

    async def authorize_command(self, run, command):
        self.authorized.append(command)
        return self.authorization

    async def dispatch_agents(self, run, calls):
        self.dispatched.append(calls)
        return self.outcomes

    def runtime_changed(self, run):
        self.changes += 1
        if not self.phases or self.phases[-1] != run.phase:
            self.phases.append(run.phase)


class BlockingReplyStream:
    """A `ReplyStream` that emits one item and then never produces another.

    This is the shape the fix exists for: a reader parked in a response that
    will not end on its own. Nothing releases it but cancellation, so a test
    that stops this stream is exercising the real path rather than a gate it
    opened itself.
    """

    def __init__(self, first="partial"):
        self.first = first
        self.blocked = asyncio.Event()  # set once the reader is parked
        self.cancelled = False
        self.closed = 0
        self.iterated = False

    def __aiter__(self):
        return self._items()

    async def _items(self):
        self.iterated = True
        yield self.first
        self.blocked.set()
        try:
            await asyncio.Event().wait()  # never set by anyone
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def aclose(self):
        self.closed += 1


def reply_stream_factory(stream):
    """A `StreamReply` handing out one prepared stream."""

    async def factory(profile, temperature, messages):
        return stream

    return factory


async def wait_for(pilot, predicate, *, tries=100, pause=0.05):
    """Settle the app until ``predicate()`` holds, or fail the test."""
    for _ in range(tries):
        await pilot.pause(pause)
        if predicate():
            return
    raise AssertionError("condition not met in time")


class ScriptedReplyStream:
    """A finite `ReplyStream` over already-known items."""

    def __init__(self, *items):
        self.items = items
        self.closed = 0

    def __aiter__(self):
        return self._items()

    async def _items(self):
        for item in self.items:
            yield item

    async def aclose(self):
        self.closed += 1


def scripted_stream(*replies):
    """A stream fake yielding one scripted reply per completion, in order.

    An exhausted script raises rather than answering. The empty reply this used
    to fall back to is the runtime's nudge trigger, so a script one entry short
    fed the nudge loop forever instead of failing the test.
    """
    calls = {"n": 0}

    async def fake(profile, temperature, messages):
        index = calls["n"]
        calls["n"] += 1
        if index >= len(replies):
            raise AssertionError(
                f"the scripted stream is exhausted: no reply for completion "
                f"{index + 1} of {len(replies)}"
            )
        reply = replies[index]
        if isinstance(reply, Exception):
            raise reply
        return ScriptedReplyStream(reply)

    return fake, calls


def _protocol_block(name: str, body: str) -> str:
    """Frame one raw body in the production block delimiters."""
    return f"[[[{name}]]]\n{body}\n[[[/{name}]]]"


def command_call(command: str) -> str:
    """Format a test command using the production command-only protocol.

    The command is wrapped, never quoted: the point of the block protocol is
    that a payload needs no escaping, so a helper that escaped one would test a
    wire format the model is never asked to produce.
    """
    return _protocol_block("jtech_cmd", command)


def result_call(status: str, content: str) -> str:
    """Format one subagent terminal result using the production protocol."""
    return _protocol_block("jtech_result", f"status: {status}\n\n{content}")


def dispatch_call(key="coder", label="Coder", profile="local", task_label="t", task="x"):
    """Format one dispatch using the production protocol."""
    return _protocol_block(
        "jtech_agent",
        f"agent_key: {key}\n"
        f"agent_label: {label}\n"
        f"profile_name: {profile}\n"
        f"task_label: {task_label}\n"
        f"\n{task}",
    )


def make_state(
    app,
    *,
    kind="primary",
    session=None,
    debug_level="none",
    reasoning="hide",
    profile=None,
) -> AgentRunState:
    return AgentRunState(
        agent_key="primary" if kind == "primary" else "coder",
        agent_label="Primary" if kind == "primary" else "Coder",
        kind=kind,
        session=session or Session(persist=False),
        transcript=app.query_one("#chat", Transcript),
        profile=profile or PROFILE,
        temperature=0.7,
        system_prompt=lambda: "SYSTEM",
        reasoning_mode=lambda: reasoning,
        debug_level=lambda: debug_level,
    )


def make_runtime(app, stream, *, host=None, cmd=None, root=None, **state_kwargs):
    host = host or FakeHost()
    runtime = AutonomousRuntime(
        make_state(app, **state_kwargs),
        host=host,
        stream_reply_fn=stream,
        cmd_policy=cmd or CmdPolicy(mode="yolo"),
        project_root=root or pathlib.Path.cwd(),
    )
    return runtime, host


def transcript_text(app) -> str:
    chat = app.query_one("#chat", Transcript)
    return "\n".join(record.content for record in chat.history.records)


def live_entries(app) -> list[tuple[str, str]]:
    """Label and *currently drawn* body of every live tail entry, in order.

    Reads the widgets rather than the records: what a finalized entry recorded
    and what it is still showing are exactly what can disagree here.
    """
    return [
        (
            str(entry.label.render()),
            entry.body._markdown
            if isinstance(entry, MarkdownTail)
            else str(entry.body.render()),
        )
        for entry in app.query_one("#chat", Transcript)._tail
    ]


def model_messages(session: Session) -> list[dict]:
    return session.messages_with_system("")


async def wait_for_shell(runtime, pilot, tries=100):
    """Wait until the run owns a live shell process."""
    for _ in range(tries):
        await pilot.pause(0.05)
        proc = runtime.state.running_proc
        if proc is not None:
            return proc
    raise AssertionError("the command never started")
